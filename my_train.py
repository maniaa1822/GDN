import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from util.time import *
from util.env import *
from sklearn.metrics import mean_squared_error
from test import *
import torch.nn.functional as F
import numpy as np
from evaluate import get_best_performance_data, get_val_performance_data, get_full_err_scores
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
from torch.utils.data import DataLoader, random_split, Subset
from scipy.stats import iqr
import torch.cuda.profiler as profiler
from torch.cuda import Event
import psutil
from torch.utils.tensorboard import SummaryWriter

# Enable cudnn benchmarking
torch.backends.cudnn.benchmark = True


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss

# 1. Move data to GPU in batch
def prepare_batch(x, labels, attack_labels, edge_index, device):
    return (x.to(device), 
            labels.to(device), 
            attack_labels.to(device),
            edge_index.to(device))

# 2. Enable automatic mixed precision
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

def train(model=None, save_path='', config={}, train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):
    seed = config['seed']
    device = get_device()
    
    # Training settings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])
    scaler = torch.cuda.amp.GradScaler()
    accumulation_steps = 4  # New parameter
    writer = SummaryWriter(log_dir='runs/experiment1')
    
    # Tracking variables
    train_loss_list = []
    min_loss = 1e+8
    early_stop_win = 15
    stop_improve_count = 0
    
    # Timing metrics
    epoch_times = []
    batch_times = []
    forward_times = []
    backward_times = []

    # Diagnostic setup
    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)
    
    # Tracking metrics
    metrics = {
        'data_load_times': [],
        'gpu_transfer_times': [],
        'forward_times': [],
        'backward_times': [],
        'optimizer_times': [],
        'gpu_memory_used': [],
        'cpu_memory_used': [],
        'batch_total_times': []
    }

    for i_epoch in range(config['epoch']):
        torch.cuda.reset_peak_memory_stats()
        epoch_start = time.time()
        
        # Print initial memory state
        print(f"\nGPU Memory at epoch start: {torch.cuda.memory_allocated()/1e9:.2f}GB")
        print(f"CPU Memory at epoch start: {psutil.Process().memory_info().rss/1e9:.2f}GB")

        acu_loss = 0.0

        # 3. Optimize the training loop
        for i, (x, labels, attack_labels, edge_index) in enumerate(train_dataloader):
            with autocast():  # Enable AMP
                batch_start = time.time()
                x, labels, attack_labels, edge_index = prepare_batch(
                    x, labels, attack_labels, edge_index, device)
            
            # Data loading time
            data_load_start = time.time()
            x, labels, edge_index = [item.float() for item in [x, labels, edge_index]]
            metrics['data_load_times'].append(time.time() - data_load_start)
            
            # GPU transfer time
            start_event.record()
            x, labels, edge_index = [item.to(device) for item in [x, labels, edge_index]]
            end_event.record()
            torch.cuda.synchronize()
            metrics['gpu_transfer_times'].append(start_event.elapsed_time(end_event))
            
            # Forward pass timing
            forward_start = time.time()
            start_event.record()
            with torch.cuda.amp.autocast():
                out = model(x, edge_index).float()
                loss = loss_func(out, labels) / accumulation_steps
            end_event.record()
            torch.cuda.synchronize()
            forward_times.append(time.time() - forward_start)
            metrics['forward_times'].append(start_event.elapsed_time(end_event))
            
            # Backward pass timing
            backward_start = time.time()
            start_event.record()
            scaler.scale(loss).backward()
            end_event.record()
            torch.cuda.synchronize()
            backward_times.append(time.time() - backward_start)
            metrics['backward_times'].append(start_event.elapsed_time(end_event))
            
            # Track memory
            metrics['gpu_memory_used'].append(torch.cuda.memory_allocated()/1e9)
            metrics['cpu_memory_used'].append(psutil.Process().memory_info().rss/1e9)
            
            # Gradient accumulation
            if (i + 1) % accumulation_steps == 0:
                start_event.record()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                end_event.record()
                torch.cuda.synchronize()
                metrics['optimizer_times'].append(start_event.elapsed_time(end_event))
                writer.add_scalar('Loss/train', loss.item() * accumulation_steps, i_epoch * len(train_dataloader) + i)
            
            acu_loss += loss.item() * accumulation_steps
            train_loss_list.append(loss.item() * accumulation_steps)
            batch_times.append(time.time() - batch_start)
            metrics['batch_total_times'].append(time.time() - batch_start)
            
            # Print diagnostics every 100 batches
            if i % 100 == 0:
                print(f"\nBatch {i} diagnostics:")
                print(f"Data loading: {np.mean(metrics['data_load_times'][-100:]):.3f}ms")
                print(f"GPU transfer: {np.mean(metrics['gpu_transfer_times'][-100:]):.3f}ms")
                print(f"Forward pass: {np.mean(metrics['forward_times'][-100:]):.3f}ms")
                print(f"Backward pass: {np.mean(metrics['backward_times'][-100:]):.3f}ms")
                print(f"GPU Memory: {metrics['gpu_memory_used'][-1]:.2f}GB")
                print(f"Memory peaked: {torch.cuda.max_memory_allocated()/1e9:.2f}GB")

        # Validation and checkpointing
        if val_dataloader is not None:
            val_loss, val_result = test(model, val_dataloader)
            writer.add_scalar('Loss/val', val_loss, i_epoch)
            
            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)
                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1

            if stop_improve_count >= early_stop_win:
                print(f"Early stopping at epoch {i_epoch}")
                break
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # Print metrics
        print(f'Epoch {i_epoch}/{config["epoch"]}:')
        print(f'  Loss: {acu_loss/len(train_dataloader):.8f}')
        print(f'  Time: {epoch_time:.2f}s')
        print(f'  Avg batch time: {np.mean(batch_times):.3f}s')
        print(f'  Avg forward time: {np.mean(forward_times):.3f}s')
        print(f'  Avg backward time: {np.mean(backward_times):.3f}s')

        # End of epoch diagnostics
        print(f"\nEpoch {i_epoch} Summary:")
        for key, values in metrics.items():
            if values:
                print(f"{key}: {np.mean(values):.3f} ± {np.std(values):.3f}")
        writer.add_scalar('Time/epoch', epoch_time, i_epoch)
        writer.add_scalar('Memory/GPU', torch.cuda.memory_allocated()/1e9, i_epoch)
        writer.add_scalar('Memory/CPU', psutil.Process().memory_info().rss/1e9, i_epoch)

    writer.close()
    # Final timing statistics
    print(f'\nTraining completed in {sum(epoch_times):.2f}s')
    print(f'Average epoch time: {np.mean(epoch_times):.2f}s')

    return train_loss_list


