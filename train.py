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
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime


def loss_func(y_pred, y_true):
    loss = F.mse_loss(y_pred, y_true, reduction='mean')

    return loss



def train_og(model = None, save_path = '', config={},  train_dataloader=None, val_dataloader=None, feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', train_dataset=None):

    seed = config['seed']

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])

    now = time.time()
    
    train_loss_list = []
    cmp_loss_list = []

    device = get_device()


    acu_loss = 0
    min_loss = 1e+8
    min_f1 = 0
    min_pre = 0
    best_prec = 0

    i = 0
    epoch = config['epoch']
    early_stop_win = 15

    model.train()

    log_interval = 1000
    stop_improve_count = 0

    dataloader = train_dataloader

    for i_epoch in range(epoch):

        acu_loss = 0
        model.train()

        for x, labels, attack_labels, edge_index in dataloader:
            _start = time.time()

            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)
            
            loss.backward()
            optimizer.step()

            
            train_loss_list.append(loss.item())
            acu_loss += loss.item()
                
            i += 1


        # each epoch
        print('epoch ({} / {}) (Loss:{:.8f}, ACU_loss:{:.8f})'.format(
                        i_epoch, epoch, 
                        acu_loss/len(dataloader), acu_loss), flush=True
            )

        # use val dataset to judge
        if val_dataloader is not None:

            val_loss, val_result = test(model, val_dataloader)

            if val_loss < min_loss:
                torch.save(model.state_dict(), save_path)

                min_loss = val_loss
                stop_improve_count = 0
            else:
                stop_improve_count += 1


            if stop_improve_count >= early_stop_win:
                print(f"Early stopping at epoch {i_epoch} due to no improvement in validation loss for {early_stop_win} consecutive epochs.")
                break

        else:
            if acu_loss < min_loss :
                torch.save(model.state_dict(), save_path)
                min_loss = acu_loss



    return train_loss_list

def train(model=None, save_path='', config={}, train_dataloader=None, val_dataloader=None, 
          feature_map={}, test_dataloader=None, test_dataset=None, dataset_name='swat', 
          train_dataset=None):
    """Enhanced training function for supervised GDN."""
    
    seed = config['seed']
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config['decay'])
    
    device = get_device()
    min_loss = float('inf')
    min_f1 = 0
    stop_improve_count = 0
    early_stop_win = 15

    # Create unique run directory
    run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join('runs', run_name)
    writer = SummaryWriter(log_dir)

    # Log hyperparameters
    writer.add_hparams(
        {
            'lr': 0.001,
            'weight_decay': config['decay'],
            'epochs': config['epoch'],
        },
        {'dummy': 0}  # Required placeholder metric
    )

    # Training loop
    for i_epoch in range(config['epoch']):
        print(f"\nDEBUG: Starting epoch {i_epoch}")
        model.train()
        acu_loss = 0
        
        for batch_idx, (x, labels, attack_labels, edge_index) in enumerate(train_dataloader):
            #print(f"\nDEBUG: Batch {batch_idx}")
            #print(f"Input shape: {x.shape}")
            #print(f"Labels shape: {labels.shape}")
            #print(f"Attack labels shape: {attack_labels.shape}")
            #print(f"Edge index shape: {edge_index.shape}")
            
            # Move everything to device and convert to float
            x, labels, attack_labels, edge_index = [
                item.float().to(device) for item in [x, labels, attack_labels, edge_index]
            ]
            
            optimizer.zero_grad()
            
            # Forward pass - now returns three outputs
            final_output, forecast_out, detection_out = model(x, edge_index)
            
            # Get total loss and components
            total_loss, forecast_loss, detection_loss = model.loss_function(
                final_output=final_output,
                forecast_out=forecast_out,
                detection_out=detection_out,
                true_values=labels,
                labels=attack_labels,
                alpha=0.5
            )
            
            # Log all loss components
            writer.add_scalar('Loss/batch_total', total_loss.item(), i_epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('Loss/batch_forecast', forecast_loss.item(), i_epoch * len(train_dataloader) + batch_idx)
            writer.add_scalar('Loss/batch_detection', detection_loss.item(), i_epoch * len(train_dataloader) + batch_idx)
            
            print(f"Loss components - Total: {total_loss.item():.4f}, "
                  f"Forecast: {forecast_loss.item():.4f}, "
                  f"Detection: {detection_loss.item():.4f}")
            
            if torch.isnan(total_loss):
                print("WARNING: Loss is NaN!")
                break
                
            total_loss.backward()
            optimizer.step()
            
            acu_loss += total_loss.item()

            # Log training metrics
            writer.add_scalar('Loss/train', total_loss.item(), i_epoch)

        # Log epoch average loss
        epoch_loss = acu_loss / len(train_dataloader)
        writer.add_scalar('Loss/epoch', epoch_loss, i_epoch)

        # Validation phase
        model.eval()
        if val_dataloader is not None:
            val_loss = 0
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                for x, labels, attack_labels, edge_index in val_dataloader:
                    x, labels, attack_labels, edge_index = [
                        item.float().to(device) for item in [x, labels, attack_labels, edge_index]
                    ]
                    
                    # During validation, we only need the final output
                    final_output = model(x, edge_index)
                    
                    # Store predictions and labels for F1 computation
                    val_predictions.extend(final_output.cpu().numpy())
                    val_labels.extend(attack_labels.cpu().numpy())
            
            # Calculate validation metrics
            # During validation
            val_predictions = np.array(val_predictions)
            # Print some statistics about predictions
            print(f"Validation predictions stats: min={val_predictions.min():.4f}, "
                f"max={val_predictions.max():.4f}, mean={val_predictions.mean():.4f}")
            # Use a dynamic threshold based on the validation set
            threshold = np.percentile(val_predictions,50 )  # Assume top 5% are anomalies
            val_predictions = val_predictions > threshold

            val_f1 = f1_score(val_labels, val_predictions)
            
            print(f'Epoch {i_epoch}: Train Loss: {epoch_loss/len(train_dataloader):.4f}, '
                  f'Val F1: {val_f1:.4f}')
            
            # Log validation metrics
            writer.add_scalar('Metrics/val_f1', val_f1, i_epoch)
            writer.add_scalar('Loss/val', val_loss, i_epoch)
            
    # Close writer at end
    writer.close()
                
    return min_f1
