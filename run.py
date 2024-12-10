# gdn_trainer.py 
from pathlib import Path
import time
import logging
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from dataclasses import dataclass
import pandas as pd
from torch.utils.data import random_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
from tqdm import tqdm
import time
from datetime import timedelta
from util.preprocess import construct_data, build_loc_net
from util.net_struct import get_feature_map, get_fc_graph_struc
from datasets.TimeDataset import TimeDataset

# Default configuration with commonly used values
@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    # Data params
    dataset: str = "msl"  # Dataset name
    data_path: str = "data/"  # Base data directory
    batch_size: int = 512
    val_ratio: float = 0.2
    slide_win: int = 25
    slide_stride: int = 5
    
    # Model params
    embed_dim: int = 64
    out_layer_num: int = 1 
    out_layer_inter_dim: int = 256
    topk: int = 5
    
    # Training params
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0
    early_stopping_patience: int = 15
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output paths
    model_dir: str = "models/"
    log_dir: str = "logs/"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        self.model_path = f"{self.model_dir}/gdn_{self.dataset}.pt"

class Timer:
    """Simple timer for tracking execution time."""
    def __init__(self, name="Task"):
        self.name = name
        self.start_time = None
        self.end_time = None
        
    def __enter__(self):
        self.start_time = time.time()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed_time = timedelta(seconds=int(self.end_time - self.start_time))
        print(f"\n{self.name} completed in {elapsed_time}")

class Metrics:
    """Handles metric calculation for anomaly detection."""
    
    @staticmethod
    def get_err_scores(predicted, target):
        """Calculate error scores between predicted and target values."""
        # Get error scores for each feature dimension
        errors = []
        for i in range(predicted.shape[1]):
            # Calculate absolute error
            err = np.abs(predicted[:, i] - target[:, i])
            
            # Calculate error stats for normalization
            err_median = np.median(err)
            err_iqr = np.percentile(err, 75) - np.percentile(err, 25)
            
            # Normalize error scores
            err_scores = (err - err_median) / (err_iqr + 1e-2)
            errors.append(err_scores)
            
        # Combine errors across features (take maximum)
        combined_errors = np.max(np.array(errors), axis=0)
        
        # Apply smoothing
        window = 3
        smoothed_scores = np.zeros_like(combined_errors)
        for i in range(window, len(combined_errors)):
            smoothed_scores[i] = np.mean(combined_errors[i-window:i+1])
            
        return smoothed_scores

    @staticmethod
    def get_best_threshold(scores, true_labels):
        """Find best threshold using F1 score."""
        best_f1 = 0
        best_threshold = 0
        
        # Ensure labels are binary
        true_labels = true_labels.astype(int)
        if len(true_labels.shape) > 1:
            true_labels = true_labels[:, 0]  # Take first column if multi-dimensional
            
        # Try different percentiles as thresholds
        for percentile in range(1, 100):
            threshold = np.percentile(scores, percentile)
            pred_labels = (scores > threshold).astype(int)
            
            try:
                f1 = f1_score(true_labels, pred_labels)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            except Exception:
                continue
                
        return best_threshold

    @staticmethod
    def calculate_metrics(scores, threshold, true_labels):
        """Calculate all metrics using the determined threshold."""
        # Ensure labels are binary
        true_labels = true_labels.astype(int)
        if len(true_labels.shape) > 1:
            true_labels = true_labels[:, 0]
            
        pred_labels = (scores > threshold).astype(int)
        
        return {
            'f1': f1_score(true_labels, pred_labels),
            'precision': precision_score(true_labels, pred_labels),
            'recall': recall_score(true_labels, pred_labels),
            'auc': roc_auc_score(true_labels, scores)
        }

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience: int = 7, min_delta: float = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.should_stop

class GDNTrainer:
    """Main trainer class for Graph Deviation Network."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self._setup_logging()
        self._setup_data()
        self._setup_model()
        
    def _setup_logging(self):
        """Configure logging and tensorboard."""
        log_file = f"{self.config.log_dir}/gdn_{self.config.dataset}_{time.strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.writer = SummaryWriter(f"runs/gdn_{self.config.dataset}_{time.strftime('%Y%m%d_%H%M%S')}")
        logging.info(f"Initialized training with config:\n{self.config}")
        
    def _setup_data(self):
        """Load and prepare datasets."""
        
        # Load data
        train_data = pd.read_csv(f"{self.config.data_path}/{self.config.dataset}/train.csv", index_col=0)
        test_data = pd.read_csv(f"{self.config.data_path}/{self.config.dataset}/test.csv", index_col=0)
        
        if 'attack' in train_data.columns:
            train_data = train_data.drop(columns=['attack'])
        
        # Get feature map and graph structure
        self.feature_map = get_feature_map(self.config.dataset)
        fc_struc = get_fc_graph_struc(self.config.dataset)
        edge_index = build_loc_net(fc_struc, list(train_data.columns), feature_map=self.feature_map)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long)
        
        # Prepare datasets
        train_dataset = TimeDataset(
            construct_data(train_data, self.feature_map),
            self.edge_index,
            mode='train',
            config={'slide_win': self.config.slide_win, 'slide_stride': self.config.slide_stride}
        )
        
        test_dataset = TimeDataset(
            construct_data(test_data, self.feature_map, labels=test_data.attack.tolist()),
            self.edge_index,
            mode='test',
            config={'slide_win': self.config.slide_win, 'slide_stride': self.config.slide_stride}
        )
    
    # Create train/val split
        from torch.utils.data import random_split
        val_size = int(len(train_dataset) * self.config.val_ratio)
        train_size = len(train_dataset) - val_size
        train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
        
        # Create data loaders
        self.train_loader = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True)
        self.val_loader = DataLoader(val_subset, batch_size=self.config.batch_size)
        self.test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size)
        
        # Get input dimension from data
        x, _, _, _ = next(iter(self.train_loader))
        self.input_dim = x.shape[-1]  # Get the last dimension of input
        self.num_nodes = len(self.feature_map)

    def _setup_model(self):
        """Initialize model, optimizer, and training components."""
        from models.GDN import GDN
        
        # Set random seeds
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            torch.backends.cudnn.deterministic = True
        
        self.device = torch.device(self.config.device)
        
    # Initialize model with correct dimensions
        self.model = GDN(
            edge_index_sets=[self.edge_index],
            node_num=self.num_nodes,
            dim=self.config.embed_dim,
            input_dim=self.input_dim,  # Add input dimension
            out_layer_num=self.config.out_layer_num,
            out_layer_inter_dim=self.config.out_layer_inter_dim,
            topk=self.config.topk
        ).to(self.device)
        
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        self.early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
        print(f"Model initialized with input_dim={self.input_dim}, num_nodes={self.num_nodes}")
        
    
    def train(self):
        """Main training loop with progress bars."""
        best_val_loss = float('inf')
        
        print("\nStarting GDN training...")
        print(f"Training on device: {self.device}")
        print(f"Number of epochs: {self.config.epochs}")
        print(f"Training batches per epoch: {len(self.train_loader)}")
        print(f"Validation batches per epoch: {len(self.val_loader)}")
        
        with Timer("Training"):
            for epoch in range(self.config.epochs):
                # Training phase
                self.model.train()
                train_loss = 0.0
                
                train_bar = tqdm(self.train_loader, 
                            desc=f'Epoch {epoch}/{self.config.epochs} [Train]',
                            leave=False)
                
                for x, y, labels, edge_index in train_bar:
                    x, y, labels, edge_index = [
                        item.float().to(self.device) for item in [x, y, labels, edge_index]
                    ]
                    
                    self.optimizer.zero_grad()
                    output = self.model(x, edge_index)
                    loss = F.mse_loss(output, y)
                    
                    loss.backward()
                    self.optimizer.step()
                    
                    train_loss += loss.item()
                    train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
                
                avg_train_loss = train_loss / len(self.train_loader)
                self.writer.add_scalar('Loss/train', avg_train_loss, epoch)
                
                # Validation phase
                val_loss, metrics = self.validate(epoch)
                
                # Logging
                log_msg = (f"Epoch {epoch}/{self.config.epochs} - "
                        f"Train Loss: {avg_train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}, "
                        f"Val MSE: {metrics['mse']:.6f}")
                
                if 'f1' in metrics:
                    log_msg += f", Val F1: {metrics['f1']:.4f}"
                
                logging.info(log_msg)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_model(epoch, val_loss)
                    logging.info(f"Saved best model with val_loss: {val_loss:.6f}")
                
                # Early stopping
                if self.early_stopping(val_loss):
                    logging.info(f"Early stopping triggered at epoch {epoch}")
                    break
                    
            self.writer.close()
            logging.info("Training completed!")

    def validate(self, epoch: int) -> Tuple[float, Dict]:
        """Validate the model with progress bar."""
        self.model.eval()
        val_loss = 0.0
        predictions = []
        targets = []
        batch_labels_list = []
        
        val_bar = tqdm(self.val_loader, 
                    desc=f'Epoch {epoch}/{self.config.epochs} [Val]',
                    leave=False)
        
        with torch.no_grad():
            for x, y, batch_labels, edge_index in val_bar:
                x, y, batch_labels, edge_index = [
                    item.float().to(self.device) for item in [x, y, batch_labels, edge_index]
                ]
                
                output = self.model(x, edge_index)
                loss = F.mse_loss(output, y)
                
                val_loss += loss.item()
                predictions.append(output.cpu().numpy())
                targets.append(y.cpu().numpy())
                batch_labels_list.append(batch_labels.cpu().numpy())
                
                val_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Concatenate all batches
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        labels = np.concatenate(batch_labels_list, axis=0)
        
        # Calculate error scores
        err_scores = Metrics.get_err_scores(predictions, targets)
        
        # Calculate base metrics
        avg_val_loss = val_loss / len(self.val_loader)
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        
        metrics = {
            'loss': avg_val_loss,
            'mse': mse,
            'mae': mae
        }
        
        # Add anomaly detection metrics if we have labels
        if labels.any():
            threshold = Metrics.get_best_threshold(err_scores, labels)
            detection_metrics = Metrics.calculate_metrics(err_scores, threshold, labels)
            metrics.update(detection_metrics)
        
        return avg_val_loss, metrics

    def test(self, model_path: Optional[str] = None):
        """Test the model with progress bar."""
        if model_path is None:
            model_path = self.config.model_path
            
        # Load best model
        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()
        predictions = []
        targets = []
        labels = []
        
        with Timer("Testing"):
            test_bar = tqdm(self.test_loader, desc='Testing', leave=True)
            
            with torch.no_grad():
                for x, y, batch_labels, edge_index in test_bar:
                    x, y, batch_labels, edge_index = [
                        item.float().to(self.device) for item in [x, y, batch_labels, edge_index]
                    ]
                    
                    output = self.model(x, edge_index)
                    loss = F.mse_loss(output, y)
                    
                    predictions.append(output.cpu().numpy())
                    targets.append(y.cpu().numpy())
                    labels.append(batch_labels.cpu().numpy())
                    
                    test_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        labels = np.concatenate(labels, axis=0)
        
        # Calculate error scores
        err_scores = Metrics.get_err_scores(predictions, targets)
        
        # Get best threshold
        threshold = Metrics.get_best_threshold(err_scores, labels)
        
        # Calculate final metrics
        metrics = Metrics.calculate_metrics(err_scores, threshold, labels)
        
        logging.info("\n=========================** Result **============================\n")
        logging.info(f"F1 score: {metrics['f1']:.4f}")
        logging.info(f"Precision: {metrics['precision']:.4f}")
        logging.info(f"Recall: {metrics['recall']:.4f}")
        logging.info(f"AUC: {metrics['auc']:.4f}\n")
        
        return metrics, err_scores, threshold
    
    def save_model(self, epoch: int, val_loss: float):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }, self.config.model_path)

# run.py
def main():
    """Main entry point for training/testing GDN."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train/Test GDN Model')
    parser.add_argument('--dataset', type=str, default='msl',
                      help='dataset name (default: msl)')
    parser.add_argument('--mode', choices=['train', 'test'], default='train',
                      help='train or test mode (default: train)')
    args = parser.parse_args()
    
    # Create config with default values, override dataset
    config = TrainingConfig(dataset=args.dataset)
    
    # Initialize trainer
    trainer = GDNTrainer(config)
    
    # Run training or testing
    if args.mode == 'train':
        trainer.train()
        # Run test after training
        trainer.test()
    else:
        trainer.test()

if __name__ == '__main__':
    main()