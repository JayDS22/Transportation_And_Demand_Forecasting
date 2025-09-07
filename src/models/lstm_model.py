import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error

class TransportationDemandDataset(Dataset):
    """Custom dataset for transportation demand forecasting"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMDemandPredictor(nn.Module):
    """
    LSTM model for transportation demand prediction
    """
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, 
                 dropout: float = 0.2, bidirectional: bool = False):
        super(LSTMDemandPredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * (2 if bidirectional else 1),
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output layers
        lstm_output_size = hidden_size * (2 if bidirectional else 1)
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 2, lstm_output_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size // 4, 1)
        )
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize model weights"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    
    def forward(self, x):
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Use the last time step output
        final_output = attn_out[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc_layers(final_output)
        
        return output.squeeze(), attn_weights

class LSTMTrainer:
    """
    Training class for LSTM demand predictor
    """
    
    def __init__(self, model: LSTMDemandPredictor, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.logger = logging.getLogger(__name__)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.get('learning_rate', 0.001),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Loss function with regularization
        self.criterion = nn.MSELoss()
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              num_epochs: int) -> Dict[str, list]:
        """
        Train the LSTM model
        """
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = self.config.get('early_stopping_patience', 10)
        
        self.model.train()
        
        for epoch in range(num_epochs):
            # Training phase
            epoch_train_loss = 0.0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output, _ = self.model(data)
                loss = self.criterion(output, target)
                
                # L2 regularization
                l2_reg = torch.tensor(0., device=self.device)
                for param in self.model.parameters():
                    l2_reg += torch.norm(param, 2)
                loss += self.config.get('l2_lambda', 1e-4) * l2_reg
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.get('grad_clip', 1.0)
                )
                
                self.optimizer.step()
                epoch_train_loss += loss.item()
            
            # Validation phase
            val_loss = self.validate(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Track losses
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
            
            if epoch % 10 == 0:
                self.logger.info(
                    f'Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}'
                )
            
            if patience_counter >= max_patience:
                self.logger.info(f'Early stopping at epoch {epoch}')
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_model.pth'))
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss
        }
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                val_loss += self.criterion(output, target).item()
        
        self.model.train()
        return val_loss / len(val_loader)
    
    def predict(self, test_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on test data"""
        self.model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                
                predictions.extend(output.cpu().numpy())
                actuals.extend(target.cpu().numpy())
        
        return np.array(predictions), np.array(actuals)
