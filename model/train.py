import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.transformer import SmartHomeTransformer
from utils.data_preprocessing import load_and_preprocess_data, create_sequences

import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.transformer import SmartHomeTransformer
from utils.data_preprocessing import load_and_preprocess_data, create_sequences

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.transformer import SmartHomeTransformer
from utils.data_preprocessing import load_and_preprocess_data, create_sequences

import json
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model(data_path, seq_length=60, batch_size=32, num_epochs=10, learning_rate=0.001):
    try:
        # Create model directory if it doesn't exist
        os.makedirs('model', exist_ok=True)
        
        logger.info("Loading and preprocessing data...")
        # Load and preprocess data
        X_train, X_test, y_energy_train, y_energy_test, y_user_train, y_user_test, y_anomaly_train, y_anomaly_test = load_and_preprocess_data(data_path)
        
        logger.info("Creating sequences...")
        # Create sequences
        X_train_seq, y_energy_train_seq, y_user_train_seq, y_anomaly_train_seq = create_sequences(
            X_train, y_energy_train, y_user_train, y_anomaly_train, seq_length
        )
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_seq)
        y_energy_train_tensor = torch.FloatTensor(y_energy_train_seq).unsqueeze(-1)
        y_user_train_tensor = torch.LongTensor(y_user_train_seq)
        y_anomaly_train_tensor = torch.FloatTensor(y_anomaly_train_seq).unsqueeze(-1)
        
        # Print shapes for verification
        logger.info(f"Training data shapes:")
        logger.info(f"X_train_tensor: {X_train_tensor.shape}")
        logger.info(f"y_energy_train_tensor: {y_energy_train_tensor.shape}")
        logger.info(f"y_user_train_tensor: {y_user_train_tensor.shape}")
        logger.info(f"y_anomaly_train_tensor: {y_anomaly_train_tensor.shape}")
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_energy_train_tensor, y_user_train_tensor, y_anomaly_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model
        input_dim = X_train_tensor.shape[2]  # Number of features
        num_users = len(torch.unique(y_user_train_tensor))
        
        model_config = {
            'input_dim': input_dim,
            'num_users': num_users,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 256
        }
        
        logger.info(f"Initializing model with config: {model_config}")
        
        # Save model configuration
        with open('model/model_config.json', 'w') as f:
            json.dump(model_config, f)
        
        model = SmartHomeTransformer(**model_config)
        
        # Define loss functions and optimizer
        energy_criterion = nn.MSELoss()
        user_criterion = nn.CrossEntropyLoss()
        anomaly_criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Training loop
        logger.info("Starting training...")
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            for batch_idx, (X_batch, y_energy_batch, y_user_batch, y_anomaly_batch) in enumerate(train_loader):
                optimizer.zero_grad()
                
                # Forward pass
                energy_pred, user_pred, anomaly_pred = model(X_batch)
                
                # Calculate losses
                energy_loss = energy_criterion(energy_pred, y_energy_batch)
                user_loss = user_criterion(user_pred.view(-1, num_users), y_user_batch.view(-1))
                anomaly_loss = anomaly_criterion(anomaly_pred, y_anomaly_batch)
                
                # Combine losses
                loss = energy_loss + user_loss + anomaly_loss
                
                # Backward pass and optimization
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 100 == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Save model
        logger.info("Saving model...")
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'model_config': model_config,
            'final_loss': avg_loss
        }
        
        torch.save(checkpoint, 'model/smart_home_model.pth')
        logger.info("Training completed successfully!")
        
        return model, model_config
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        data_path = "data/smart_home_data.csv"  # Update this path to your data file
        model, config = train_model(data_path)
        logger.info("Script completed successfully!")
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        raise
# def train_model(data_path, seq_length, batch_size, num_epochs):
#     # Load and preprocess data
#     X_train, X_test, y_energy_train, y_energy_test, y_user_train, y_user_test, y_anomaly_train, y_anomaly_test = load_and_preprocess_data(data_path)
    
#     # Create sequences
#     X_train_seq, y_energy_train_seq, y_user_train_seq, y_anomaly_train_seq = create_sequences(X_train, y_energy_train, y_user_train, y_anomaly_train, seq_length)
    
#     # Convert to PyTorch tensors
#     X_train_tensor = torch.FloatTensor(X_train_seq)
#     y_energy_train_tensor = torch.FloatTensor(y_energy_train_seq).unsqueeze(-1)
#     y_user_train_tensor = torch.LongTensor(y_user_train_seq)
#     y_anomaly_train_tensor = torch.FloatTensor(y_anomaly_train_seq).unsqueeze(-1)
    
#     # Create DataLoader
#     train_dataset = TensorDataset(X_train_tensor, y_energy_train_tensor, y_user_train_tensor, y_anomaly_train_tensor)
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
#     # Initialize model
#     input_dim = X_train.shape[1]
#     num_users = len(torch.unique(y_user_train_tensor))
#     model = SmartHomeTransformer(input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, num_users=num_users)
    
#     # Define loss functions and optimizer
#     energy_criterion = nn.MSELoss()
#     user_criterion = nn.CrossEntropyLoss()
#     anomaly_criterion = nn.BCELoss()
#     optimizer = optim.Adam(model.parameters())
    
#     # Training loop
#     for epoch in range(num_epochs):
#         model.train()
#         total_loss = 0
#         for batch in train_loader:
#             X_batch, y_energy_batch, y_user_batch, y_anomaly_batch = batch
            
#             optimizer.zero_grad()
#             energy_pred, user_pred, anomaly_pred = model(X_batch)
            
#             energy_loss = energy_criterion(energy_pred, y_energy_batch)
#             user_loss = user_criterion(user_pred, y_user_batch)
#             anomaly_loss = anomaly_criterion(anomaly_pred, y_anomaly_batch)
            
#             loss = energy_loss + user_loss + anomaly_loss
#             loss.backward()
#             optimizer.step()
            
#             total_loss += loss.item()
        
#         print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
#     return model

# if __name__ == "__main__":
#     model = train_model("data/smart_home_data.csv", seq_length=60, batch_size=32, num_epochs=10)
#     torch.save(model.state_dict(), "model/smart_home_model.pth")
#     print("Model trained and saved successfully!")