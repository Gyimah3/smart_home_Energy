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

def train_model(data_path, seq_length, batch_size, num_epochs):
    # Load and preprocess data
    X_train, X_test, y_energy_train, y_energy_test, y_user_train, y_user_test, y_anomaly_train, y_anomaly_test = load_and_preprocess_data(data_path)
    
    # Create sequences
    X_train_seq, y_energy_train_seq, y_user_train_seq, y_anomaly_train_seq = create_sequences(
        X_train, y_energy_train, y_user_train, y_anomaly_train, seq_length
    )
    
    # Convert to PyTorch tensors with proper shapes
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_energy_train_tensor = torch.FloatTensor(y_energy_train_seq).unsqueeze(-1)
    y_user_train_tensor = torch.LongTensor(y_user_train_seq)
    y_anomaly_train_tensor = torch.FloatTensor(y_anomaly_train_seq).unsqueeze(-1)
    
    # Ensure anomaly values are between 0 and 1
    y_anomaly_train_tensor = torch.clamp(y_anomaly_train_tensor, 0, 1)
    
    print(f"Training data shapes:")
    print(f"X_train_tensor: {X_train_tensor.shape}")
    print(f"y_energy_train_tensor: {y_energy_train_tensor.shape}")
    print(f"y_user_train_tensor: {y_user_train_tensor.shape}")
    print(f"y_anomaly_train_tensor: {y_anomaly_train_tensor.shape}")
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_energy_train_tensor, y_user_train_tensor, y_anomaly_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train_tensor.shape[2]  # Get input dimension from the data
    num_users = len(torch.unique(y_user_train_tensor))
    model_config = {
        'input_dim': input_dim,
        'num_users': num_users,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 2,
        'dim_feedforward': 256
    }
    
    with open('model/model_config.json', 'w') as f:
        json.dump(model_config, f)
    
    model = SmartHomeTransformer(**model_config)
    
    # Define loss functions and optimizer
    energy_criterion = nn.MSELoss()
    user_criterion = nn.CrossEntropyLoss()
    anomaly_criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            X_batch, y_energy_batch, y_user_batch, y_anomaly_batch = batch
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            energy_pred, user_pred, anomaly_pred = model(X_batch)
            
            # Print shapes for debugging (first batch of first epoch)
            if epoch == 0 and batch_idx == 0:
                print("\nPrediction shapes:")
                print(f"energy_pred: {energy_pred.shape}")
                print(f"user_pred: {user_pred.shape}")
                print(f"anomaly_pred: {anomaly_pred.shape}")
                print("\nTarget shapes:")
                print(f"y_energy_batch: {y_energy_batch.shape}")
                print(f"y_user_batch: {y_user_batch.shape}")
                print(f"y_anomaly_batch: {y_anomaly_batch.shape}")
            
            # Ensure anomaly predictions are between 0 and 1
            anomaly_pred = torch.clamp(anomaly_pred, 0, 1)
            
            # Calculate losses
            energy_loss = energy_criterion(energy_pred, y_energy_batch)
            user_loss = user_criterion(user_pred, y_user_batch)
            anomaly_loss = anomaly_criterion(anomaly_pred, y_anomaly_batch)
            
            # Combine losses
            loss = energy_loss + user_loss + anomaly_loss
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
    
    return model

if __name__ == "__main__":
    try:
        model = train_model("data/smart_home_data.csv", seq_length=60, batch_size=32, num_epochs=10)
        
        # Save both model and configuration
        torch.save({
            'model_state_dict': model.state_dict(),
            'input_dim': model.embedding.weight.shape[1]
        }, "model/smart_home_model.pth")
        
        print("Model trained and saved successfully!")
    except Exception as e:
        print(f"Error during training: {str(e)}")
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