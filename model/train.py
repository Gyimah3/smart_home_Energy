import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from model.transformer import SmartHomeTransformer
from utils.data_preprocessing import load_and_preprocess_data, create_sequences

print("started training")

def train_model(data_path, seq_length, batch_size, num_epochs):
    # Load and preprocess data
    X_train, X_test, y_energy_train, y_energy_test, y_user_train, y_user_test, y_anomaly_train, y_anomaly_test = load_and_preprocess_data(data_path)
    
    # Create sequences
    X_train_seq, y_energy_train_seq, y_user_train_seq, y_anomaly_train_seq = create_sequences(X_train, y_energy_train, y_user_train, y_anomaly_train, seq_length)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_seq)
    y_energy_train_tensor = torch.FloatTensor(y_energy_train_seq).unsqueeze(-1)
    y_user_train_tensor = torch.FloatTensor(y_user_train_seq)
    y_anomaly_train_tensor = torch.FloatTensor(y_anomaly_train_seq).unsqueeze(-1)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_energy_train_tensor, y_user_train_tensor, y_anomaly_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train.shape[1]
    num_users = y_user_train.shape[1]
    model = SmartHomeTransformer(input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, num_users=num_users)
    
    # Define loss functions and optimizer
    energy_criterion = nn.MSELoss()
    user_criterion = nn.CrossEntropyLoss()
    anomaly_criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            X_batch, y_energy_batch, y_user_batch, y_anomaly_batch = batch
            
            optimizer.zero_grad()
            energy_pred, user_pred, anomaly_pred = model(X_batch)
            
            energy_loss = energy_criterion(energy_pred, y_energy_batch)
            user_loss = user_criterion(user_pred, y_user_batch.argmax(dim=1))
            anomaly_loss = anomaly_criterion(anomaly_pred, y_anomaly_batch)
            
            loss = energy_loss + user_loss + anomaly_loss
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
    
    return model

if __name__ == "__main__":
    model = train_model("data/smart_home_data.csv", seq_length=60, batch_size=32, num_epochs=10)
    torch.save(model.state_dict(), "model/smart_home_model.pth")
    print("Model trained and saved successfully!")