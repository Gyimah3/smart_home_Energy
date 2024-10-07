import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class SmartHomeTransformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_users):
        super(SmartHomeTransformer, self).__init__()
        
        # Input embedding
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # Energy prediction (regression)
        self.energy_decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # User prediction (classification)
        self.user_decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_users)
        )
        
        # Anomaly prediction (binary classification)
        self.anomaly_decoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Ensures output is between 0 and 1
        )

    def forward(self, src):
        # Get the last dimension of src tensor for debugging
        batch_size, seq_len, feat_dim = src.size()
        
        # Input embedding and positional encoding
        src = self.embedding(src)
        src = self.pos_encoder(src)
        
        # Transformer encoding
        output = self.transformer_encoder(src)
        
        # Get the last sequence element for predictions
        last_hidden = output[:, -1, :]
        
        # Make predictions
        energy_pred = self.energy_decoder(last_hidden)
        user_pred = self.user_decoder(last_hidden)
        anomaly_pred = self.anomaly_decoder(last_hidden)
        
        # Double-check anomaly predictions are in [0, 1]
        anomaly_pred = torch.clamp(anomaly_pred, 0, 1)
        
        return energy_pred, user_pred, anomaly_pred
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super(PositionalEncoding, self).__init__()
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         pe = pe.unsqueeze(0).transpose(0, 1)
#         self.register_buffer('pe', pe)

#     def forward(self, x):
#         return x + self.pe[:x.size(0), :]

# class SmartHomeTransformer(nn.Module):
#     def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, num_users):
#         super(SmartHomeTransformer, self).__init__()
#         self.embedding = nn.Linear(input_dim, d_model)
#         self.pos_encoder = PositionalEncoding(d_model)
#         encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
#         self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
#         self.energy_decoder = nn.Linear(d_model, 1)
#         self.user_decoder = nn.Linear(d_model, num_users)
#         self.anomaly_decoder = nn.Sequential(
#             nn.Linear(d_model, 1),
#             nn.Sigmoid()  # Ensure anomaly prediction is between 0 and 1
#         )

#     def forward(self, src):
#         src = self.embedding(src)
#         src = self.pos_encoder(src)
#         output = self.transformer_encoder(src)
#         energy_pred = self.energy_decoder(output[:, -1, :])
#         user_pred = self.user_decoder(output[:, -1, :])
#         anomaly_pred = self.anomaly_decoder(output[:, -1, :])
#         return energy_pred, user_pred, anomaly_pred