import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import skew

print("started preprocessing")

def sin_transformer(period):
    return lambda x: np.sin(x * (2. * np.pi / period))

def cos_transformer(period):
    return lambda x: np.cos(x * (2. * np.pi / period))

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    df['season'] = (df['month'] % 12 + 3) // 3
    
    # Create cyclical features
    cyclical_features = ['hour', 'day_of_week', 'month']
    cycles = [24, 7, 12]
    for feature, cycle in zip(cyclical_features, cycles):
        df[f"{feature}_sin"] = sin_transformer(cycle)(df[feature])
        df[f"{feature}_cos"] = cos_transformer(cycle)(df[feature])
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_cols = ['power_watts', 'energy_kwh', 'room_temp', 'outdoor_temp', 'humidity', 'light_level', 'wifi_signal', 'electricity_price']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Create rolling statistics
    rolling_windows = [60, 1440, 10080]  # 1 hour, 1 day, 1 week (assuming 1-minute intervals)
    for col in ['power_watts', 'energy_kwh']:
        for window in rolling_windows:
            df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window).mean()
            df[f"{col}_rolling_max_{window}"] = df[col].rolling(window=window).max()
            df[f"{col}_rolling_min_{window}"] = df[col].rolling(window=window).min()
            df[f"{col}_rolling_sum_{window}"] = df[col].rolling(window=window).sum()
            df[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window).std()
            df[f"{col}_rolling_skew_{window}"] = df[col].rolling(window=window).apply(skew)
    
    # Create location-based statistics
    for col in numerical_cols:
        df[f"{col}_location_mean"] = df.groupby(['location'])[col].transform('mean')
        df[f"{col}_location_std"] = df.groupby(['location'])[col].transform('std')
        df[f"{col}_location_min"] = df.groupby(['location'])[col].transform('min')
        df[f"{col}_location_max"] = df.groupby(['location'])[col].transform('max')
        df[f"{col}_location_skew"] = df.groupby(['location'])[col].transform(skew)
    
    # One-hot encode categorical variables
    categorical_cols = ['device_type', 'location', 'weather_condition', 'operational_status', 'door_status', 'user_presence']
    df = pd.get_dummies(df, columns=categorical_cols)
    
    # Prepare features and targets
    features = df.drop(['timestamp', 'user_id', 'energy_kwh', 'anomaly_score'], axis=1)
    energy_target = df['energy_kwh']
    user_target = pd.get_dummies(df['user_id'])
    anomaly_target = df['anomaly_score']
    
    return train_test_split(features, energy_target, user_target, anomaly_target, test_size=0.2, random_state=42)

def create_sequences(features, energy_target, user_target, anomaly_target, seq_length):
    X, y_energy, y_user, y_anomaly = [], [], [], []
    for i in range(len(features) - seq_length + 1):
        X.append(features.iloc[i:i+seq_length].values)
        y_energy.append(energy_target.iloc[i+seq_length-1])
        y_user.append(user_target.iloc[i+seq_length-1].values)
        y_anomaly.append(anomaly_target.iloc[i+seq_length-1])
    
    return (
        np.array(X, dtype=np.float32),
        np.array(y_energy, dtype=np.float32),
        np.array(y_user, dtype=np.float32),
        np.array(y_anomaly, dtype=np.float32)
    )