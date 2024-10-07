import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.stats import skew
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
print("started preprocessing")


# def sin_transformer(period):
#     return lambda x: np.sin(x * (2. * np.pi / period))

# def cos_transformer(period):
#     return lambda x: np.cos(x * (2. * np.pi / period))


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp to ensure correct rolling calculations
    df = df.sort_values('timestamp')
    
    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    df['season'] = (df['month'] % 12 + 3) // 3
    
    # Create cyclical features
    def sin_transformer(period):
        return lambda x: np.sin(x * (2. * np.pi / period))
    
    def cos_transformer(period):
        return lambda x: np.cos(x * (2. * np.pi / period))
    
    cyclical_features = ['hour', 'day_of_week', 'month']
    cycles = [24, 7, 12]
    for feature, cycle in zip(cyclical_features, cycles):
        df[f"{feature}_sin"] = sin_transformer(cycle)(df[feature])
        df[f"{feature}_cos"] = cos_transformer(cycle)(df[feature])
    
    # Create rolling statistics
    rolling_windows = [60, 1440, 10080]  # 1 hour, 1 day, 1 week (assuming 1-minute intervals)
    for col in ['power_watts', 'energy_kwh']:
        for window in rolling_windows:
            # Mean
            df[f"{col}_rolling_mean_{window}"] = df[col].rolling(
                window=window, min_periods=1).mean()
            
            # Standard deviation
            df[f"{col}_rolling_std_{window}"] = df[col].rolling(
                window=window, min_periods=1).std().fillna(0)
            
            # Min and Max
            df[f"{col}_rolling_min_{window}"] = df[col].rolling(
                window=window, min_periods=1).min()
            df[f"{col}_rolling_max_{window}"] = df[col].rolling(
                window=window, min_periods=1).max()
            
            # Difference from mean
            df[f"{col}_diff_mean_{window}"] = df[col] - df[f"{col}_rolling_mean_{window}"]
    
    # Identify numerical and categorical columns
    numerical_cols = [
        'power_watts', 'energy_kwh', 'room_temp', 'outdoor_temp', 
        'humidity', 'light_level', 'wifi_signal', 'electricity_price'
    ]
    # Add cyclical features
    numerical_cols += [f"{feat}_{func}" for feat in cyclical_features for func in ['sin', 'cos']]
    # Add rolling features
    for col in ['power_watts', 'energy_kwh']:
        for window in rolling_windows:
            numerical_cols += [
                f"{col}_rolling_mean_{window}",
                f"{col}_rolling_std_{window}",
                f"{col}_rolling_min_{window}",
                f"{col}_rolling_max_{window}",
                f"{col}_diff_mean_{window}"
            ]
    
    categorical_cols = [
        'device_id', 'device_type', 'location', 'operational_status',
        'motion_detected', 'door_status', 'user_id', 'user_presence',
        'weather_condition'
    ]
    
    # Handle missing values in rolling features
    df[numerical_cols] = df[numerical_cols].fillna(method='ffill').fillna(0)
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Encode categorical features
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le
    
    # Normalize anomaly score to [0, 1]
    df['anomaly_score'] = MinMaxScaler().fit_transform(df[['anomaly_score']])
    
    # Print feature information
    print(f"Total number of features: {len(numerical_cols) + len(categorical_cols)}")
    print(f"Number of numerical features: {len(numerical_cols)}")
    print(f"Number of categorical features: {len(categorical_cols)}")
    
    # Prepare features and targets
    features = df.drop(['timestamp', 'energy_kwh', 'anomaly_score'], axis=1).astype(np.float32)
    energy_target = df['energy_kwh'].astype(np.float32)
    user_target = df['user_id'].astype(np.int64)
    anomaly_target = df['anomaly_score'].astype(np.float32)
    
    return train_test_split(features, energy_target, user_target, anomaly_target, test_size=0.2, random_state=42)

def create_sequences(features, energy_target, user_target, anomaly_target, seq_length):
    X, y_energy, y_user, y_anomaly = [], [], [], []
    
    features_array = features.values if isinstance(features, pd.DataFrame) else features
    
    for i in range(len(features) - seq_length + 1):
        X.append(features_array[i:i+seq_length])
        y_energy.append(energy_target.iloc[i+seq_length-1] if isinstance(energy_target, pd.Series) else energy_target[i+seq_length-1])
        y_user.append(user_target.iloc[i+seq_length-1] if isinstance(user_target, pd.Series) else user_target[i+seq_length-1])
        y_anomaly.append(anomaly_target.iloc[i+seq_length-1] if isinstance(anomaly_target, pd.Series) else anomaly_target[i+seq_length-1])
    
    return (
        np.array(X, dtype=np.float32),
        np.array(y_energy, dtype=np.float32),
        np.array(y_user, dtype=np.int64),
        np.array(y_anomaly, dtype=np.float32)
    )