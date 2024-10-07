import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, SplineTransformer
from sklearn.model_selection import train_test_split
print("started preprocessing")
def sin_transformer(period):
    return lambda x: np.sin(x * (2. * np.pi / period))

def cos_transformer(period):
    return lambda x: np.cos(x * (2. * np.pi / period))

def periodic_spline_transformer(period, n_splines=6, degree=3):
    n_knots = n_splines + 1
    knots = np.linspace(0, period, n_knots)[:-1]
    return SplineTransformer(n_knots=n_knots, degree=degree, knots=knots,
                             extrapolation="periodic", include_bias=False)

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Extract time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
    
    # Create cyclical features
    df["hour_sin"] = sin_transformer(24)(df["hour"])
    df["hour_cos"] = cos_transformer(24)(df["hour"])
    df["month_sin"] = sin_transformer(12)(df["month"])
    df["month_cos"] = cos_transformer(12)(df["month"])
    
    spline_cols_month = [f"cyclic_month_spline_{i}" for i in range(1, 7)]
    df[spline_cols_month] = periodic_spline_transformer(12, n_splines=6).fit_transform(np.array(df.month).reshape(-1,1))
    
    # One-hot encode categorical variables
    categorical_cols = ['device_type', 'location', 'weather_condition', 'operational_status', 'door_status', 'user_presence']
    df = pd.get_dummies(df, columns=categorical_cols)
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    numerical_cols = ['power_watts', 'energy_kwh', 'room_temp', 'outdoor_temp', 'humidity', 'light_level', 'wifi_signal', 'electricity_price']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Create rolling statistics
    rolling_windows = [17, 39, 52]
    for col in numerical_cols:
        for window in rolling_windows:
            df[f"{col}_rolling_mean_{window}"] = df[col].rolling(window=window).mean()
            df[f"{col}_rolling_max_{window}"] = df[col].rolling(window=window).max()
            df[f"{col}_rolling_min_{window}"] = df[col].rolling(window=window).min()
            df[f"{col}_rolling_std_{window}"] = df[col].rolling(window=window).std()
    
    # Create location-based statistics
    for col in numerical_cols:
        df[f"{col}_location_mean"] = df.groupby(['location'])[col].transform('mean')
        df[f"{col}_location_std"] = df.groupby(['location'])[col].transform('std')
        df[f"{col}_location_min"] = df.groupby(['location'])[col].transform('min')
        df[f"{col}_location_max"] = df.groupby(['location'])[col].transform('max')
    
    # Prepare features and targets
    features = df.drop(['timestamp', 'user_id', 'energy_kwh', 'anomaly_score'], axis=1)
    energy_target = df['energy_kwh']
    user_target = pd.get_dummies(df['user_id'])
    anomaly_target = df['anomaly_score']
    
    return train_test_split(features, energy_target, user_target, anomaly_target, test_size=0.2, random_state=42)

def create_sequences(features, energy_target, user_target, anomaly_target, seq_length):
    X, y_energy, y_user, y_anomaly = [], [], [], []
    for i in range(len(features) - seq_length):
        X.append(features[i:i+seq_length].values)
        y_energy.append(energy_target[i+seq_length])
        y_user.append(user_target[i+seq_length].values)
        y_anomaly.append(anomaly_target[i+seq_length])
    return np.array(X), np.array(y_energy), np.array(y_user), np.array(y_anomaly)