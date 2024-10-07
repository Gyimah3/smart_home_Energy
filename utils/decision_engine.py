import numpy as np

def decision_engine(energy_pred, user_pred, anomaly_pred, thresholds, device_status, device_type):
    actions = []
    
    # Energy consumption decision
    if energy_pred > thresholds['energy']:
        actions.append(f"High energy consumption detected for {device_type}. Consider optimizing usage.")
    
    # User identification
    user_id = np.argmax(user_pred)
    actions.append(f"User {user_id} identified as the current active user.")
    
    # Anomaly detection
    if anomaly_pred > thresholds['anomaly']:
        actions.append(f"Unusual energy usage pattern detected for {device_type}. Investigating...")
    
    # Device-specific rules
    if device_type == 'light' and device_status == 'on' and user_pred.max() < thresholds['user_presence']:
        actions.append(f"Light left on with low user presence probability. Recommend turning off.")
    
    if device_type in ['TV', 'computer'] and device_status == 'on' and energy_pred < thresholds['standby']:
        actions.append(f"{device_type} may be in standby mode. Consider turning it off completely.")
    
    return actions