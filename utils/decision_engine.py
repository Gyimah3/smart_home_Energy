import numpy as np

def decision_engine(energy_pred, user_pred, anomaly_pred, thresholds, device_status='unknown', device_type='unknown'):
    """
    Generate decisions based on model predictions and device information.
    
    Args:
        energy_pred (float): Predicted energy consumption
        user_pred (np.ndarray): User prediction logits
        anomaly_pred (float): Anomaly prediction score
        thresholds (dict): Dictionary containing threshold values
        device_status (str): Current device status
        device_type (str): Type of device
        
    Returns:
        list: List of recommended actions
    """
    try:
        actions = []
        
        # Normalize predictions if needed
        energy_pred = float(energy_pred)
        anomaly_pred = float(anomaly_pred)
        user_probs = np.exp(user_pred) / np.sum(np.exp(user_pred))  # Convert logits to probabilities
        
        # Energy consumption decision
        if energy_pred > thresholds['energy']:
            actions.append(f"⚠️ High energy consumption detected for {device_type}. Consider optimizing usage.")
        
        # User identification
        user_id = np.argmax(user_probs)
        user_confidence = user_probs[user_id]
        if user_confidence > 0.7:
            actions.append(f"👤 User {user_id} identified with high confidence ({user_confidence:.2%})")
        else:
            actions.append(f"👥 Multiple users may be present. Primary user: {user_id}")
        
        # Anomaly detection
        if anomaly_pred > thresholds['anomaly']:
            actions.append(f"🚨 Unusual energy usage pattern detected for {device_type}. Investigating...")
        
        # Device-specific rules
        if device_status != 'unknown' and device_type != 'unknown':
            # Lighting optimization
            if device_type.lower() in ['light', 'lighting'] and device_status.lower() == 'on':
                if np.max(user_probs) < thresholds['user_presence']:
                    actions.append(f"💡 Light left on with low user presence probability. Recommend turning off.")
            
            # Standby power optimization
            if device_type.lower() in ['tv', 'television', 'computer', 'pc'] and device_status.lower() == 'on':
                if energy_pred < thresholds['standby']:
                    actions.append(f"⚡ {device_type} may be in standby mode. Consider turning it off completely.")
            
            # General device recommendations
            if device_status.lower() == 'on' and energy_pred > 0.9:
                actions.append(f"📊 {device_type} is consuming more energy than usual. Check for inefficiencies.")
                
        # Add general energy-saving recommendations if needed
        if not actions:
            actions.append("✅ No immediate actions needed. Device is operating within normal parameters.")
        
        return actions
        
    except Exception as e:
        return [f"Error in decision engine: {str(e)}"]