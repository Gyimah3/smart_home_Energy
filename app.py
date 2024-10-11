import streamlit as st
import pandas as pd
import numpy as np
import torch
import json
import io
from model.transformer import SmartHomeTransformer
from utils.data_preprocessing import load_and_preprocess_data, create_sequences
from utils.decision_engine import decision_engine

def get_model_config():
    """Load model configuration from JSON file"""
    try:
        with open("model/model_config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            'input_dim': 55,
            'num_users': 5,
            'd_model': 64,
            'nhead': 4,
            'num_layers': 2,
            'dim_feedforward': 256
        }

@st.cache_resource
def load_model():
    try:
        model_config = get_model_config()
        model = SmartHomeTransformer(**model_config)
        
        try:
            checkpoint = torch.load("model/smart_home_model.pth")
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            st.sidebar.success("Model loaded successfully!")
        except FileNotFoundError:
            st.warning("No pre-trained model found. Using initialized model.")
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def get_thresholds():
    """Define thresholds for decision engine"""
    return {
        'energy': 0.8,
        'anomaly': 0.7,
        'user_presence': 0.5,
        'standby': 0.1
    }

def process_predictions(model, X_seq, device_status='unknown', device_type='unknown'):
    """Process model predictions and get decisions"""
    try:
        with torch.no_grad():
            # Ensure input tensor has correct shape
            X_tensor = torch.FloatTensor(X_seq)
            
            # Get model predictions
            energy_pred, user_pred, anomaly_pred = model(X_tensor)
            
            # Extract last timestep predictions
            energy_val = energy_pred[0].item()
            user_logits = user_pred[0]
            user_id = torch.argmax(user_logits).item()
            anomaly_val = anomaly_pred[0].item()
            
            # Get decisions from decision engine
            thresholds = get_thresholds()
            decisions = decision_engine(
                energy_val,
                user_logits.numpy(),
                anomaly_val,
                thresholds,
                device_status,
                device_type
            )
            
            return {
                'energy_pred': energy_val,
                'user_id': user_id,
                'anomaly_score': anomaly_val,
                'decisions': decisions
            }
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None

def process_data(data):
    """Process input data and create sequences"""
    try:
        # Preprocess the data
        X_train, X_test, y_energy_train, y_energy_test, y_user_train, y_user_test, y_anomaly_train, y_anomaly_test = load_and_preprocess_data(io.StringIO(data))
        
        # Create sequence for prediction
        if len(X_train) >= 60:
            X_seq = X_train[:60].reshape(1, 60, -1)
        else:
            # Pad sequences shorter than 60 timesteps
            padding = np.zeros((60 - len(X_train), X_train.shape[1]))
            padded_data = np.vstack([X_train, padding])
            X_seq = padded_data.reshape(1, 60, -1)
        
        return X_seq, X_train.shape
    except Exception as e:
        st.error(f"Data processing error: {str(e)}")
        return None, None

def display_predictions(predictions):
    """Display model predictions and decisions"""
    st.write("### Predictions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Energy Consumption", f"{predictions['energy_pred']:.4f}")
    with col2:
        st.metric("User ID", str(predictions['user_id']))
    with col3:
        st.metric("Anomaly Score", f"{predictions['anomaly_score']:.4f}")
    
    st.write("### Recommended Actions")
    for decision in predictions['decisions']:
        st.info(decision)

def main():
    st.title("Smart Home Energy Monitoring System")
    
    # Debug information in sidebar
    st.sidebar.title("Debug Information")
    
    # Load model
    model = load_model()
    
    # File uploader
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read file contents
            file_contents = uploaded_file.getvalue().decode('utf-8')
            data = pd.read_csv(io.StringIO(file_contents))
            
            if data.empty:
                st.error("The uploaded file is empty.")
                return
            
            # Display data preview
            with st.expander("Data Preview"):
                st.write(data.head())
                st.write(f"Number of rows: {len(data)}")
                st.write(f"Number of columns: {len(data.columns)}")
            
            # Process data
            X_seq, data_shape = process_data(file_contents)
            
            if X_seq is not None and model is not None:
                st.sidebar.write(f"Preprocessed data shape: {data_shape}")
                st.sidebar.write(f"Sequence data shape: {X_seq.shape}")
                
                # Get device information from data if available
                device_status = data['operational_status'].iloc[-1] if 'operational_status' in data.columns else 'unknown'
                device_type = data['device_type'].iloc[-1] if 'device_type' in data.columns else 'unknown'
                
                # Make predictions and get decisions
                predictions = process_predictions(model, X_seq, device_status, device_type)
                
                if predictions:
                    display_predictions(predictions)
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
            st.sidebar.error(f"Detailed error: {str(e)}")

if __name__ == "__main__":
    main()
# def get_input_dim():
#     """Calculate the input dimension from the data"""
#     try:
#         X_train, _, _, _, _, _, _, _ = load_and_preprocess_data("data/smart_home_data.csv")
#         input_dim = X_train.shape[1]
#         print(f"Calculated input dimension: {input_dim}")
#         return input_dim
#     except Exception as e:
#         print(f"Error calculating input dimension: {str(e)}")
#         return 100  # fallback default

# # Load the trained model
# @st.cache_resource
# def load_model():
#     try:
#         # Load model configuration
#         checkpoint = torch.load("model/smart_home_model.pth")
#         input_dim = checkpoint['input_dim']
        
#         print(f"Loading model with input_dim={input_dim}")
#         model = SmartHomeTransformer(
#             input_dim=input_dim,
#             d_model=64,
#             nhead=4,
#             num_layers=2,
#             dim_feedforward=256,
#             num_users=5  # Update this based on your data
#         )
        
#         model.load_state_dict(checkpoint['model_state_dict'])
#         return model
#     except Exception as e:
#         print(f"Error loading model: {str(e)}")
#         st.error("Error loading the pre-trained model. Using untrained model instead.")
#         return None


# # # Streamlit app
# # st.title("Smart Home Energy Monitoring System")

# # uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# # if uploaded_file is not None:
# #     data = pd.read_csv(uploaded_file)
# #     st.write("Data Preview:")
# #     st.write(data.head())

# #     # Preprocess the data
# #     X, _, _, _ = load_and_preprocess_data(uploaded_file)
# #     X_seq, _, _, _ = create_sequences(X, pd.Series([0]*len(X)), pd.DataFrame([[0]*5]*len(X)), pd.Series([0]*len(X)), seq_length=60)

# # Streamlit app
# st.title("Smart Home Energy Monitoring System")

# # Add debugging information
# st.sidebar.header("Debug Information")
# if st.sidebar.checkbox("Show Debug Info"):
#     input_dim = get_input_dim()
#     st.sidebar.write(f"Input dimension: {input_dim}")

# model = load_model()

# uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

# if uploaded_file is not None:
#     try:
#         data = pd.read_csv(uploaded_file)
#         if data.empty:
#             st.error("The uploaded file is empty. Please check your file and try again.")
#         else:
#             st.write("Data Preview:")
#             st.write(data.head())
            
#             # Check if the required columns are present
#             required_columns = ['timestamp', 'device_id', 'device_type', 'power_watts', 'energy_kwh']  # Add all required columns
#             missing_columns = [col for col in required_columns if col not in data.columns]
            
#             if missing_columns:
#                 st.error(f"The following required columns are missing from your file: {', '.join(missing_columns)}")
#             else:
#                 # Proceed with preprocessing and prediction
#                 try:
#                     X, _, _, _ = load_and_preprocess_data(uploaded_file)
#                     st.write(f"Preprocessed data shape: {X.shape}")
        
#         # Make predictions
#         async def make_predictions(input_data):
#             with torch.no_grad():
#                 energy_pred, user_pred, anomaly_pred = model(torch.FloatTensor(input_data))
#             return energy_pred[:, -1, :], user_pred[:, -1, :], anomaly_pred[:, -1, :]

#         # Decision engine
#         async def get_decisions(energy_pred, user_pred, anomaly_pred, device_status, device_type):
#             thresholds = {
#                 'energy': 0.8,
#                 'anomaly': 0.7,
#                 'user_presence': 0.5,
#                 'standby': 0.1
#             }
#             return decision_engine(energy_pred.item(), user_pred.numpy(), anomaly_pred.item(), thresholds, device_status, device_type)

#         # Run predictions and decision engine concurrently
#         async def process_data(input_data, device_status, device_type):
#             energy_pred, user_pred, anomaly_pred = await make_predictions(input_data)
#             decisions = await get_decisions(energy_pred[0], user_pred[0], anomaly_pred[0], device_status, device_type)
#             return energy_pred[0].item(), user_pred[0].numpy(), anomaly_pred[0].item(), decisions

#         # Run the async functions for each device
#         results = []
#         for i in range(len(X_seq)):
#             device_status = data.iloc[i]['operational_status']
#             device_type = data.iloc[i]['device_type']
#             results.append(asyncio.run(process_data(X_seq[i:i+1], device_status, device_type)))

#         # Display results
#         for i, (energy_pred, user_pred, anomaly_pred, decisions) in enumerate(results):
#             st.write(f"Device {i+1}:")
#             st.write(f"Energy Consumption Prediction: {energy_pred:.2f}")
#             st.write(f"User Identification: {np.argmax(user_pred)}")
#             st.write(f"Anomaly Score: {anomaly_pred:.2f}")
            
#             st.write("Decisions:")
#             for decision in decisions:
#                 st.write(f"- {decision}")
#             st.write("---")

#     except Exception as e:
#         st.error(f"Error processing data: {str(e)}")