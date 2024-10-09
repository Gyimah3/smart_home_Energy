import streamlit as st
import pandas as pd
import torch
import asyncio
from model.transformer import SmartHomeTransformer
from utils.data_preprocessing import load_and_preprocess_data, create_sequences
from utils.decision_engine import decision_engine

# # Load the trained model
# @st.cache_resource
# def load_model():
#     input_dim = 100  # Update this based on your actual input dimension after preprocessing
#     num_users = 5  # Update this based on your actual number of users
#     model = SmartHomeTransformer(input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, num_users=num_users)
#     model.load_state_dict(torch.load("model/smart_home_model.pth"))
#     model.eval()
#     return model
import io

# model = load_model()
def get_input_dim(file_path):
    try:
        X_train, _, _, _, _, _, _, _ = load_and_preprocess_data(file_path)
        input_dim = X_train.shape[1]
        print(f"Calculated input dimension: {input_dim}")
        return input_dim
    except Exception as e:
        print(f"Error calculating input dimension: {str(e)}")
        return 100  # fallback default

@st.cache_resource
def load_model(input_dim):
    try:
        with open("model/model_config.json", "r") as f:
            model_config = json.load(f)
        
        model_config['input_dim'] = input_dim
        print(f"Loading model with config: {model_config}")
        model = SmartHomeTransformer(**model_config)
        
        checkpoint = torch.load("model/smart_home_model.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        st.error("Error loading the pre-trained model. Using untrained model instead.")
        return None

st.title("Smart Home Energy Monitoring System")

st.sidebar.header("Debug Information")
show_debug = st.sidebar.checkbox("Show Debug Info")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    try:
        file_contents = uploaded_file.getvalue().decode('utf-8')
        if show_debug:
            first_lines = file_contents.split('\n')[:5]
            st.sidebar.write("First few lines of the file:")
            for line in first_lines:
                st.sidebar.text(line)

        data = pd.read_csv(io.StringIO(file_contents))
        
        if data.empty:
            st.error("The uploaded file is empty. Please check your file and try again.")
        else:
            st.write("Data Preview:")
            st.write(data.head())
            st.write(f"Number of rows: {len(data)}")
            st.write(f"Number of columns: {len(data.columns)}")
            st.write("Columns in the file:")
            st.write(data.columns.tolist())
            
            expected_columns = [
                'timestamp', 'device_id', 'device_type', 'location', 'power_watts', 'energy_kwh',
                'operational_status', 'room_temp', 'outdoor_temp', 'humidity', 'light_level',
                'motion_detected', 'door_status', 'user_id', 'user_presence', 'wifi_signal',
                'weather_condition', 'electricity_price', 'anomaly_score'
            ]
            missing_columns = [col for col in expected_columns if col not in data.columns]
            
            if missing_columns:
                st.error(f"The following expected columns are missing from your file: {', '.join(missing_columns)}")
                st.error("Please ensure your CSV file contains all necessary columns for prediction.")
            else:
                try:
                    X_train, X_test, y_energy_train, y_energy_test, y_user_train, y_user_test, y_anomaly_train, y_anomaly_test = load_and_preprocess_data(io.StringIO(file_contents))
                    st.write(f"Preprocessed data shape: {X_train.shape}")
                    
                    input_dim = X_train.shape[1]
                    if show_debug:
                        st.sidebar.write(f"Input dimension: {input_dim}")
                    
                    model = load_model(input_dim)
                    
                    X_seq = X_train[:60].reshape(1, 60, -1)
                    st.write(f"Sequence data shape: {X_seq.shape}")
                    
                    if model is not None:
                        with torch.no_grad():
                            energy_pred, user_pred, anomaly_pred = model(torch.FloatTensor(X_seq))
                        
                        st.write("Predictions:")
                        st.write(f"Energy Consumption: {energy_pred[0, -1].item():.4f}")
                        st.write(f"User ID: {torch.argmax(user_pred[0, -1]).item()}")
                        st.write(f"Anomaly Score: {anomaly_pred[0, -1].item():.4f}")
                    else:
                        st.error("Model not loaded properly. Please check the model files.")
                        
                except Exception as e:
                    st.error(f"Error during preprocessing or prediction: {str(e)}")
                    st.error("Please check if the data format matches the expected input for the model.")
                    if show_debug:
                        import traceback
                        st.sidebar.text(traceback.format_exc())

    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty. Please check your file and try again.")
    except pd.errors.ParserError as e:
        st.error(f"Error parsing the CSV file: {str(e)}. Please ensure your file is a valid CSV.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
        st.error("Please check the file format and contents in the sidebar for more information.")



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