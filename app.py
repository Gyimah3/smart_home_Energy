import streamlit as st
import pandas as pd
import torch
import asyncio
from model.transformer import SmartHomeTransformer
from utils.data_processing import load_and_preprocess_data, create_sequences
from utils.decision_engine import decision_engine

# Load the trained model
@st.cache_resource
def load_model():
    input_dim = 20  # Update this based on your actual input dimension
    num_users = 5  # Update this based on your actual number of users
    model = SmartHomeTransformer(input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, num_users=num_users)
    model.load_state_dict(torch.load("model/smart_home_model.pth"))
    model.eval()
    return model

model = load_model()

# Streamlit app
st.title("Smart Home Energy Monitoring System")

uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(data.head())

    # Preprocess the data
    X, _, _, _ = load_and_preprocess_data(uploaded_file)
    X_seq, _, _, _ = create_sequences(X, pd.Series([0]*len(X)), pd.DataFrame([[0]*5]*len(X)), pd.Series([0]*len(X)), seq_length=60)
    
    # Make predictions
    async def make_predictions():
        with torch.no_grad():
            energy_pred, user_pred, anomaly_pred = model(torch.FloatTensor(X_seq))
        return energy_pred[:, -1, :], user_pred[:, -1, :], anomaly_pred[:, -1, :]

    # Decision engine
    async def get_decisions(energy_pred, user_pred, anomaly_pred):
        thresholds = {'energy': 0.8, 'anomaly': 0.7}
        return decision_engine(energy_pred.item(), user_pred.numpy(), anomaly_pred.item(), thresholds)

    # Run predictions and decision engine concurrently
    async def process_data():
        energy_pred, user_pred, anomaly_pred = await make_predictions()
        decisions = await get_decisions(energy_pred[0], user_pred[0], anomaly_pred[0])
        return energy_pred[0].item(), user_pred[0].numpy(), anomaly_pred[0].item(), decisions

    # Run the async functions
    energy_pred, user_pred, anomaly_pred, decisions = asyncio.run(process_data())

    # Display results
    st.write("Energy Consumption Prediction:", energy_pred)
    st.write("User Identification:", np.argmax(user_pred))
    st.write("Anomaly Score:", anomaly_pred)
    
    st.write("Decisions:")
    for decision in decisions:
        st.write(f"- {decision}")

else:
    st.write("Please upload a CSV file to start the analysis.")