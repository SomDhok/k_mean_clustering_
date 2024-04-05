import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


background_image = 'OIP.jpeg'



# Load the data
data = pd.read_csv('mall_customers.csv')

# Select relevant features
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)

# Streamlit UI
st.title('Customer Cluster Prediction')

# Input fields for user to enter data
income = st.slider('Annual Income (k$)', min_value=0, max_value=200, value=50)
spending_score = st.slider('Spending Score (1-100)', min_value=1, max_value=100, value=50)

# Predict button
if st.button('Predict Cluster'):
    # Preprocess the input data
    input_data = np.array([[income, spending_score]])
    input_data_scaled = scaler.transform(input_data)
    
    # Predict the cluster
    cluster = kmeans.predict(input_data_scaled)[0]
    
    # Define cluster categories
    cluster_categories = {
        0:'Medium Spending, Medium Income',
    1: 'High Spending, High Income',
    2: 'High Spending, Low Income',
    3: 'Low Spending, High Income',
    4: 'Low Spending, Low Income'
    }
    
    # Display the predicted cluster category
    st.write(f'The customer belongs to Cluster {cluster}: {cluster_categories[cluster]}')
   