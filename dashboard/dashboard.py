import streamlit as st
import pandas as pd
import numpy as np
import requests
import json

# Set page config
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title and description
st.title("Real Estate Price Prediction")
st.write("Enter property details to predict the price")

# Create tabs for different sections
tab1, tab2 = st.tabs(["Prediction", "Visualizations"])

with tab1:
    # Create two columns for a cleaner form layout
    col1, col2 = st.columns(2)

    with col1:
      city = st.selectbox("City", ["Oakville", "Milton", "Mississauga", "Guelph"])
      bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
      bathrooms = st.number_input("Number of Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.5)

    with col2:
        sqft = st.number_input("Square Footage", min_value=500, max_value=10000, value=2000)
        lot_area = st.number_input("Lot Area (sqft)", min_value=0, value=5000)
        home_type = st.selectbox("Property Type", ["SINGLE_FAMILY", "TOWNHOUSE", "CONDO", "MULTI_FAMILY"])

    # Calculate bed/bath ratio
    bed_bath_ratio = bedrooms / bathrooms

    # Create property data dictionary
    property_data = {
        "city": city,
        "bed": bedrooms,
        "bath": bathrooms,
        "sqft": sqft,
        "lotArea": lot_area,
        "homeType": home_type,
        "bed_bath_ratio": bed_bath_ratio
    }

    # Create predict button
    if st.button("Predict Price"):
        try:
            # Make API request to your Flask service
            api_url = "http://localhost:5000/api/predict"  # Change the URL as needed
            
            # Send POST request
            response = requests.post(api_url, json=property_data)
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Display prediction
                st.success(f"Predicted Price: {result['formatted_prediction']}")
                
                # Add additional details if you want
                st.write("Prediction Details:")
                st.json(result)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
            # Display the property data even if API call fails
            st.write("Property data:", property_data)

with tab2:
    st.header("Property Price Visualizations")
    
    # Example visualization - Price distribution by city
    # In a real app, you would fetch this data from your API
    st.subheader("Average Price by City")
    
    # Sample data - replace with data from your API
    sample_data = {
        "Oakville": 1200000,
        "Milton": 950000,
        "Mississauga": 1100000,
        "Guelph": 850000
    }
    
    # Create a bar chart
    st.bar_chart(sample_data)