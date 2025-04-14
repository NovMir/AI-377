import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Define the API URL (change this based on your deployment)
API_URL = "http://localhost:5000/api"

# Title and description
st.title("Real Estate Price Prediction")
st.write("Enter property details to predict the price")

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["Prediction", "Visualizations", "Model Performance"])

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
            api_url = f"{API_URL}/predict"
            
            # Send POST request
            response = requests.post(api_url, json=property_data)
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                
                # Display prediction with larger font and styling
                st.markdown(f"""
                <div style="padding: 20px; background-color: #f0f7ff; border-radius: 10px; text-align: center;">
                    <h2>Predicted Price</h2>
                    <h1 style="color: #0066cc;">{result['formatted_prediction']}</h1>
                </div>
                """, unsafe_allow_html=True)
                
                # Show prediction details in an expandable section
                with st.expander("View Prediction Details"):
                    st.json(result)
            else:
                st.error(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
            # Display the property data even if API call fails
            st.write("Property data:", property_data)

with tab2:
    st.header("Property Price Visualizations")
    
    # Fetch feature importance
    try:
        feature_response = requests.get(f"{API_URL}/feature-importance")
        if feature_response.status_code == 200:
            feature_data = feature_response.json()['feature_importance']
            
            # Convert to DataFrame for easier plotting
            feature_df = pd.DataFrame(feature_data)
            
            # Display top 10 features
            st.subheader("Feature Importance")
            st.write("See which factors influence property prices the most")
            
            # Plot feature importance
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = feature_df.head(10)
            
            # Clean up feature names for display
            top_features['Feature'] = top_features['feature'].str.replace('cat__', '').str.replace('_', ' ')
            
            # Create bar chart
            sns.barplot(x='importance', y='Feature', data=top_features, palette='viridis')
            plt.title('Top 10 Features Influencing Property Price')
            plt.xlabel('Importance')
            plt.tight_layout()
            
            # Display the plot
            st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not load feature importance: {e}")
    
    # Average price by city (static example - in production would come from API)
    st.subheader("Average Price by City")
    
    # Sample data - replace with data from your API
    city_data = {
        "Oakville": 1200000,
        "Milton": 950000,
        "Mississauga": 1100000,
        "Guelph": 850000
    }
    
    # Create a bar chart
    st.bar_chart(city_data)
    
    # Price vs. Bedrooms & Bathrooms (interactive visualization)
    st.subheader("Price by Bedrooms and Bathrooms")
    
    # Sample data for visualization - in production would come from API
    # Create a grid of bedroom/bathroom combinations
    bed_values = list(range(1, 7))
    bath_values = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    
    bed_bath_grid = []
    for bed in bed_values:
        for bath in bath_values:
            bed_bath_grid.append({
                "bed": bed,
                "bath": bath,
                "city": city,  # Use the currently selected city
                "sqft": 2000,  # Use a standard size
                "lotArea": 5000,
                "homeType": "SINGLE_FAMILY",
                "bed_bath_ratio": bed/bath
            })
    
    # Create a heatmap of prices
    try:
        # Use a small subset for the example (would call API in production)
        # In production: response = requests.post(f"{API_URL}/batch-predict", json=bed_bath_grid)
        # Simulate some data for demonstration
        np.random.seed(42)
        prices = []
        for item in bed_bath_grid:
            # Simulate price (in real app, get from API)
            base_price = 800000
            bed_factor = item["bed"] * 100000
            bath_factor = item["bath"] * 120000
            # Add some randomness
            random_factor = np.random.normal(0, 50000)
            price = base_price + bed_factor + bath_factor + random_factor
            prices.append(price)
        
        # Create dataframe for heatmap
        heatmap_data = pd.DataFrame(bed_bath_grid)
        heatmap_data['price'] = prices
        
        # Pivot for heatmap
        pivot_data = heatmap_data.pivot_table(
            values='price', 
            index='bed',
            columns='bath',
            aggfunc='mean'
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            pivot_data/1000000, 
            annot=True, 
            fmt=".2f",
            cmap="YlOrRd",
            linewidths=0.5,
            cbar_kws={'label': 'Price (Millions $)'}
        )
        plt.title(f'Estimated Property Prices by Bedroom and Bathroom Count in {city}')
        plt.xlabel('Number of Bathrooms')
        plt.ylabel('Number of Bedrooms')
        st.pyplot(fig)
        
        # Add explanation
        st.info("""
        This heatmap shows how property prices vary based on the number of bedrooms and bathrooms.
        Values shown are in millions of dollars. Darker colors indicate higher prices.
        """)
    except Exception as e:
        st.warning(f"Could not generate bedroom/bathroom price matrix: {e}")
    
    # Price vs Square Footage scatter plot
    st.subheader("Price vs Square Footage")
    
    # Generate sample data (in production, fetch from API)
    np.random.seed(42)
    n_points = 100
    sqft_values = np.random.uniform(1000, 4000, n_points)
    price_base = 500000
    price_values = price_base + (sqft_values * 300) + np.random.normal(0, 100000, n_points)
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(sqft_values, price_values, alpha=0.6)
    plt.title('Property Price vs Square Footage')
    plt.xlabel('Square Footage')
    plt.ylabel('Price ($)')
    
    # Add trend line
    z = np.polyfit(sqft_values, price_values, 1)
    p = np.poly1d(z)
    plt.plot(sqft_values, p(sqft_values), "r--")
    
    # Display the plot
    st.pyplot(fig)
    
    # Add explanation
    st.info("""
    This scatter plot shows the relationship between a property's square footage and its price.
    The red dashed line represents the general trend - as square footage increases, price tends to increase.
    """)

with tab3:
    st.header("Model Performance Metrics")
    
    # Fetch model info from API
    try:
        model_info_response = requests.get(f"{API_URL}/model-info")
        if model_info_response.status_code == 200:
            model_info = model_info_response.json()
            
            # Display model type
            st.subheader(f"Current Model: {model_info.get('model_type', 'Unknown')}")
            
            # Display performance metrics if available
            if 'performance' in model_info:
                metrics = model_info['performance']
                
                # Create metrics display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="RMSE",
                        value=f"${metrics.get('rmse', 0):,.2f}",
                        help="Root Mean Squared Error (lower is better)"
                    )
                
                with col2:
                    st.metric(
                        label="MAE",
                        value=f"${metrics.get('mae', 0):,.2f}",
                        help="Mean Absolute Error (lower is better)"
                    )
                
                with col3:
                    st.metric(
                        label="R² Score",
                        value=f"{metrics.get('r2', 0):.4f}",
                        help="Coefficient of determination (higher is better, max 1.0)"
                    )
                
                # Add explanation of metrics
                st.info("""
                **Understanding these metrics:**
                - **RMSE (Root Mean Squared Error)**: The average magnitude of prediction errors, with larger errors penalized more heavily.
                - **MAE (Mean Absolute Error)**: The average magnitude of prediction errors, regardless of direction.
                - **R² Score**: Indicates how well the model explains price variations. A score of 1.0 means perfect predictions.
                """)
    except Exception as e:
        st.warning(f"Could not load model information: {e}")
    
    # Add a section about model accuracy
    st.subheader("Model Accuracy Visualization")
    
    # Generate sample actual vs predicted data (in production, fetch from API)
    np.random.seed(42)
    n_samples = 50
    actual_prices = np.random.uniform(500000, 2000000, n_samples)
    predicted_prices = actual_prices + np.random.normal(0, 100000, n_samples)
    
    # Create scatter plot
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(actual_prices, predicted_prices, alpha=0.6)
    
    # Add perfect prediction line
    min_val = min(actual_prices.min(), predicted_prices.min())
    max_val = max(actual_prices.max(), predicted_prices.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('Actual vs Predicted Prices')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.grid(True, alpha=0.3)
    
    # Format axes with dollar signs
    plt.ticklabel_format(style='plain', axis='both')
    
    # Display the plot
    st.pyplot(fig)
    
    # Add explanation
    st.info("""
    This plot shows how well the model's predictions match actual property prices.
    Points closer to the red dashed line indicate more accurate predictions.
    Points above the line are overestimations, while points below are underestimations.
    """)

# Footer
st.markdown("---")
st.markdown("© 2025 Real Estate Price Predictor | COMP377 Project")