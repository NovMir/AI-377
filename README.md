# Real Estate Price Prediction System

A full-stack machine learning application for predicting real estate prices in the Greater Toronto Area. This project implements multiple ML algorithms to analyze property data and predict prices based on features like location, size, and property type.

## Project Overview

This system was developed as part of the COMP377 course project on "Developing Full-Stack Intelligent Apps". It demonstrates:

1. Machine learning for real estate price prediction
2. Data analysis and visualization
3. RESTful API development
4. Interactive dashboard creation

The application follows modern software architecture principles, implementing the MVC design pattern for the backend API and using a component-based architecture for the frontend.

## Directory Structure

```
.
├── .gitignore                                # Git ignore file
├── dashboard/                                # Streamlit frontend
│   └── dashboard.py                          # Streamlit application
├── LICENSE                                   # License file
├── ml_service/                               # Machine learning service
│   ├── api/                                  # API implementation
│   │   └── app.py                            # Flask API endpoints
│   ├── data_processing.py                    # Data loading and preprocessing
│   ├── data/                                 # Data directory
│   │   ├── processed/                        # Processed data
│   │   │   └── combined_properties.csv       # Combined dataset
│   │   └── raw/                              # Raw data files
│   │       ├── GuelphPrice.csv               # Guelph properties
│   │       ├── MiltonPrice.csv               # Milton properties
│   │       ├── MissisaugaPrice.csv           # Mississauga properties
│   │       └── OakvillePrice.csv             # Oakville properties
│   └── model/                                # Machine learning models
│       ├── model_training.py                 # Model training script
│       ├── models/                           # Saved model files
│       │   └── best_model_type.txt           # Best model identifier
│       └── visualizations/                   # Model visualizations
│           ├── decision_tree.dot             # Decision tree visualization
│           └── model_performance.csv         # Performance metrics
└── requirements.txt                          # Python dependencies
```

## Features

- **Data Processing**:
  - Combines property data from multiple cities
  - Handles missing values intelligently
  - Creates derived features for better predictions

- **Data Visualization**:
  - Creates insightful visualizations of property data
  - Analyzes price trends by location, property size, and type
  - Visualizes relationships between different property features

- **Machine Learning Models**:
  - Random Forest Regressor (with GridSearch optimization)
  - Decision Tree Regressor (with visualization)
  - XGBoost Regressor (gradient boosting)
  - K-Nearest Neighbors Regressor
  - Neural Network (using TensorFlow/Keras)

- **Model Evaluation**:
  - Compares models using RMSE, MAE, and R² metrics
  - Visualizes prediction accuracy and residuals
  - Identifies the most important features for price prediction

- **RESTful API**:
  - Provides endpoints for single and batch predictions
  - Returns model information and feature importance
  - Includes documentation and health check endpoints

- **Interactive Dashboard**:
  - User-friendly interface for property price prediction
  - Displays data visualizations and insights
  - Shows model performance metrics

## Technologies Used

- **Backend**:
  - Python 3.x
  - Flask (RESTful API)
  - Scikit-learn (ML algorithms)
  - XGBoost (Gradient boosting)
  - TensorFlow/Keras (Neural Networks)
  - Pandas & NumPy (Data processing)
  - Matplotlib & Seaborn (Visualization)

- **Frontend**:
  - Streamlit (Interactive dashboard)
  - Matplotlib/Seaborn (Visualizations)

## Installation and Setup

### Prerequisites

- Python 3.8+ installed
- Git (optional, for cloning)
- Pip (Python package manager)

### Step 1: Clone the Repository (or download)

```bash
git clone https://github.com/yourusername/real-estate-prediction.git
cd real-estate-prediction
```

### Step 2: Create and Activate a Virtual Environment (Optional but Recommended)

```bash
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Process the Data and Train Models

```bash
cd ml_service
python data_processing.py
python model/model_training.py
```

This will:
1. Load and process the raw data files
2. Create visualizations in the `visualizations` directory
3. Train all machine learning models
4. Compare their performance and save the best model

### Step 5: Start the API Server

```bash
# From the ml_service/api directory
cd api
python app.py
```

The API will be available at http://localhost:5000/api

### Step 6: Launch the Streamlit Dashboard

```bash
# From the project root, in a new terminal
cd dashboard
streamlit run dashboard.py
```

The dashboard will open in your browser at http://localhost:8501

## API Endpoints

- **GET /api/health**: Check API health and model status
- **POST /api/predict**: Predict price for a single property
- **GET /api/model-info**: Get information about the model
- **GET /api/feature-importance**: Get feature importance data

## Usage

### 1. Property Price Prediction

1. Navigate to the Streamlit dashboard
2. Enter property details (city, bedrooms, bathrooms, square footage, etc.)
3. Click "Predict Price" to get an estimated property value

### 2. Data Exploration

The dashboard provides various visualizations to explore:
- Price distribution by city
- Relationship between property size and price
- Impact of bedrooms and bathrooms on price
- Property type distribution

### 3. Model Performance

You can view model performance metrics and compare different algorithms:
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R² Score (Coefficient of determination)

## Performance Results

Based on evaluation metrics, the models performed in the following order (best to worst):

1. **XGBoost**: RMSE = 3.23e+05, R² = 0.9462
2. **Random Forest**: RMSE = 3.73e+05, R² = 0.9284
3. **KNN**: RMSE = 4.41e+05, R² = 0.8998
4. **Decision Tree**: RMSE = 4.45e+05, R² = 0.8980
5. **Neural Network**: RMSE = 2.17e+06, R² = -1.4381

The XGBoost model proved most effective for this dataset, providing the best balance of prediction accuracy and generalization capability.


## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data sourced from real estate listings in the Greater Toronto Area from the site zillow.com
- COMP377 course instructors and teaching assistants
- Scikit-learn, TensorFlow, and Streamlit documentation and communities