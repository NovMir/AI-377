from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
import traceback
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Constants
MODEL_PATH = r"C:\House Prediction\AI-377\ml_service\model\models\best_model.pkl"
MODEL_INFO_PATH = os.path.join(os.path.dirname(MODEL_PATH), "model_performance.csv")

# Global variables to store model and preprocessor
model = None
model_type = None
model_info = None

def load_model():
    """
    Load the trained model and preprocessor
    """
    global model, model_type, model_info
    
    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        
        # Determine model type (tree-based or other)
        if hasattr(model, 'named_steps') and hasattr(model.named_steps['regressor'], 'feature_importances_'):
            model_type = "tree-based"
        else:
            model_type = "non-tree-based"
        
        logger.info(f"Model type: {model_type}")
        
        # Load model info if available
        try:
            if os.path.exists(MODEL_INFO_PATH):
                model_info = pd.read_csv(MODEL_INFO_PATH)
                logger.info("Model performance info loaded")
        except Exception as e:
            logger.warning(f"Could not load model info: {e}")
            model_info = None
        
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        traceback.print_exc()
        return False

@app.route('/api/health', methods=['GET'])
def health_check():
    """
    Endpoint to check API health and model status
    """
    if model is None:
        status = "Model not loaded"
        healthy = False
    else:
        status = "Model loaded and ready"
        healthy = True
    
    return jsonify({
        'status': status,
        'healthy': healthy,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict property price based on features
    
    Expected JSON format:
    {
        "city": "Oakville",
        "bed": 4,
        "bath": 3,
        "sqft": 2000,
        "lotArea": 5000,
        "homeType": "SINGLE_FAMILY"
    }
    """
    # Check if model is loaded
    if model is None:
        success = load_model()
        if not success:
            return jsonify({
                'error': 'Model could not be loaded'
            }), 500
    
    # Get request data
    try:
        data = request.get_json()
        logger.info(f"Received prediction request: {data}")
        
        # Validate required fields
        required_fields = ['city', 'bed', 'bath']
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f"Missing required field: {field}"
                }), 400
        
        # Calculate bed_bath_ratio if not provided
        if 'bed_bath_ratio' not in data:
            data['bed_bath_ratio'] = data['bed'] / data['bath']
        
        # Create DataFrame from input
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        
        # Format response
        response = {
            'prediction': float(prediction),
            'formatted_prediction': f"${prediction:,.2f}",
            'input': data,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction: ${prediction:,.2f}")
        return jsonify(response)
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e)
        }), 500


@app.route('/api/model-info', methods=['GET'])
def get_model_info():
    """
    Endpoint to get information about the model
    """
    # Check if model is loaded
    if model is None:
        success = load_model()
        if not success:
            return jsonify({
                'error': 'Model could not be loaded'
            }), 500
    
    try:
        # Basic model info
        model_name = type(model).__name__
        if hasattr(model, 'named_steps') and 'regressor' in model.named_steps:
            model_name = type(model.named_steps['regressor']).__name__
        
        info = {
            'model_type': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Add performance metrics if available
        if model_info is not None:
            # Find the row for this model
            model_row = model_info[model_info['model_name'].str.contains(model_name, case=False)]
            if not model_row.empty:
                metrics = model_row.iloc[0].to_dict()
                info['performance'] = {
                    'rmse': metrics.get('rmse'),
                    'mae': metrics.get('mae'),
                    'r2': metrics.get('r2')
                }
        
        return jsonify(info)
    
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/feature-importance', methods=['GET'])
def get_feature_importance():
    """
    Endpoint to get feature importance (for tree-based models)
    """
    # Check if model is loaded
    if model is None:
        success = load_model()
        if not success:
            return jsonify({
                'error': 'Model could not be loaded'
            }), 500
    
    try:
        # Check if model supports feature importance
        if model_type != "tree-based":
            return jsonify({
                'error': 'Current model does not support feature importance'
            }), 400
        
        # Extract feature importance
        regressor = model.named_steps['regressor']
        
        # Get feature names after preprocessing
        preprocessor = model.named_steps['preprocessor']
        
        # Get feature names for categorical features
        cat_features = preprocessor.transformers_[1][2]  # Categorical features
        num_features = preprocessor.transformers_[0][2]  # Numerical features
        
        # Get one-hot encoder
        onehotencoder = preprocessor.transformers_[1][1].named_steps['onehot']
        
        # Get all feature names after transformation
        cat_feature_names = onehotencoder.get_feature_names_out(cat_features)
        feature_names = np.append(num_features, cat_feature_names)
        
        # Get feature importances
        importances = regressor.feature_importances_
        
        # Create a list of feature importance objects
        feature_importance = [
            {'feature': feature, 'importance': float(importance)}
            for feature, importance in zip(feature_names, importances)
        ]
        
        # Sort by importance (descending)
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        return jsonify({
            'feature_importance': feature_importance,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        logger.error(f"Error getting feature importance: {e}")
        traceback.print_exc()
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api', methods=['GET'])
def api_documentation():
    """
    Endpoint to provide API documentation
    """
    docs = {
        'api_name': 'Real Estate Price Prediction API',
        'version': '1.0',
        'description': 'API for predicting real estate prices based on property features',
        'endpoints': [
            {
                'path': '/api/health',
                'method': 'GET',
                'description': 'Check API health and model status'
            },
            {
                'path': '/api/predict',
                'method': 'POST',
                'description': 'Predict price for a single property',
                'example_body': {
                    'city': 'Oakville',
                    'bed': 4,
                    'bath': 3,
                    'sqft': 2000,
                    'lotArea': 5000,
                    'homeType': 'SINGLE_FAMILY'
                }
            },
            {
                'path': '/api/batch-predict',
                'method': 'POST',
                'description': 'Predict prices for multiple properties',
                'example_body': '[{property1}, {property2}, ...]'
            },
            {
                'path': '/api/model-info',
                'method': 'GET',
                'description': 'Get information about the model'
            },
            {
                'path': '/api/feature-importance',
                'method': 'GET',
                'description': 'Get feature importance data (for tree-based models)'
            }
        ]
    }
    
    return jsonify(docs)

# Custom error handlers
@app.errorhandler(400)
def bad_request(error):
    return jsonify({
        'error': 'Bad request',
        'message': str(error)
    }), 400

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Not found',
        'message': 'The requested resource does not exist'
    }), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': str(error)
    }), 500

# At the top after creating the app
with app.app_context():
    # Load the model when the application starts
    load_model()

if __name__ == '__main__':
    # Set the port
    port = int(os.environ.get('PORT', 5000))
    
    # No need to preload the model again, it's already loaded above
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=port, debug=True)