import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend for matplotlib
import matplotlib.pyplot as plt
plt.ioff()
import seaborn as sns
import os
#############################################################################
#scikit-learn and other libraries for preprocessing and modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
##################################################################################
#tensorflow and keras for neural network modeling
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Create necessary directories
os.makedirs('visualizations', exist_ok=True)
os.makedirs('models', exist_ok=True)

def load_and_preprocess_data(file_path='../data/processed/combined_properties.csv'):
    """
    Load and preprocess the real estate data
    """
    # Load the dataset
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    
    # Display basic info
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Data cleaning and preprocessing
    print("Cleaning and preprocessing data...")
    
    # Handle missing values for key features
    df['bed'].fillna(df['bed'].median(), inplace=True)
    df['bath'].fillna(df['bath'].median(), inplace=True)
    
    # Filter extreme values (remove outliers)
    df = df[df['price'] > 100000]  # Remove properties with unrealistically low prices
    df = df[df['price'] < 10000000]  # Remove extremely expensive properties
    
    # Create new features
    df['bed_bath_ratio'] = df['bed'] / df['bath']
    
    # For missing sqft, create an estimated value based on price and location
    if df['sqft'].isna().sum() > 0 and 'pricePerSf' in df.columns:
        # Group by city and calculate median price per sqft
        city_price_per_sqft = df.groupby('city')['pricePerSf'].median().to_dict()
        
        # For rows with missing sqft, estimate based on price and city
        for city, price_per_sqft in city_price_per_sqft.items():
            if not np.isnan(price_per_sqft):
                # Estimate sqft for properties in this city with missing sqft
                mask = (df['city'] == city) & (df['sqft'].isna())
                df.loc[mask, 'sqft'] = df.loc[mask, 'price'] / price_per_sqft
    
    # If lotArea is missing but we have sqft, use a reasonable multiplier
    if 'lotArea' in df.columns and df['lotArea'].isna().sum() > 0:
        lot_area_mask = df['lotArea'].isna() & df['sqft'].notna()
        df.loc[lot_area_mask, 'lotArea'] = df.loc[lot_area_mask, 'sqft'] * 2.5  # Reasonable estimate
    
    # Select features for modeling
    features = ['city', 'bed', 'bath', 'sqft', 'lotArea', 'homeType', 'bed_bath_ratio']
    target = 'price'
    
    # Keep only rows where all selected features are available
    modeling_df = df[features + [target]].dropna()
    
    print(f"After preprocessing, dataset shape: {modeling_df.shape}")
    
    return modeling_df, features, target

def prepare_data(df, features, target, test_size=0.2):
    """
    Prepare data for modeling by splitting into train/test sets
    and creating a preprocessing pipeline
    """
    # Split data into features and target
    X = df[features]
    y = df[target]
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Create preprocessing pipeline
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['number']).columns.tolist()
    
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Save the preprocessor
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("Preprocessor saved to 'models/preprocessor.pkl'")
    
    return X_train, X_test, y_train, y_test, preprocessor

def get_feature_names(pipeline_model):
    """
    Extract feature names after preprocessing transformations
    """
    preprocessor = pipeline_model.named_steps['preprocessor']
    
    # Get feature names for numerical and categorical features
    cat_features = preprocessor.transformers_[1][2]  # Categorical features
    num_features = preprocessor.transformers_[0][2]  # Numerical features
    
    # Get one-hot encoder
    onehotencoder = preprocessor.transformers_[1][1].named_steps['onehot']
    
    # Get all feature names after transformation
    cat_feature_names = onehotencoder.get_feature_names_out(cat_features)
    feature_names = np.append(num_features, cat_feature_names)
    
    return feature_names

def plot_predictions(y_test, y_pred, model_name=None):
    """
    Plot actual vs predicted values
    
    Args:
        y_test: Actual values
        y_pred: Predicted values
        model_name: Name of the model (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    
    title = 'Actual vs Predicted Prices'
    if model_name:
        title += f' - {model_name}'
    
    plt.title(title)
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    filename = 'predictions.png'
    if model_name:
        filename = f'{model_name.lower().replace(" ", "_")}_predictions.png'
    
    plt.savefig(f'visualizations/{filename}')
    plt.close()

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance, plot residuals and predictions,
    and return metrics
    """
    # Make predictions
    if model_name == "Neural Network":
        # Neural network predictions need preprocessed data
        y_pred = model.predict(X_test).flatten()
    else:
        # Sklearn models have preprocessing in their pipeline
        y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n{model_name} Performance:")
    print(f" - MSE : {mse:.2f}")
    print(f" - RMSE: {rmse:.2f}")
    print(f" - MAE : {mae:.2f}")
    print(f" - R²  : {r2:.4f}")
    
    # Create a residual plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.hlines(0, xmin=y_pred.min(), xmax=y_pred.max(), colors='red', linestyles='--')
    plt.xlabel("Predicted Values")
    plt.ylabel("Residuals")
    plt.title(f"Residuals Plot - {model_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name.lower().replace(" ","_")}_residuals.png')
    plt.close()
    
    # Plot actual vs predicted
    plot_predictions(y_test, y_pred, model_name)
    
    return {
        'model_name': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'predictions': y_pred
    }

def train_random_forest(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train and evaluate a Random Forest model with GridSearch
    """
    print("\nTraining Random Forest model with GridSearch...")
    
    # Create pipeline with preprocessing and model
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42))
    ])
    
    # Define hyperparameters to search
    param_grid = {
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5],
        'regressor__min_samples_leaf': [1, 2]
    }
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        rf_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Get the best model
    best_rf_model = grid_search.best_estimator_
    
    # Save the model
    joblib.dump(best_rf_model, 'models/random_forest_model.pkl')
    print("Random Forest model saved to 'models/random_forest_model.pkl'")
    
    # Plot feature importance
    plot_feature_importance(best_rf_model)
    
    # Evaluate model
    metrics = evaluate_model(best_rf_model, X_test, y_test, "Random Forest")
    
    return best_rf_model, metrics

def plot_feature_importance(model):
    """
    Plot feature importance from a tree-based model
    """
    # Check if model has feature_importances_
    if not hasattr(model.named_steps['regressor'], 'feature_importances_'):
        print("This model does not support feature importance visualization")
        return
    
    # Get feature names after preprocessing
    feature_names = get_feature_names(model)
    
    # Get feature importances
    importances = model.named_steps['regressor'].feature_importances_
    
    # Create DataFrame for visualization
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Plot top 15 features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
    
    model_name = type(model.named_steps['regressor']).__name__
    plt.title(f'Top 15 Feature Importance - {model_name}')
    plt.tight_layout()
    plt.savefig(f'visualizations/{model_name.lower()}_feature_importance.png')
    plt.close()

def train_decision_tree(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train and evaluate a Decision Tree model
    """
    print("\nTraining Decision Tree model...")
    
    # Create pipeline with preprocessing and model
    dt_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', DecisionTreeRegressor(random_state=42))
    ])
    
    # Define parameters for grid search
    param_grid = {
        'regressor__max_depth': [None, 10, 20],
        'regressor__min_samples_split': [2, 5, 10],
        'regressor__min_samples_leaf': [1, 2, 4]
    }
    
    # Grid search
    grid_search = GridSearchCV(
        dt_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Get the best model
    best_dt_model = grid_search.best_estimator_
    
    # Save the model
    joblib.dump(best_dt_model, 'models/decision_tree_model.pkl')
    print("Decision Tree model saved to 'models/decision_tree_model.pkl'")
    
    # Create tree visualization (saved as DOT file for Graphviz)
    try:
        # Get the tree from the pipeline
        dt = best_dt_model.named_steps['regressor']
        
        # Get feature names
        feature_names = get_feature_names(best_dt_model)
        
        # Export to DOT file (can be visualized with Graphviz)
        export_graphviz(
            dt, 
            out_file='visualizations/decision_tree.dot',
            feature_names=feature_names,
            filled=True, 
            rounded=True,
            max_depth=3  # Limit depth for visualization
        )
        
        print("Decision tree DOT file saved to 'visualizations/decision_tree.dot'")
        
    except Exception as e:
        print(f"Error creating tree visualization: {e}")
    
    # Evaluate model
    metrics = evaluate_model(best_dt_model, X_test, y_test, "Decision Tree")
    
    return best_dt_model, metrics

def train_xgboost(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train and evaluate an XGBoost model
    """
    print("\nTraining XGBoost model...")
    
    # Create pipeline with preprocessing and model
    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', xgb.XGBRegressor(objective='reg:squarederror', random_state=42))
    ])
    
    # Define parameters for grid search
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__learning_rate': [0.01, 0.1],
        'regressor__max_depth': [3, 6, 9]
    }
    
    # Grid search
    grid_search = GridSearchCV(
        xgb_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Get the best model
    best_xgb_model = grid_search.best_estimator_
    
    # Save the model
    joblib.dump(best_xgb_model, 'models/xgboost_model.pkl')
    print("XGBoost model saved to 'models/xgboost_model.pkl'")
    
    # Plot feature importance
    plot_feature_importance(best_xgb_model)
    
    # Evaluate model
    metrics = evaluate_model(best_xgb_model, X_test, y_test, "XGBoost")
    
    return best_xgb_model, metrics

def train_knn(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train and evaluate a KNN model
    """
    print("\nTraining KNN model...")
    
    # Create pipeline with preprocessing and model
    knn_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', KNeighborsRegressor())
    ])
    
    # Define parameters for grid search
    param_grid = {
        'regressor__n_neighbors': [3, 5, 7, 9],
        'regressor__weights': ['uniform', 'distance'],
        'regressor__p': [1, 2]  # 1 for Manhattan, 2 for Euclidean
    }
    
    # Grid search
    grid_search = GridSearchCV(
        knn_pipeline, param_grid, cv=3, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Best parameters
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Get the best model
    best_knn_model = grid_search.best_estimator_
    
    # Save the model
    joblib.dump(best_knn_model, 'models/knn_model.pkl')
    print("KNN model saved to 'models/knn_model.pkl'")
    
    # Evaluate model
    metrics = evaluate_model(best_knn_model, X_test, y_test, "KNN")
    
    return best_knn_model, metrics

def train_neural_network(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train and evaluate a Neural Network model
    
    Note: This function follows the same interface as other training functions,
    but internally handles preprocessing differently since Keras doesn't use sklearn pipelines
    """
    print("\nTraining Neural Network model...")
    
    # Process the data with the preprocessor
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Split train data to create a validation set
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train_processed, y_train, test_size=0.2, random_state=42
    )
    
    # Get input dimensions
    input_dim = X_train_processed.shape[1]
    
    # Build the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(1)  # Output layer
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    # Define callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath='models/neural_network_model.weights.h5',
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train_split, y_train_split,
        validation_data=(X_val, y_val),
        batch_size=32,
        epochs=100,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save preprocessor separately for neural network
    joblib.dump(preprocessor, 'models/neural_network_preprocessor.pkl')
    print("Neural Network preprocessor saved to 'models/neural_network_preprocessor.pkl'")
    
    # Evaluate model
    metrics = evaluate_model(model, X_test_processed, y_test, "Neural Network")
    
    return model, metrics

def plot_training_history(history):
    """Plot training and validation loss and MAE."""
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Plot training & validation mean absolute error
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('visualizations/neural_network_training_history.png')
    plt.close()
    
    print("Training history plot saved to 'visualizations/neural_network_training_history.png'")

def compare_models(metrics_list):
    """
    Compare performance of different models
    """
    print("\nComparing model performance...")
    
    # Create DataFrame from metrics
    metrics_df = pd.DataFrame(metrics_list)
    
    # Sort by R2 score (higher is better)
    metrics_df = metrics_df.sort_values('r2', ascending=False)
    
    # Display metrics table
    pd.set_option('display.precision', 4)
    print("\nModel Performance Comparison:")
    print(metrics_df[['model_name', 'rmse', 'mae', 'r2']])
    
    # Save metrics to CSV
    metrics_df.to_csv('visualizations/model_performance.csv', index=False)
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    # Plot RMSE (lower is better)
    plt.subplot(2, 2, 1)
    sns.barplot(x='model_name', y='rmse', data=metrics_df)
    plt.title('RMSE by Model (Lower is Better)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    # Plot MAE (lower is better)
    plt.subplot(2, 2, 2)
    sns.barplot(x='model_name', y='mae', data=metrics_df)
    plt.title('MAE by Model (Lower is Better)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    # Plot R2 (higher is better)
    plt.subplot(2, 2, 3)
    sns.barplot(x='model_name', y='r2', data=metrics_df)
    plt.title('R² Score by Model (Higher is Better)')
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    plt.tight_layout()
    plt.savefig('visualizations/model_comparison.png')
    plt.close()
    
    print("Model comparison plots saved to 'visualizations/model_comparison.png'")
    
    # Return the best model name based on R2
    return metrics_df.iloc[0]['model_name']

def save_best_model(best_model_name, models):
    """
    Save a copy of the best model
    """
    print(f"\nSaving the best model: {best_model_name}")
    
    if best_model_name == "Neural Network":
        # For Neural Network, the model is already saved during training
        print("Best model (Neural Network) already saved to 'models/neural_network_model.h5'")
    else:
        # For scikit-learn models, make a copy
        model_filename = f"models/{best_model_name.lower().replace(' ', '_')}_model.pkl"
        best_model = models[best_model_name]
        joblib.dump(best_model, 'models/best_model.pkl')
        print(f"Best model saved to 'models/best_model.pkl'")
    
    # Also save a file indicating the best model type
    with open('models/best_model_type.txt', 'w') as f:
        f.write(best_model_name)
    
    print(f"Best model type saved to 'models/best_model_type.txt'")

def main():
    """
    Main function to run the entire pipeline
    """
    print("="*80)
    print("REAL ESTATE PRICE PREDICTION PIPELINE")
    print("="*80)
    
    try:
        # 1. Load and preprocess data
        df, features, target = load_and_preprocess_data()
        
        # 2. Prepare data for modeling
        X_train, X_test, y_train, y_test, preprocessor = prepare_data(df, features, target)
        
        # 3. Train and evaluate models
        models = {}
        metrics_list = []
        
        # Random Forest
        rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test, preprocessor)
        models["Random Forest"] = rf_model
        metrics_list.append(rf_metrics)
        
        # Decision Tree
        dt_model, dt_metrics = train_decision_tree(X_train, X_test, y_train, y_test, preprocessor)
        models["Decision Tree"] = dt_model
        metrics_list.append(dt_metrics)
        
        # XGBoost
        xgb_model, xgb_metrics = train_xgboost(X_train, X_test, y_train, y_test, preprocessor)
        models["XGBoost"] = xgb_model
        metrics_list.append(xgb_metrics)
        
        # KNN
        knn_model, knn_metrics = train_knn(X_train, X_test, y_train, y_test, preprocessor)
        models["KNN"] = knn_model
        metrics_list.append(knn_metrics)
        
        # Neural Network
        nn_model, nn_metrics = train_neural_network(X_train, X_test, y_train, y_test, preprocessor)
        models["Neural Network"] = nn_model
        metrics_list.append(nn_metrics)
        
        # 4. Compare models
        best_model_name = compare_models(metrics_list)
        
        # 5. Save the best model
        save_best_model(best_model_name, models)
        
        print("\nPipeline completed successfully!")
        print(f"The best model is: {best_model_name}")
        print("="*80)
        
        return 0
        
    except Exception as e:
        print(f"Error in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    main()