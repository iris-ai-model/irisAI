import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
import shap
from sklearn.model_selection import StratifiedKFold

class IrisModel:
    def __init__(self, model_type='RandomForest'):
        """
        Initializes the IrisModel with a chosen machine learning model type.
        
        model_type: str, optional (default='RandomForest')
            The type of model to use for training. Options: ['RandomForest', 'XGBoost', 'LogisticRegression']
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')
        print(f"IrisModel initialized with {self.model_type} model.")
    
    def preprocess_data(self, df):
        """
        Preprocesses the transaction data, including handling missing values and scaling features.
        
        df: DataFrame
            The transaction data to preprocess.
        
        Returns:
        --------
        X: DataFrame
            The preprocessed features for training.
        y: Series
            The target variable (if available).
        """
        # Select features and target
        feature_columns = ['amount', 'fee', 'slot', 'block_time']
        target_column = 'target'  # This assumes the target is a column in the dataset
        
        # Handle missing values by imputing
        df[feature_columns] = self.imputer.fit_transform(df[feature_columns])
        
        # Extract features (X) and target (y)
        X = df[feature_columns]
        y = df[target_column] if target_column in df else None
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def feature_engineering(self, df):
        """
        Adds engineered features to the dataset, such as temporal features and transaction statistics.
        
        df: DataFrame
            The transaction data.
        
        Returns:
        --------
        df: DataFrame
            The dataset with added engineered features.
        """
        # Create additional temporal features (e.g., day of week, hour of day)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['hour_of_day'] = df['timestamp'].dt.hour
        df['is_weekend'] = df['day_of_week'].isin([5, 6])  # 0=Monday, 6=Sunday
        
        # Aggregate statistics per sender or receiver (just an example of aggregation)
        sender_stats = df.groupby('sender')['amount'].agg(['mean', 'std', 'sum'])
        df = df.merge(sender_stats, on='sender', how='left', suffixes=('', '_sender'))
        
        # Add more features as needed...
        
        return df
    
    def train_model(self, X, y):
        """
        Trains the selected model (e.g., RandomForest, XGBoost, Logistic Regression).
        
        X: DataFrame
            The input features for training.
        y: Series
            The target variable.
        """
        if self.model_type == 'RandomForest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif self.model_type == 'XGBoost':
            self.model = xgb.XGBClassifier(random_state=42)
        elif self.model_type == 'LogisticRegression':
            self.model = LogisticRegression(random_state=42)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train the model using cross-validation
        self.model.fit(X, y)
        
        print(f"{self.model_type} model trained successfully.")
    
    def evaluate_model(self, X, y):
        """
        Evaluates the trained model on a test dataset and prints performance metrics.
        
        X: DataFrame
            The input features for evaluation.
        y: Series
            The true target variable for evaluation.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        # Generate predictions
        y_pred = self.model.predict(X)
        y_prob = self.model.predict_proba(X)[:, 1]  # Probability of the positive class (binary classification)
        
        # Evaluate performance
        print("Classification Report:\n", classification_report(y, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
        print("ROC-AUC Score:", roc_auc_score(y, y_prob))
        
        # Plot confusion matrix
        self.plot_confusion_matrix(confusion_matrix(y, y_pred))
    
    def plot_confusion_matrix(self, cm):
        """Visualizes the confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Fraud', 'Fraud'], yticklabels=['No Fraud', 'Fraud'])
        plt.title(f'Confusion Matrix ({self.model_type})')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def feature_importance(self):
        """Displays the feature importance from the trained model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        # Extract feature importance for tree-based models (e.g., RandomForest, XGBoost)
        if hasattr(self.model, 'feature_importances_'):
            feature_importances = self.model.feature_importances_
            feature_names = ['amount', 'fee', 'slot', 'block_time', 'day_of_week', 'hour_of_day', 'is_weekend']  # Add other feature names
            feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
            feature_df = feature_df.sort_values(by='Importance', ascending=False)
            
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_df)
            plt.title(f'Feature Importance ({self.model_type})')
            plt.show()
    
    def hyperparameter_tuning(self, X, y):
        """
        Performs hyperparameter tuning using GridSearchCV.
        
        X: DataFrame
            The input features for training.
        y: Series
            The target variable.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        # Define parameter grid
        if self.model_type == 'RandomForest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
            }
        elif self.model_type == 'XGBoost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
            }
        else:
            raise ValueError(f"Hyperparameter tuning not supported for {self.model_type}")
        
        # Perform grid search
        grid_search = GridSearchCV(self.model, param_grid, cv=3, n_jobs=-1, verbose=1)
        grid_search.fit(X, y)
        
        print(f"Best parameters for {self.model_type}: {grid_search.best_params_}")
        self.model = grid_search.best_estimator_
    
    def predict(self, X):
        """Predicts the target variable for new data."""
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        # Make predictions
        return self.model.predict(X)
    
    def explain_model(self, X):
        """
        Uses SHAP to explain the model's predictions.
        
        X: DataFrame
            The input features for which to explain predictions.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X)
        
        # Visualize SHAP values
        shap.summary_plot