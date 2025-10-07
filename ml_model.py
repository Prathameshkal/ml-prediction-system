import pandas as pd
import numpy as np
import pickle
import os
import json
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score, confusion_matrix
from sklearn.impute import SimpleImputer
from config import config
import warnings
import time
warnings.filterwarnings('ignore')

class MLPredictor:
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.label_encoders = {}
        self.feature_names = {}
        self.encoders = {}
        
    def train_model(self, filepath, target_column, model_type='random_forest'):
        """Train ML model with large dataset support"""
        
        start_time = time.time()
        
        try:
            # Check file size to determine processing strategy
            file_size = os.path.getsize(filepath)
            use_chunks = file_size > 10 * 1024 * 1024  # 10MB threshold
            
            if use_chunks:
                print("Using chunked processing for large file...")
                df = self._load_large_dataset(filepath)
            else:
                df = pd.read_csv(filepath)
            
            print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Basic validation
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in dataset")
            
            if len(df) < 10:
                raise ValueError("Dataset too small. Need at least 10 samples")
            
            if len(df) > config.MAX_ROWS:
                # Sample data if too large
                df = df.sample(n=config.MAX_ROWS, random_state=42)
                print(f"Sampled to {config.MAX_ROWS} rows for training")
            
            # Handle missing values
            df = self._handle_missing_values(df)
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Detect problem type
            problem_type = self._detect_problem_type(y)
            print(f"Problem type: {problem_type}")
            
            # Preprocess features (optimized for large datasets)
            X_processed, feature_names, preprocessor = self._preprocess_features_optimized(X, problem_type)
            self.feature_names[model_type] = feature_names
            self.encoders[model_type] = preprocessor
            
            # Encode target
            if problem_type == 'classification':
                le = LabelEncoder()
                y_encoded = le.fit_transform(y)
                self.label_encoders[model_type] = le
                print(f"Classes: {len(le.classes_)}")
            else:
                y_encoded = y.values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y_encoded, test_size=0.2, random_state=42,
                stratify=y_encoded if problem_type == 'classification' else None
            )
            
            print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
            
            # Scale features (memory efficient)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers[model_type] = scaler
            
            # Get appropriate model (optimized for large data)
            model = self._get_optimized_model(model_type, problem_type, X_train.shape[0])
            
            print(f"Training {model_type} model...")
            model.fit(X_train_scaled, y_train)
            
            # Predictions and metrics
            y_pred = model.predict(X_test_scaled)
            metrics = self._calculate_comprehensive_metrics(y_test, y_pred, problem_type, model, X_test_scaled)
            
            # Cross-validation with fewer folds for large data
            cv_folds = 3 if X_train.shape[0] > 10000 else 5
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=cv_folds, 
                scoring='accuracy' if problem_type == 'classification' else 'r2'
            )
            
            metrics['cross_val_mean'] = cv_scores.mean()
            metrics['cross_val_std'] = cv_scores.std()
            
            # Store model
            self.models[model_type] = {
                'model': model,
                'feature_names': feature_names,
                'problem_type': problem_type,
                'metrics': metrics,
                'target_column': target_column,
                'training_time': time.time() - start_time
            }
            
            # Save model
            self._save_model(model_type)
            
            return {
                'metrics': metrics,
                'problem_type': problem_type,
                'feature_importance': self._get_feature_importance(model, feature_names),
                'model_type': model_type,
                'dataset_info': {
                    'rows': len(df),
                    'features': len(feature_names),
                    'target_classes': len(np.unique(y)) if problem_type == 'classification' else 'regression',
                    'training_time_seconds': time.time() - start_time
                }
            }
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            raise e
    
    def _load_large_dataset(self, filepath):
        """Load large dataset efficiently"""
        chunks = []
        for chunk in pd.read_csv(filepath, chunksize=config.CHUNK_SIZE):
            # Basic cleaning for each chunk
            chunk = chunk.dropna(axis=1, how='all')
            chunks.append(chunk)
            
            # Stop if we have enough data
            total_rows = sum(len(c) for c in chunks)
            if total_rows >= config.MAX_ROWS:
                break
        
        df = pd.concat(chunks, ignore_index=True)
        
        # If still too large, sample
        if len(df) > config.MAX_ROWS:
            df = df.sample(n=config.MAX_ROWS, random_state=42)
        
        return df
    
    def _preprocess_features_optimized(self, X, problem_type):
        """Optimized preprocessing for large datasets"""
        
        numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
        categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
        
        print(f"Numerical: {len(numerical_columns)}, Categorical: {len(categorical_columns)}")
        
        # For very large datasets, limit categorical columns
        if len(categorical_columns) > 50:
            print("Too many categorical columns. Selecting top 50 by cardinality...")
            categorical_columns = categorical_columns[:50]
        
        # Handle numerical columns
        numerical_imputer = SimpleImputer(strategy='median')  # More robust for large data
        X_numerical = numerical_imputer.fit_transform(X[numerical_columns])
        
        # Handle categorical columns
        if categorical_columns:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            X_categorical = categorical_imputer.fit_transform(X[categorical_columns])
            
            # One-hot encode with max categories limit
            encoder = OneHotEncoder(
                drop='first', 
                sparse_output=False, 
                handle_unknown='ignore',
                max_categories=20  # Limit categories per column
            )
            X_categorical_encoded = encoder.fit_transform(X_categorical)
            
            # Get feature names
            categorical_feature_names = []
            for i, col in enumerate(categorical_columns):
                categories = encoder.categories_[i][1:21]  # Limit to 20 categories
                categorical_feature_names.extend([f"{col}_{cat}" for cat in categories])
            
            # Combine features
            X_processed = np.hstack([X_numerical, X_categorical_encoded])
            feature_names = numerical_columns + categorical_feature_names
            
            preprocessor = {
                'numerical_imputer': numerical_imputer,
                'categorical_imputer': categorical_imputer,
                'encoder': encoder,
                'numerical_columns': numerical_columns,
                'categorical_columns': categorical_columns
            }
        else:
            X_processed = X_numerical
            feature_names = numerical_columns
            preprocessor = {
                'numerical_imputer': numerical_imputer,
                'numerical_columns': numerical_columns,
                'categorical_columns': []
            }
        
        return X_processed, feature_names, preprocessor
    
    def _get_optimized_model(self, model_type, problem_type, n_samples):
        """Get model optimized for dataset size"""
        
        if n_samples > 100000:  # Very large dataset
            if model_type == 'random_forest':
                if problem_type == 'classification':
                    return RandomForestClassifier(
                        n_estimators=50,  # Fewer trees
                        max_depth=15,
                        random_state=42,
                        n_jobs=-1  # Use all cores
                    )
                else:
                    return RandomForestRegressor(
                        n_estimators=50,
                        max_depth=15,
                        random_state=42,
                        n_jobs=-1
                    )
        
        # Standard models for smaller datasets
        model_map = {
            'random_forest': {
                'classification': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
                'regression': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            },
            'logistic_regression': {
                'classification': LogisticRegression(random_state=42, max_iter=1000, n_jobs=-1),
                'regression': LinearRegression(n_jobs=-1)
            },
            'svm': {
                'classification': SVC(random_state=42, probability=True),
                'regression': SVR()
            }
        }
        
        return model_map.get(model_type, model_map['random_forest'])[problem_type]

