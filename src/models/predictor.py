"""
Tabular Price Predictor - Simplified, Fast, Interpretable
Uses only structured features (no text) for reliable predictions
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
import os
import warnings
warnings.filterwarnings('ignore')


class TabularPricePredictor:
    """Price predictor using only structured/tabular features"""
    
    def __init__(self, model_type='random_forest'):
        """
        Args:
            model_type: 'random_forest' or 'gradient_boosting'
        """
        self.scaler = RobustScaler()
        self.model = None
        self.model_type = model_type
        self.feature_names = None
        
    def prepare_features(self, df):
        """Prepare tabular features from DataFrame"""
        # One-hot encoding with fixed categories
        property_types = ['Apartment', 'Villa', 'Townhouse', 'Studio', 'Duplex']
        furnishing_types = ['Unfurnished', 'Semi-furnished', 'Furnished']
        
        features = []
        feature_names = []
        
        # 1. Numeric features
        features.append(df['area_sqm'].fillna(df['area_sqm'].median()))
        feature_names.append('area_sqm')
        
        features.append(df['bedrooms'].fillna(0))
        feature_names.append('bedrooms')
        
        features.append(df['bathrooms'].fillna(1))
        feature_names.append('bathrooms')
        
        # 2. Property type one-hot
        for ptype in property_types:
            features.append((df['property_type'] == ptype).astype(int))
            feature_names.append(f'property_type_{ptype}')
        
        # 3. Furnishing one-hot
        for furn in furnishing_types:
            features.append((df['furnishing'] == furn).astype(int))
            feature_names.append(f'furnishing_{furn}')
        
        # 4. Derived features
        # Log area captures non-linear relationship
        log_area = np.log1p(df['area_sqm'].fillna(100))
        features.append(log_area)
        feature_names.append('log_area')
        
        # Bedroom ratio (beds per sqm)
        bed_ratio = df['bedrooms'] / df['area_sqm'].clip(lower=1)
        features.append(bed_ratio.fillna(0))
        feature_names.append('bedroom_ratio')
        
        # Bathroom ratio
        bath_ratio = df['bathrooms'] / df['area_sqm'].clip(lower=1)
        features.append(bath_ratio.fillna(0))
        feature_names.append('bathroom_ratio')
        
        self.feature_names = feature_names
        
        X = pd.concat(features, axis=1).values
        y = df['price'].values
        
        # Log transform target for better distribution
        y_log = np.log1p(y)
        
        return X, y_log, y
    
    def prepare_single_features(self, area_sqm, bedrooms, bathrooms, 
                                 property_type, furnishing, amenities=None,
                                 condition=None, view=None, floor_level=None):
        """Prepare features for a single prediction with structured inputs"""
        # Create a single-row dict
        data = {
            'area_sqm': area_sqm,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'property_type': property_type,
            'furnishing': furnishing,
        }
        df = pd.DataFrame([data])
        
        # Same feature engineering as training
        features = []
        
        # Numeric
        features.append(df['area_sqm'].iloc[0])
        features.append(df['bedrooms'].iloc[0])
        features.append(df['bathrooms'].iloc[0])
        
        # Property type one-hot (fixed order)
        property_types = ['Apartment', 'Villa', 'Townhouse', 'Studio', 'Duplex']
        for ptype in property_types:
            features.append(1 if property_type == ptype else 0)
        
        # Furnishing one-hot (fixed order)
        furnishing_types = ['Unfurnished', 'Semi-furnished', 'Furnished']
        for furn in furnishing_types:
            features.append(1 if furnishing == furn else 0)
        
        # Derived features
        log_area = np.log1p(area_sqm)
        features.append(log_area)
        
        bed_ratio = bedrooms / max(area_sqm, 1)
        features.append(bed_ratio)
        
        bath_ratio = bathrooms / max(area_sqm, 1)
        features.append(bath_ratio)
        
        return np.array([features])
    
    def train(self, df, test_size=0.2):
        """Train the model"""
        print("\n" + "="*60)
        print("Training Price Prediction Model")
        print("="*60)
        
        X, y_log, y_original = self.prepare_features(df)
        
        # Split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_log, test_size=test_size, random_state=42
        )
        
        # Fit scaler on training data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Train model
        print(f"\nTraining {self.model_type}...")
        if self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=150,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:  # gradient_boosting
            self.model = GradientBoostingRegressor(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred_log = self.model.predict(X_val_scaled)
        y_pred = np.expm1(y_pred_log)
        y_actual = np.expm1(y_val)
        
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        r2 = r2_score(y_actual, y_pred)
        
        print(f"\nEvaluation Results:")
        print(f"   MAE: {mae:,.0f} EGP")
        print(f"   RMSE: {rmse:,.0f} EGP")
        print(f"   R² Score: {r2:.4f}")
        print(f"   Features: {X.shape[1]}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            print(f"\nTop Features:")
            for i, idx in enumerate(indices[:10]):
                name = self.feature_names[idx] if self.feature_names else f"Feature {idx}"
                print(f"   {i+1}. {name}: {importances[idx]:.4f}")
        
        return self.model
    
    def predict(self, area_sqm, bedrooms, bathrooms, property_type, 
                furnishing, **kwargs):
        """Predict price for a single property
        
        Args:
            area_sqm: Property area in square meters
            bedrooms: Number of bedrooms
            bathrooms: Number of bathrooms
            property_type: One of ['Apartment', 'Villa', 'Townhouse', 'Studio', 'Duplex']
            furnishing: One of ['Unfurnished', 'Semi-furnished', 'Furnished']
            **kwargs: Ignored (for API compatibility)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Prepare features
        X = self.prepare_single_features(
            area_sqm, bedrooms, bathrooms, property_type, furnishing
        )
        
        # Scale
        X_scaled = self.scaler.transform(X)
        
        # Predict
        price_log = self.model.predict(X_scaled)[0]
        price = np.expm1(price_log)
        
        return float(price)
    
    def save(self, path='models/price_predictor.pkl'):
        """Save model"""
        os.makedirs('models', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type,
            }, f)
        print(f"Model saved to {path}")
    
    def load(self, path='models/price_predictor.pkl'):
        """Load model"""
        if not os.path.exists(path):
            print(f"Model not found at {path}")
            return False
        
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_names = data['feature_names']
        self.model_type = data.get('model_type', 'random_forest')
        
        print(f"Model loaded from {path}")
        return True


# Legacy wrapper for API compatibility
class PricePredictor(TabularPricePredictor):
    """Wrapper for backward compatibility"""
    def __init__(self):
        super().__init__(model_type='random_forest')
    
    def predict(self, description, area_sqm, bedrooms, bathrooms, 
                property_type="Apartment", furnishing="Unfurnished"):
        """Predict price (ignores description)"""
        return super().predict(
            area_sqm=area_sqm,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            property_type=property_type,
            furnishing=furnishing
        )
