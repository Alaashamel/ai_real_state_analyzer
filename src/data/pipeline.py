"""
Data Processing Pipeline - Clean and prepare real estate data
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from pathlib import Path


class DataPipeline:
    """End-to-end data processing pipeline"""
    
    def __init__(self, raw_dir='data/raw', processed_dir='data/processed'):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_json_files(self, pattern='*.json'):
        """Load all JSON files from raw directory"""
        all_data = []
        
        for filepath in self.raw_dir.glob(pattern):
            print(f"Loading: {filepath.name}")
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        all_data.extend(data)
                    else:
                        all_data.append(data)
            except Exception as e:
                print(f"  Error: {e}")
        
        print(f"Loaded {len(all_data)} total records")
        return all_data
    
    def to_dataframe(self, records):
        """Convert records to DataFrame with validation"""
        df = pd.DataFrame(records)
        
        # Ensure required columns exist
        required_cols = {
            'price': 0,
            'area_sqm': 0,
            'bedrooms': 0,
            'bathrooms': 1,
            'property_type': 'Apartment',
            'furnishing': 'Unfurnished',
            'location': 'Unknown',
        }
        
        for col, default in required_cols.items():
            if col not in df.columns:
                df[col] = default
        
        return df
    
    def clean_data(self, df):
        """Clean and validate dataset"""
        initial_count = len(df)
        print(f"Initial records: {initial_count}")
        
        # 1. Remove duplicates (by URL if available)
        if 'url' in df.columns:
            df = df.drop_duplicates(subset=['url'], keep='first')
        print(f"After deduplication: {len(df)} records")
        
        # 2. Filter valid prices
        df = df[(df['price'] >= 100000) & (df['price'] <= 100000000)]
        print(f"After price filter: {len(df)} records")
        
        # 3. Fix data types
        df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce').fillna(0).astype(int)
        df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce').fillna(1).astype(int)
        df['area_sqm'] = pd.to_numeric(df['area_sqm'], errors='coerce')
        
        # 4. Fill missing area with median
        area_median = df['area_sqm'].median()
        df['area_sqm'] = df['area_sqm'].fillna(area_median)
        
        # 5. Standardize categorical values
        df['property_type'] = df['property_type'].fillna('Apartment')
        df['furnishing'] = df['furnishing'].fillna('Unfurnished')
        df['location'] = df['location'].fillna('Unknown')
        
        # Normalize property types
        type_mapping = {
            'villa': 'Villa',
            'Villa': 'Villa',
            'apartment': 'Apartment',
            'Apartment': 'Apartment',
            'studio': 'Studio',
            'Studio': 'Studio',
            'townhouse': 'Townhouse',
            'Townhouse': 'Townhouse',
            'duplex': 'Duplex',
            'Duplex': 'Duplex',
        }
        df['property_type'] = df['property_type'].map(lambda x: type_mapping.get(str(x).title(), 'Apartment'))
        
        # Normalize furnishing
        furn_mapping = {
            'fully': 'Furnished',
            'furnished': 'Furnished',
            'semi': 'Semi-furnished',
            'semi-furnished': 'Semi-furnished',
            'unfurnished': 'Unfurnished',
            'none': 'Unfurnished',
        }
        df['furnishing'] = df['furnishing'].map(lambda x: furn_mapping.get(str(x).lower(), 'Unfurnished'))
        
        # 6. Remove impossible values
        df = df[df['area_sqm'] >= 20]  # Min 20 sqm
        df = df[df['bedrooms'] >= 0]
        df = df[df['bathrooms'] >= 1]
        
        # 7. Add derived features
        df['price_per_sqm'] = df['price'] / df['area_sqm']
        df['log_price'] = np.log1p(df['price'])
        
        # 8. Extract city from location
        df['city'] = df['location'].apply(self._extract_city)
        
        final_count = len(df)
        print(f"Final clean dataset: {final_count} records ({initial_count - final_count} removed)")
        
        return df
    
    def _extract_city(self, location):
        """Extract city name from location string"""
        if not isinstance(location, str):
            return 'Cairo'
        
        cities = ['Cairo', 'Alexandria', 'Giza', 'New Cairo', 'Sheikh Zayed',
                  '6th October', 'Maadi', 'Heliopolis', 'Nasr City', 'El Shorouk']
        
        location_lower = location.lower()
        for city in cities:
            if city.lower() in location_lower:
                return city
        return 'Cairo'
    
    def save_csv(self, df, filename='final_dataset.csv'):
        """Save processed dataset"""
        filepath = self.processed_dir / filename
        df.to_csv(filepath, index=False)
        print(f"Saved processed data -> {filepath}")
        return filepath
    
    def run(self):
        """Execute full pipeline"""
        print("\n" + "="*60)
        print("DATA PROCESSING PIPELINE")
        print("="*60 + "\n")
        
        # Load
        records = self.load_json_files()
        if not records:
            print("No data found. Run scraping or generate sample data first.")
            return None
        
        # Transform
        df = self.to_dataframe(records)
        df_clean = self.clean_data(df)
        
        # Save
        self.save_csv(df_clean)
        
        # Summary
        print(f"\nDataset Summary:")
        print(f"  Total properties: {len(df_clean)}")
        print(f"  Price range: {df_clean['price'].min():,.0f} - {df_clean['price'].max():,.0f} EGP")
        print(f"  Avg price: {df_clean['price'].mean():,.0f} EGP")
        print(f"  Property types: {df_clean['property_type'].value_counts().to_dict()}")
        print(f"  Cities: {df_clean['city'].nunique()}")
        
        return df_clean


def load_dataset(filepath='data/processed/final_dataset.csv'):
    """Quick load function"""
    return pd.read_csv(filepath)
