"""
Model Training Pipeline
Run: python -m src.models.trainer
"""

import pandas as pd
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.predictor import TabularPricePredictor
from src.data.pipeline import load_dataset


def main():
    print("\n" + "="*60)
    print("REAL ESTATE PRICE PREDICTOR - TRAINING")
    print("="*60 + "\n")
    
    # Load dataset
    print("Loading dataset...")
    df = load_dataset()
    
    if df.empty:
        print("No data found!")
        print("   Run: python -m src.data.collector")
        print("   Or generate sample: python -c 'from src.data.collector import generate_sample_data; generate_sample_data(200)'")
        return
    
    print(f"Loaded {len(df)} properties")
    print(f"   Price range: {df['price'].min():,.0f} - {df['price'].max():,.0f} EGP")
    
    # Train model
    predictor = TabularPricePredictor(model_type='random_forest')
    predictor.train(df)
    predictor.save('models/price_predictor.pkl')
    
    print(f"\nModel saved to models/price_predictor.pkl")
    print("\nRun dashboard: streamlit run src/visualization/dashboard.py")


if __name__ == "__main__":
    main()
