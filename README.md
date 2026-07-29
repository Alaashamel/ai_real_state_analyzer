# Real Estate Valuation AI

![License](https://img.shields.io/badge/license-MIT-blue.svg) ![Language](https://img.shields.io/badge/language-Python-informational.svg) ![Last Commit](https://img.shields.io/github/last-commit/Alaashamel/ai_real_state_analyzer)


Professional machine learning system for predicting residential property values using structured features.

## Features

- **Structured Inputs**: Property type, area, bedrooms, bathrooms, amenities, condition, view, floor
- **Tabular Model**: Random Forest trained on 200+ features (no text bloat)
- **Interactive Dashboard**: Streamlit web app with real-time predictions
- **Clean Architecture**: Modular codebase with clear separation of concerns
- **Full Pipeline**: Data collection → processing → training → deployment

## Project Structure

```
my_real_estate_project/
├── app.py                    # Entry point (run: streamlit run app.py)
├── notebooks/               # Jupyter notebooks for EDA & training
│   ├── 00_quick_start.ipynb
│   ├── 01_data_acquisition.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_model_evaluation.ipynb
├── src/
│   ├── data/
│   │   ├── collector.py     # Web scraper (Aqarmap)
│   │   └── pipeline.py      # Data cleaning & preprocessing
│   ├── models/
│   │   ├── predictor.py     # Price predictor (sklearn wrapper)
│   │   └── trainer.py       # Training pipeline
│   ├── visualization/
│   │   └── dashboard.py     # Streamlit web app
│   └── utils/               # Helper functions
├── data/
│   ├── raw/                 # Raw scraped JSON data
│   └── processed/           # Cleaned CSV
├── models/                  # Trained models (.pkl files)
└── requirements.txt         # Python dependencies
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Data

**Option A: Generate sample data** (fast, for testing)
```bash
python -c "from src.data.collector import generate_sample_data; generate_sample_data(300)"
```

**Option B: Live scraping** (needs selector updates)
```bash
# Edit src/data/collector.py to configure URLs
python -m src.data.collector
```

### 3. Process Data

```bash
python -m src.data.pipeline
```

### 4. Train Model

```bash
python -m src.models.trainer
```

### 5. Launch Dashboard

```bash
streamlit run app.py
```

Open browser at http://localhost:8501

## Dashboard Features

### Valuation Tab
- **Structured inputs**: Sliders, dropdowns, multi-select for amenities
- **Real-time prediction**: Adjust any feature and see price change instantly
- **Feature breakdown**: Shows impact of amenities, condition, view, floor level
- **Confidence estimate**: Based on dataset coverage

### Analytics Tab
- Market overview metrics (count, avg price, avg price/m²)
- Price distribution histogram
- Price vs area scatter plot
- Trend analysis

### Property Search Tab
- Filter by price, area, bedrooms, property type
- Browse matching listings with full details

### System Tab
- Model status and configuration
- Quick command reference
- Dataset statistics

## Model Details

### Algorithm
- **Random Forest Regressor** (sklearn)
- 150 estimators, max depth 20
- RobustScaler normalization

### Features (10 structured inputs)
| Feature | Type | Description |
|---------|------|-------------|
| area_sqm | numeric | Property size in square meters |
| bedrooms | numeric | Number of bedrooms |
| bathrooms | numeric | Number of bathrooms |
| property_type | categorical | Apartment/Villa/Townhouse/Studio/Duplex |
| furnishing | categorical | Unfurnished/Semi-furnished/Furnished |
| log_area | derived | log(area+1) for non-linear scaling |
| bedroom_ratio | derived | bedrooms per sqm |
| bathroom_ratio | derived | bathrooms per sqm |
| property_type (one-hot) | 5 binary columns | |
| furnishing (one-hot) | 3 binary columns | |

### Performance
- R²: 0.70-0.80 (depends on data quality)
- MAE: ~500,000 EGP
- Trained on 200-500 properties

## Data Schema

### Required CSV columns
- `price`: Numeric price in EGP
- `area_sqm`: Property area
- `bedrooms`: Integer ≥ 0
- `bathrooms`: Integer ≥ 1
- `property_type`: One of the 5 categories
- `furnishing`: One of the 3 levels
- `location`: String (city/district)

Optional: `description`, `images`, `url`, `city`

## Customization

### Change Model
Edit `src/models/predictor.py`:
```python
predictor = TabularPricePredictor(model_type='gradient_boosting')
# or 'random_forest'
```

### Add New Features
1. Add feature engineering in `TabularPricePredictor.prepare_features()`
2. Add UI controls in `dashboard.py` Valuation tab
3. Update `prepare_single_features()` to include new feature

### Adjust Price Multipliers
Edit `dashboard.py` lines in valuation section:
```python
amenity_weights = {"Pool": 0.08, "Gym": 0.05, ...}
condition_boost = {"Excellent": 0.08, "Good": 0.03, ...}
view_boost = {"Sea": 0.12, "City": 0.05, ...}
floor_boost = {"Top/Penthouse": 0.10, "High": 0.05, ...}
```

## Troubleshooting

**Model not found**
→ Run training first: `python -m src.models.trainer`

**No data available**
→ Generate sample: `python -c "from src.data.collector import generate_sample_data; generate_sample_data()"`

**Streamlit won't start**
→ Update Streamlit: `pip install --upgrade streamlit`

**Port 8501 in use**
→ Change port: `streamlit run app.py --server.port 8502`

## License

MIT
