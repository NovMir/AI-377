# House Price Prediction System

A fullstack application for predicting house prices using Machine Learning, with a Flask ML service and Django dashboard or streamlit

## Project Structure
```
.
├── ml_service/         # Flask ML service for predictions
│   ├── data/          # Data directory
│   │   ├── raw/       # Original CSV files
│   │   └── processed/ # Processed and combined datasets
│   └── load_data.py   # Data processing script
└── dashboard/          # Django dashboard application
```

## Dataset Information

The system uses property data from four cities in Ontario:
- Oakville (454 properties)
- Guelph (320 properties)
- Mississauga (393 properties)
- Milton (436 properties)

### Data Processing Steps
1. Load individual city datasets
2. Combine all datasets into a single dataframe
3. Add city identification column
4. Handle missing values:
   - Fill missing price values with city means
   - Drop completely empty columns (zestimate, rentZestimate)
5. Save processed data to `ml_service/data/processed/combined_properties.csv`

### Dataset Structure
The combined dataset contains 1603 properties with the following columns:
- address: Property address
- zipCode: Postal code
- city: City name
- state: Province (ON)
- price: Property price (float64)
- bed: Number of bedrooms
- bath: Number of bathrooms
- sqft: Square footage
- pricePerSf: Price per square foot
- lotArea: Lot area
- lotAreaType: Lot area unit (sqft/acres)
- zillowUrl: Zillow listing URL
- latitude: Property latitude
- longitude: Property longitude
- homeType: Type of property
- imageUrl: Property image URL

## Setup Instructions

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Process the data:
```bash
cd ml_service
python load_data.py
```

```

```
