# Bus Travel Time Prediction - Streamlit Interface

A professional web-based interface for predicting bus journey travel times using a trained Random Forest machine learning model.

## Features

✅ **Interactive Route Selection** - Choose from available bus routes and directions  
✅ **Stop Selection** - Select your starting and destination stops  
✅ **Time Selection** - Pick your departure time with rush hour detection  
✅ **Real-time Predictions** - Get accurate travel time predictions in seconds  
✅ **Detailed Analysis** - View segment breakdown, average speed, and model confidence  
✅ **Visual Route Map** - See your journey path on an interactive map  
✅ **Model Metrics** - Transparent display of R² score, MAE, and error ranges

## Installation

1. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Verify Model Files Exist:**
   Ensure the following model files are present in `/Users/dikshanta/Documents/Assignment-Big-Data/Model/`:
   - `rf_model_latest/`
   - `scaler_latest/`
   - `assembler_latest/`
   - `model_metadata_20260207_101744.json`

3. **Verify Data File:**
   Ensure `combined.csv` exists in `/Users/dikshanta/Documents/Assignment-Big-Data/`

## Running the Application

Launch the Streamlit app:

```bash
cd /Users/dikshanta/Documents/Assignment-Big-Data/Interface
streamlit run app.py
```

The application will open automatically in your default web browser at `http://localhost:8501`

## Usage

1. **Select Route:** Choose a bus route from the dropdown menu
2. **Select Direction:** Pick Inbound or Outbound direction
3. **Choose Departure Time:** Use the slider to select your departure hour (0-23)
4. **Select Stops:** Choose your starting stop and destination stop
5. **Predict:** Click the "🔮 Predict Travel Time" button
6. **View Results:** See your predicted travel time, journey details, and route map

## Model Information

- **Algorithm:** Random Forest Regressor
- **Trees:** 100
- **Max Depth:** 15
- **R² Score:** ~0.84
- **MAE:** ~0.11 minutes per segment
- **Features:** 12 (including route, direction, distance, time, coordinates)

## Technology Stack

- **Frontend:** Streamlit
- **ML Framework:** PySpark MLlib
- **Visualization:** Folium (interactive maps)
- **Data Processing:** Pandas

## Notes

- The interface uses the same prediction logic as the CLI system
- All predictions are made using pre-trained models (no retraining)
- Rush hour detection: 7-9 AM and 4-6 PM
- Distance calculations use Haversine formula for geographic accuracy

## Troubleshooting

**If you see "Failed to load models":**

- Check that model files exist in the correct directory
- Verify PySpark is installed correctly

**If you see "Failed to load route data":**

- Ensure `combined.csv` exists in the parent directory
- Check file path in `app.py` line 118

**If the map doesn't display:**

- Ensure `streamlit-folium` is installed
- Check browser console for errors
