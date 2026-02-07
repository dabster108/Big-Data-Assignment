# Configuration file for Bus Travel Time Prediction System
# Modify these paths if your setup is different

# Path to the model directory
MODEL_DIR = '/Users/dikshanta/Documents/Assignment-Big-Data/Model'

# Path to the combined CSV data file
DATA_CSV = '/Users/dikshanta/Documents/Assignment-Big-Data/combined.csv'

# Model metadata filename
METADATA_FILENAME = 'model_metadata_20260207_101744.json'

# Spark configuration
SPARK_CONFIG = {
    'app_name': 'BusPredictionInterface',
    'master': 'local[*]',
    'driver_memory': '4g',
    'shuffle_partitions': '8'
}

# Rush hour definition (hours)
RUSH_HOUR_MORNING = (7, 9)  # 7 AM to 9 AM
RUSH_HOUR_EVENING = (16, 18)  # 4 PM to 6 PM

# Time of day categories
TIME_CATEGORIES = {
    'Morning': (6, 11),
    'Afternoon': (12, 16),
    'Evening': (17, 20),
    'Night': (21, 5)  # 9 PM to 5 AM
}

# Map configuration
MAP_DEFAULT_ZOOM = 13
MAP_WIDTH = 1200
MAP_HEIGHT = 500

# Segment filtering
MIN_SEGMENT_DISTANCE = 0.01  # km
MIN_SEGMENTS_PER_ROUTE = 2

# UI Configuration
APP_TITLE = "Bus Travel Time Predictor"
APP_ICON = "🚍"
PAGE_LAYOUT = "wide"
