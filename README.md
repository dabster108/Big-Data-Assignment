# Big Data Programming Project - Bus Timetable Analysis

## Assignment Information

- **Module Name:** Big Data Programming Project
- **Module Code:** ST5011CEM
- **Assignment Title:** Coursework
- **Assignment Due:** 27th January 2026 [11:55 PM]

## Student Information

- **Student ID:** 240226
- **Student Name:** Dikshanta Chapagain

---

## Project Overview

This project analyzes Falcon Bus timetable data to predict bus journey runtimes using machine learning techniques with PySpark. The dataset consists of TransXChange XML files containing bus route information, stop locations, schedules, and journey patterns.

**Current Status:** 🚧 In Progress - Core functionality implemented, additional features pending

---

## 📁 Project Structure

```
Assignment-Big-Data/
├── README.md                          # Project documentation
├── Converter(XML_CSV).py              # XML to CSV converter script
├── combined.csv                       # Processed dataset (output from converter)
├── TimeTableAnalysis.ipynb            # Main Jupyter notebook with analysis
├── modelseclect.py                    # Model selection utilities (pending)
├── Falcon_Bus_Dataset(XML)/           # Raw XML timetable files
│   ├── FALC_12_*.xml                 # Route 12 schedules
│   ├── FALC_28_*.xml                 # Route 28 schedules
│   ├── FALC_400_*.xml                # Route 400 schedules
│   └── ... (237 XML files total)
```

---

## 🔧 Technologies Used

- **Python 3.10**
- **PySpark 3.x** - Distributed data processing
- **Apache Spark** - Big data analytics engine
- **Java 8** (Temurin) - Required for PySpark
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **PySpark ML** - Machine learning models

---

## 🚀 Getting Started

### Prerequisites

1. **Java 8 Installation** (Required for PySpark)

   ```bash
   # macOS
   brew install --cask temurin@8
   ```

2. **Python Packages**
   ```bash
   pip install pyspark pandas numpy matplotlib seaborn
   ```

### Environment Setup

The notebook automatically configures Java and Spark paths:

```python
os.environ['JAVA_HOME'] = '/Library/Java/JavaVirtualMachines/temurin-8.jdk/Contents/Home'
os.environ['SPARK_HOME'] = '/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/pyspark'
```

---

## 📊 Data Pipeline

### 1. XML to CSV Conversion

**Script:** `Converter(XML_CSV).py`

#### Purpose

Converts TransXChange XML files (UK bus timetable standard) into a flat CSV format suitable for analysis.

#### How It Works

The converter extracts the following information from XML files:

**Stop Information:**

- Stop references, names, and coordinates (latitude/longitude)
- Sequence numbers for stop ordering

**Journey Details:**

- Operator name and route/line numbers
- Departure times and journey codes
- Direction (inbound/outbound)

**Timing Data:**

- Runtime between stops (PT format, e.g., "PT5M" = 5 minutes)
- Timing status (PTP - Principal Timing Point, TIP - Timing Information Point)
- Activity types at stops

**Operating Information:**

- School organization names (for school services)
- Operating date ranges

#### Usage

```python
# Configuration
xml_folder = "/path/to/Falcon_Bus_Dataset(XML)"
output_csv = "combined.csv"

# Run converter
process_xml_to_csv(xml_folder, output_csv)
```

**Input:** 237 XML files from `Falcon_Bus_Dataset(XML)/`  
**Output:** `combined.csv` with ~50,000+ rows

#### Output CSV Columns

| Column           | Description                         |
| ---------------- | ----------------------------------- |
| FileName         | Source XML filename                 |
| OperatorName     | Bus operator (Falcon Buses)         |
| LineName         | Route number (e.g., FALC_400)       |
| Direction        | inbound/outbound                    |
| DepartureTime    | Journey start time (HH:mm:ss)       |
| JourneyCode      | Unique journey identifier           |
| Sequence         | Stop sequence number                |
| Activity         | pickUp/setDown                      |
| TimingStatus     | PTP/TIP/etc.                        |
| RunTime          | Duration to next stop (PTxM format) |
| FromStopRef      | Origin stop ID                      |
| FromStopName     | Origin stop name                    |
| FromLat, FromLon | Origin coordinates                  |
| ToStopRef        | Destination stop ID                 |
| ToStopName       | Destination stop name               |
| ToLat, ToLon     | Destination coordinates             |
| SchoolOrgName    | School name (if applicable)         |
| OperatingDates   | Service dates                       |

---

### 2. Data Analysis & ML Pipeline

**Notebook:** `TimeTableAnalysis.ipynb`

#### Implemented Features ✅

##### **Data Ingestion**

- Load CSV with PySpark
- Schema inference and validation
- Convert to Pandas for visualization

##### **Data Cleaning & Preprocessing**

- **Runtime Conversion:** Parse "PT5M" format → 5.0 minutes
- **Distance Calculation:** Haversine formula for stop-to-stop distance
- **Feature Engineering:**
  - Extract hour and minute from departure time
  - Create `IsRushHour` flag (7-9 AM, 4-6 PM)
  - Encode categorical variables (LineName, Direction, TimingStatus)
- **Feature Scaling:** StandardScaler normalization
- **Data Validation:** Remove null/invalid records

##### **Machine Learning Model**

**Algorithm:** Random Forest Regressor

**Features Used:**

- `LineName_Encoded` - Route identifier
- `Direction_Encoded` - Travel direction
- `Sequence` - Stop order
- `TimingStatus_Encoded` - Stop timing classification
- `Distance_km` - Haversine distance
- `Hour` - Departure hour
- `IsRushHour` - Rush hour indicator
- `FromLat`, `FromLon`, `ToLat`, `ToLon` - GPS coordinates

**Model Configuration:**

- Number of trees: 100
- Max depth: 15
- Train/test split: 80/20
- Seed: 42 (for reproducibility)

**Performance Metrics:**

- MAE (Mean Absolute Error): ~0.2-0.5 minutes
- RMSE: ~0.5-1.0 minutes
- R² Score: ~0.85-0.95

##### **Interactive Prediction System**

- User selects origin/destination stops
- Choose route and direction
- Input departure time
- Get predicted runtime instantly

##### **Visualizations**

- Actual vs Predicted scatter plot
- Residual analysis
- Feature importance chart
- Error distribution histogram

---

## 🎯 Current Capabilities

### What Works ✅

1. **Data Conversion:** XML → CSV transformation
2. **Data Loading:** PySpark ingestion and schema detection
3. **Preprocessing:** Feature engineering and encoding
4. **Model Training:** Random Forest with 100 trees
5. **Predictions:** Runtime estimation for new journeys
6. **Evaluation:** Comprehensive metrics and visualizations

### Example Prediction

```
Journey Details:
  From: Coventry Pool Meadow Bus Station
  To: Falcon Lodge Drive
  Route: FALC_400
  Direction: outbound
  Time: 08:00
  Distance: 3.45 km
  Rush Hour: Yes

  PREDICTED RUNTIME: 4.32 minutes
```

---

## 📈 Next Steps / TODO

### Pending Features 🚧

1. **Model Comparison** (`modelseclect.py`)
   - Implement Linear Regression
   - Implement Gradient Boosting
   - Compare performance metrics
   - Select best model

2. **Advanced Analytics**
   - Peak hour analysis
   - Route efficiency scoring
   - Stop delay patterns
   - Seasonal variations

3. **Enhanced Predictions**
   - Real-time traffic integration
   - Weather impact analysis
   - Day-of-week effects
   - Holiday adjustments

4. **Visualization Dashboard**
   - Interactive route maps
   - Real-time prediction interface
   - Performance monitoring

5. **Model Optimization**
   - Hyperparameter tuning
   - Cross-validation
   - Ensemble methods

---

## 📝 How to Run

### Complete Workflow

```bash
# Step 1: Convert XML to CSV (if not already done)
python Converter(XML_CSV).py

# Step 2: Open Jupyter Notebook
jupyter notebook TimeTableAnalysis.ipynb

# Step 3: Run cells sequentially
# - Cell 1: Environment setup
# - Cell 2: Import libraries
# - Cell 3-4: Load data
# - Cell 5-6: View data in Pandas
# - Cell 7: Data preprocessing
# - Cell 8: Train-test split
# - Cell 9: Random Forest training
# - Cell 10: Interactive predictions
```

### Quick Test

```python
# In Python/Jupyter
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("BusAnalysis").getOrCreate()
df = spark.read.csv('combined.csv', header=True, inferSchema=True)

print(f"Total records: {df.count()}")
df.show(5)
```

---

## 🔍 Data Insights

### Dataset Statistics

- **Total Routes:** 50+ bus lines
- **Total Stops:** 200+ unique stops
- **Total Journeys:** 1000+ daily services
- **Time Coverage:** 05:00 - 23:00
- **Geographic Area:** Coventry and surrounding areas

### Key Findings

- Average runtime: 3-7 minutes between stops
- Rush hour impact: +15-25% longer runtimes
- Most frequent route: FALC_400
- Longest route: ~45 minutes end-to-end

---

## ⚠️ Known Issues

1. Some XML files have missing coordinates
2. Runtime parsing edge cases (PT0M values)
3. Memory usage high for full dataset in Pandas
4. Model retraining required if data changes

---

## 📚 References

- TransXChange Schema: UK bus timetable standard
- PySpark Documentation: [spark.apache.org](https://spark.apache.org)
- Haversine Formula: Geographic distance calculation
- Random Forest: Ensemble learning method

---

## 📧 Contact

**Student:** Dikshanta Chapagain  
**ID:** 240226  
**Module:** ST5011CEM - Big Data Programming Project

---

**Last Updated:** January 31, 2026  
**Status:** 🚧 Active Development
