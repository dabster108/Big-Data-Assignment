"""
Smart Timetable Travel Time Predictor - Premium Interface
=========================================================
AI-powered bus journey prediction with real route intelligence
"""

import streamlit as st
import pandas as pd
import json
import os
import math
import folium
from streamlit_folium import st_folium
from datetime import datetime
from pyspark.sql import SparkSession, Row
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml.feature import StandardScalerModel, VectorAssembler

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Smart Travel Time Predictor",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# PREMIUM CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Custom Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: rgba(255,255,255,0.05);
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 3rem 2rem 2rem 2rem;
        background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
        border-radius: 20px;
        margin-bottom: 2rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
        animation: fadeIn 0.8s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
        letter-spacing: -1px;
        text-shadow: 0 0 30px rgba(102,126,234,0.3);
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255,255,255,0.8);
        font-weight: 300;
        margin-bottom: 0.5rem;
    }
    
    .hero-badge {
        display: inline-block;
        background: rgba(102,126,234,0.2);
        color: #667eea;
        padding: 0.5rem 1.5rem;
        border-radius: 50px;
        font-size: 0.9rem;
        font-weight: 600;
        border: 1px solid rgba(102,126,234,0.3);
        margin-top: 1rem;
    }
    
    /* Glass Card Styles */
    .glass-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 25px;
        padding: 2rem;
        border: 1px solid rgba(255,255,255,0.1);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(102,126,234,0.2);
        border-color: rgba(102,126,234,0.3);
    }
    
    /* Map Container with Curved Border */
    .map-container {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
        border-radius: 30px;
        padding: 1.5rem;
        border: 2px solid rgba(102,126,234,0.3);
        box-shadow: 0 10px 40px rgba(102,126,234,0.2);
        margin: 1.5rem 0;
        overflow: hidden;
        width: 100%;
    }
    
    .map-container iframe {
        border-radius: 20px !important;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .section-icon {
        font-size: 1.8rem;
    }
    
    /* Input Styling */
    .stSelectbox label, .stRadio label, .stSlider label {
        color: rgba(255,255,255,0.9) !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(255,255,255,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 15px;
        color: white;
    }
    
    .stRadio > div {
        background: rgba(255,255,255,0.05);
        padding: 1rem;
        border-radius: 15px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.2rem;
        font-weight: 700;
        border-radius: 15px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 10px 30px rgba(102,126,234,0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 40px rgba(102,126,234,0.5);
    }
    
    /* Result Card */
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 20px 60px rgba(102,126,234,0.4);
        animation: slideUp 0.5s ease;
    }
    
    @keyframes slideUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-time {
        font-size: 4rem;
        font-weight: 900;
        color: white;
        margin: 0;
        text-shadow: 0 5px 20px rgba(0,0,0,0.3);
    }
    
    .result-label {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        font-weight: 400;
        margin-top: 0.5rem;
    }
    
    /* Metric Cards */
    .metric-box {
        background: rgba(255,255,255,0.08);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 20px;
        border: 1px solid rgba(255,255,255,0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-box:hover {
        background: rgba(255,255,255,0.12);
        border-color: rgba(102,126,234,0.5);
        transform: scale(1.05);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: rgba(255,255,255,0.7);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Alert Boxes */
    .alert-box {
        padding: 1rem 1.5rem;
        border-radius: 20px;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
        animation: slideIn 0.4s ease;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .alert-warning {
        background: rgba(245,158,11,0.15);
        border-left: 4px solid #F59E0B;
        color: #FCD34D;
    }
    
    .alert-success {
        background: rgba(16,185,129,0.15);
        border-left: 4px solid #10B981;
        color: #6EE7B7;
    }
    
    .alert-info {
        background: rgba(102,126,234,0.15);
        border-left: 4px solid #667eea;
        color: #A5B4FC;
    }
    
    /* Table Styling */
    .dataframe {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 10px !important;
    }
    
    .dataframe th {
        background: rgba(102,126,234,0.2) !important;
        color: white !important;
        font-weight: 600 !important;
    }
    
    .dataframe td {
        color: rgba(255,255,255,0.9) !important;
    }
    
    /* Divider */
    .custom-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(102,126,234,0.5), transparent);
        margin: 2rem 0;
    }
    
    /* Spinner Override */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SPARK SESSION
# ============================================================================
@st.cache_resource(show_spinner=False)
def init_spark():
    """Initialize Spark session"""
    spark = SparkSession.builder \
        .appName("BusPredictionInterface") \
        .master("local[*]") \
        .config("spark.driver.memory", "4g") \
        .config("spark.ui.showConsoleProgress", "false") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

# ============================================================================
# LOAD MODELS AND DATA
# ============================================================================
@st.cache_resource(show_spinner=False)
def load_models():
    """Load trained models from disk - cached for performance"""
    model_dir = '/Users/dikshanta/Documents/Assignment-Big-Data/Model'
    
    try:
        rf_model = RandomForestRegressionModel.load(os.path.join(model_dir, 'rf_model_latest'))
        scaler_model = StandardScalerModel.load(os.path.join(model_dir, 'scaler_latest'))
        assembler = VectorAssembler.load(os.path.join(model_dir, 'assembler_latest'))
        
        # Load metadata
        with open(os.path.join(model_dir, 'model_metadata_20260207_101744.json'), 'r') as f:
            metadata = json.load(f)
        
        return rf_model, scaler_model, assembler, metadata
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

@st.cache_data(show_spinner=False)
def load_route_data():
    """Load and prepare route data from the processed dataset - cached for performance"""
    csv_path = '/Users/dikshanta/Documents/Assignment-Big-Data/Obtained_Dataset/combined.csv'
    
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        st.error(f"Error loading route data: {e}")
        return None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def calculate_haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates using Haversine formula"""
    R = 6371  # Earth's radius in km
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)
    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r
    a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def get_time_of_day(hour):
    """Convert hour to time of day category"""
    if 6 <= hour <= 11:
        return 'Morning'
    elif 12 <= hour <= 16:
        return 'Afternoon'
    elif 17 <= hour <= 20:
        return 'Evening'
    else:
        return 'Night'

def is_rush_hour(hour):
    """Check if given hour is rush hour"""
    return 1 if (7 <= hour <= 9) or (16 <= hour <= 18) else 0

def build_route_database(df):
    """Build route database with valid combinations"""
    import builtins
    
    # Create encoding mappings on-the-fly
    unique_routes = sorted(df['LineName'].unique())
    route_encoding = {route: float(i) for i, route in enumerate(unique_routes)}
    
    unique_directions = sorted(df['Direction'].unique())
    direction_encoding = {direction: float(i) for i, direction in enumerate(unique_directions)}
    
    # Time of day encoding
    timeofday_categories = ['Morning', 'Afternoon', 'Evening', 'Night']
    timeofday_encoding = {category: float(i) for i, category in enumerate(timeofday_categories)}
    
    # Get route sequences
    required_cols = ['LineName', 'Direction', 'JourneyCode', 'Sequence', 
                     'FromStopRef', 'FromStopName', 'FromLat', 'FromLon',
                     'ToStopRef', 'ToStopName', 'ToLat', 'ToLon']
    route_sequences = df[required_cols].copy()
    
    # Build valid combinations
    valid_combinations = {}
    
    for (route, direction), group in route_sequences.groupby(['LineName', 'Direction']):
        journey_counts = group['JourneyCode'].value_counts()
        if len(journey_counts) == 0:
            continue
        
        representative_journey = journey_counts.index[0]
        journey = group[group['JourneyCode'] == representative_journey].sort_values('Sequence')
        
        meaningful_segments = 0
        for _, row in journey.iterrows():
            dist = calculate_haversine_distance(
                row['FromLat'], row['FromLon'],
                row['ToLat'], row['ToLon']
            )
            if dist > 0.01:
                meaningful_segments += 1
        
        if meaningful_segments >= 2:
            if route not in valid_combinations:
                valid_combinations[route] = {}
            valid_combinations[route][direction] = meaningful_segments
    
    return route_sequences, valid_combinations, route_encoding, direction_encoding, timeofday_encoding

def get_route_segments(route_sequences, route_name, direction):
    """Get all meaningful unique segments for a route"""
    route_data = route_sequences[
        (route_sequences['LineName'] == route_name) & 
        (route_sequences['Direction'] == direction)
    ].copy()
    
    if len(route_data) == 0:
        return [], []
    
    # Get representative journey
    journey_counts = route_data['JourneyCode'].value_counts()
    representative_journey = journey_counts.index[0]
    journey = route_data[route_data['JourneyCode'] == representative_journey].sort_values('Sequence')
    
    # Build segments and stops
    segments = []
    stops = []
    seen_stops = set()
    seen_segments = set()
    
    for _, row in journey.iterrows():
        dist = calculate_haversine_distance(
            row['FromLat'], row['FromLon'],
            row['ToLat'], row['ToLon']
        )
        
        if dist > 0.01:
            segment_key = (row['FromStopRef'], row['ToStopRef'])
            
            if segment_key not in seen_segments:
                segments.append({
                    'Sequence': row['Sequence'],
                    'FromStopRef': row['FromStopRef'],
                    'FromStopName': row['FromStopName'],
                    'FromLat': row['FromLat'],
                    'FromLon': row['FromLon'],
                    'ToStopRef': row['ToStopRef'],
                    'ToStopName': row['ToStopName'],
                    'ToLat': row['ToLat'],
                    'ToLon': row['ToLon'],
                    'Distance': dist
                })
                seen_segments.add(segment_key)
            
            if row['FromStopRef'] not in seen_stops:
                stops.append({
                    'StopRef': row['FromStopRef'],
                    'StopName': row['FromStopName'],
                    'Lat': row['FromLat'],
                    'Lon': row['FromLon']
                })
                seen_stops.add(row['FromStopRef'])
    
    # Add final TO stop
    if len(segments) > 0:
        last_seg = segments[-1]
        if last_seg['ToStopRef'] not in seen_stops:
            stops.append({
                'StopRef': last_seg['ToStopRef'],
                'StopName': last_seg['ToStopName'],
                'Lat': last_seg['ToLat'],
                'Lon': last_seg['ToLon']
            })
    
    return segments, stops

def predict_segment(spark, segment, route_name, direction, hour, 
                   route_encoding, direction_encoding, timeofday_encoding,
                   assembler, scaler_model, rf_model):
    """Predict travel time for a single segment"""
    distance = segment['Distance']
    rush_hour = is_rush_hour(hour)
    time_of_day = get_time_of_day(hour)
    
    line_encoded = route_encoding.get(route_name, 0)
    direction_encoded = direction_encoding.get(direction, 0)
    timing_encoded = 0
    timeofday_encoded = timeofday_encoding.get(time_of_day, 0)
    
    input_row = Row(
        LineName_Encoded=float(line_encoded),
        Direction_Encoded=float(direction_encoded),
        Sequence=float(segment['Sequence']),
        TimingStatus_Encoded=float(timing_encoded),
        Distance_km=float(distance),
        Hour=float(hour),
        IsRushHour=float(rush_hour),
        TimeOfDay_Encoded=float(timeofday_encoded),
        FromLat=float(segment['FromLat']),
        FromLon=float(segment['FromLon']),
        ToLat=float(segment['ToLat']),
        ToLon=float(segment['ToLon'])
    )
    
    input_df = spark.createDataFrame([input_row])
    input_assembled = assembler.transform(input_df)
    input_scaled = scaler_model.transform(input_assembled)
    prediction_result = rf_model.transform(input_scaled)
    
    return prediction_result.select('prediction').collect()[0][0]

def create_route_overview_map(stops, route_name, direction):
    """Create a folium map showing all stops on a route"""
    if len(stops) == 0:
        return None
    
    # Calculate center point
    all_lats = [stop['Lat'] for stop in stops]
    all_lons = [stop['Lon'] for stop in stops]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    
    # Create map with Google Maps-like default tiles
    m = folium.Map(
        location=[center_lat, center_lon], 
        zoom_start=12
    )
    
    # Draw lines between consecutive stops
    for i in range(len(stops) - 1):
        folium.PolyLine(
            locations=[
                [stops[i]['Lat'], stops[i]['Lon']],
                [stops[i+1]['Lat'], stops[i+1]['Lon']]
            ],
            color='#667eea',
            weight=3,
            opacity=0.8
        ).add_to(m)
    
    # Add markers for all stops
    for i, stop in enumerate(stops):
        # Color code: first stop green, last stop red, others blue
        if i == 0:
            icon_color = 'green'
            icon_name = 'play'
        elif i == len(stops) - 1:
            icon_color = 'red'
            icon_name = 'stop'
        else:
            icon_color = 'blue'
            icon_name = 'circle'
        
        folium.CircleMarker(
            location=[stop['Lat'], stop['Lon']],
            radius=6,
            popup=f"<b>{i+1}. {stop['StopName']}</b>",
            tooltip=stop['StopName'],
            color=icon_color,
            fill=True,
            fillColor=icon_color,
            fillOpacity=0.8
        ).add_to(m)
    
    return m

def create_route_map(segments, from_stop, to_stop, from_idx, to_idx):
    """Create a folium map showing the route"""
    # Calculate center point
    all_lats = [seg['FromLat'] for seg in segments] + [segments[-1]['ToLat']]
    all_lons = [seg['FromLon'] for seg in segments] + [segments[-1]['ToLon']]
    center_lat = sum(all_lats) / len(all_lats)
    center_lon = sum(all_lons) / len(all_lons)
    
    # Create map
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13)
    
    # Draw route segments
    for seg in segments:
        folium.PolyLine(
            locations=[
                [seg['FromLat'], seg['FromLon']],
                [seg['ToLat'], seg['ToLon']]
            ],
            color='blue',
            weight=4,
            opacity=0.7
        ).add_to(m)
    
    # Add start marker (green)
    folium.Marker(
        location=[from_stop['Lat'], from_stop['Lon']],
        popup=f"<b>Start:</b> {from_stop['StopName']}",
        icon=folium.Icon(color='green', icon='play')
    ).add_to(m)
    
    # Add end marker (red)
    folium.Marker(
        location=[to_stop['Lat'], to_stop['Lon']],
        popup=f"<b>Destination:</b> {to_stop['StopName']}",
        icon=folium.Icon(color='red', icon='stop')
    ).add_to(m)
    
    return m

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Initialize session state for first load tracking
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False
    
    # ========================================================================
    # HERO SECTION - WOW FACTOR
    # ========================================================================
    st.markdown("""
    <div class="hero-container">
        <div class="hero-title">Smart Timetable Travel Time Predictor</div>
        <div class="hero-subtitle">AI-powered bus journey prediction with real route intelligence</div>
        <div class="hero-badge">🚀 Random Forest ML · 100 Trees · 84% Accuracy</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models and data (silent caching, only on first load)
    if not st.session_state.models_loaded:
        spark = init_spark()
        rf_model, scaler_model, assembler, metadata = load_models()
        
        if rf_model is None:
            st.markdown('<div class="alert-box alert-warning">⚠️ Failed to load models. Please check the model directory.</div>', unsafe_allow_html=True)
            return
        
        df = load_route_data()
        
        if df is None:
            st.markdown('<div class="alert-box alert-warning">⚠️ Failed to load route data. Please check the data file path.</div>', unsafe_allow_html=True)
            return
        
        route_sequences, valid_combinations, route_encoding, direction_encoding, timeofday_encoding = build_route_database(df)
        
        # Store in session state
        st.session_state.models_loaded = True
        st.session_state.spark = spark
        st.session_state.rf_model = rf_model
        st.session_state.scaler_model = scaler_model
        st.session_state.assembler = assembler
        st.session_state.metadata = metadata
        st.session_state.route_sequences = route_sequences
        st.session_state.valid_combinations = valid_combinations
        st.session_state.route_encoding = route_encoding
        st.session_state.direction_encoding = direction_encoding
        st.session_state.timeofday_encoding = timeofday_encoding
    else:
        # Load from session state (instant)
        spark = st.session_state.spark
        rf_model = st.session_state.rf_model
        scaler_model = st.session_state.scaler_model
        assembler = st.session_state.assembler
        metadata = st.session_state.metadata
        route_sequences = st.session_state.route_sequences
        valid_combinations = st.session_state.valid_combinations
        route_encoding = st.session_state.route_encoding
        direction_encoding = st.session_state.direction_encoding
        timeofday_encoding = st.session_state.timeofday_encoding
    
    st.markdown(f'<div class="alert-box alert-success">✓ System Ready: {len(valid_combinations)} routes loaded | Model R² = {metadata["r2_score"]:.4f}</div>', unsafe_allow_html=True)
    
    # Custom divider
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # ========================================================================
    # ROUTE VISUALIZATION - Show all stops on map
    # ========================================================================
    st.markdown('<div class="section-header"><span class="section-icon">🗺️</span>Route Visualization</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("**Select a route to visualize all available stops**")
        
        # Route selection for visualization
        sorted_routes = sorted(valid_combinations.keys())
        route_display_viz = []
        for route in sorted_routes:
            directions = list(valid_combinations[route].keys())
            direction_str = ", ".join(directions)
            route_display_viz.append(f"Route {route} ({direction_str})")
        
        col_viz1, col_viz2 = st.columns([2, 1], gap="medium")
        
        with col_viz1:
            selected_route_viz = st.selectbox(
                "Choose Route to Visualize",
                options=route_display_viz,
                key="route_visualization",
                label_visibility="collapsed"
            )
            route_num_viz = selected_route_viz.split()[1]
        
        with col_viz2:
            available_directions_viz = list(valid_combinations[route_num_viz].keys())
            if len(available_directions_viz) == 1:
                selected_direction_viz = available_directions_viz[0]
                st.info(f"📍 Direction: **{selected_direction_viz}**")
            else:
                selected_direction_viz = st.radio(
                    "Direction",
                    options=available_directions_viz,
                    horizontal=True,
                    key="direction_visualization",
                    label_visibility="collapsed"
                )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Get and display all stops for the selected route
    segments_viz, stops_viz = get_route_segments(route_sequences, route_num_viz, selected_direction_viz)
    
    if len(stops_viz) > 0:
        with st.container():
            st.markdown('<div class="map-container">', unsafe_allow_html=True)
            overview_map = create_route_overview_map(stops_viz, route_num_viz, selected_direction_viz)
            if overview_map:
                st_folium(overview_map, height=500, use_container_width=True, returned_objects=[])
            
            # Show stop count
            st.markdown(f"""
            <div style="text-align: center; color: rgba(255,255,255,0.8); margin-top: 1rem;">
                <strong>{len(stops_viz)}</strong> stops available on this route
            </div>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Custom divider
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    # ========================================================================
    # JOURNEY CONFIGURATION - PREMIUM GLASS CARD
    # ========================================================================
    st.markdown('<div class="section-header"><span class="section-icon">🎯</span>Journey Configuration</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="text-align: center; color: rgba(255,255,255,0.7); margin-bottom: 1.5rem;">Configure your journey details based on the route selected above</div>', unsafe_allow_html=True)
    
    # Use container for glass effect
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.markdown("**🚍 Selected Route**")
            st.markdown(f'<div class="alert-box alert-info">Route <strong>{route_num_viz}</strong> · Direction: <strong>{selected_direction_viz}</strong></div>', unsafe_allow_html=True)
            
            # Use the selected route from visualization
            route_num = route_num_viz
            selected_direction = selected_direction_viz
        
        with col2:
            st.markdown("**⏰ Departure Time**")
            
            hour = st.slider(
                "Hour",
                min_value=0,
                max_value=23,
                value=12,
                format="%d:00",
                label_visibility="collapsed"
            )
            
            time_of_day = get_time_of_day(hour)
            rush_hour_status = is_rush_hour(hour)
            
            # Time status display
            if rush_hour_status:
                st.markdown(f'<div class="alert-box alert-warning">🚨 <strong>RUSH HOUR</strong> · Expect increased travel time</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="alert-box alert-success">✓ <strong>{time_of_day.upper()}</strong> · Off-peak travel</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close glass-card
    
    # Get segments and stops
    segments, stops = get_route_segments(route_sequences, route_num, selected_direction)
    
    if len(stops) < 2:
        st.markdown('<div class="alert-box alert-warning">⚠️ Insufficient stops for this route. Please select another.</div>', unsafe_allow_html=True)
        return
    
    # Stop selection section
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span class="section-icon">📍</span>Stop Selection</div>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        
        col3, col4 = st.columns([1, 1], gap="large")
        
        with col3:
            st.markdown("**🟢 Starting Stop**")
            from_stop_name = st.selectbox(
                "From",
                options=[stop['StopName'] for stop in stops],
                label_visibility="collapsed"
            )
            from_idx = next(i for i, stop in enumerate(stops) if stop['StopName'] == from_stop_name)
            from_stop = stops[from_idx]
        
        with col4:
            st.markdown( "**🔴 Destination Stop**")
            available_to_stops = stops[from_idx + 1:]
            
            if len(available_to_stops) == 0:
                st.markdown('<div class="alert-box alert-warning">⚠️ No destinations available after this stop</div>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                return
            
            to_stop_name = st.selectbox(
                "To",
                options=[stop['StopName'] for stop in available_to_stops],
                label_visibility="collapsed"
            )
            to_stop = next(stop for stop in available_to_stops if stop['StopName'] == to_stop_name)
            to_idx = stops.index(to_stop)
        
        st.markdown('</div>', unsafe_allow_html=True)  # Close glass-card
    
    # ========================================================================
    # PREDICTION BUTTON - PREMIUM STYLE
    # ========================================================================
    st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
    
    predict_button = st.button("🚀 PREDICT TRAVEL TIME", type="primary", use_container_width=True)
    
    if predict_button:
        with st.spinner("🔄 Computing optimal journey time..."):
            # Find journey segments
            journey_segments = []
            for seg in segments:
                seg_from_idx = next((i for i, s in enumerate(stops) if s['StopRef'] == seg['FromStopRef']), None)
                if seg_from_idx is not None and from_idx <= seg_from_idx < to_idx:
                    journey_segments.append(seg)
            
            if len(journey_segments) == 0:
                st.markdown('<div class="alert-box alert-warning">⚠️ No valid segments found between these stops.</div>', unsafe_allow_html=True)
                return
            
            # Predict each segment
            total_time = 0
            total_distance = 0
            segment_details = []
            
            for segment in journey_segments:
                segment_time = predict_segment(
                    spark, segment, route_num, selected_direction, hour,
                    route_encoding, direction_encoding, timeofday_encoding,
                    assembler, scaler_model, rf_model
                )
                total_time += segment_time
                total_distance += segment['Distance']
                
                segment_details.append({
                    'from': segment['FromStopName'],
                    'to': segment['ToStopName'],
                    'time': segment_time,
                    'distance': segment['Distance']
                })
        
        # ====================================================================
        # RESULTS DISPLAY - PREMIUM LAYOUT
        # ====================================================================
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="section-icon">✨</span>Prediction Results</div>', unsafe_allow_html=True)
        
        # Main result - Hero Card
        mins = int(total_time)
        secs = int((total_time - mins) * 60)
        avg_speed = (total_distance / total_time) * 60 if total_time > 0 else 0
        
        st.markdown(f"""
        <div class="result-box">
            <div class="result-time">{mins}<span style="font-size: 2.5rem;">m</span> {secs}<span style="font-size: 2.5rem;">s</span></div>
            <div class="result-label">Predicted Travel Time: {total_time:.2f} minutes</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Journey details - Metric Cards
        st.markdown('<div style="margin-top: 2rem;"></div>', unsafe_allow_html=True)
        
        col5, col6, col7, col8 = st.columns(4, gap="medium")
        
        with col5:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{route_num}</div>
                <div class="metric-label">ROUTE</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col6:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{len(journey_segments)}</div>
                <div class="metric-label">SEGMENTS</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col7:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{total_distance:.1f}</div>
                <div class="metric-label">KM</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col8:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{avg_speed:.0f}</div>
                <div class="metric-label">KM/H</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Journey Info Card
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            
            col9, col10 = st.columns([1, 1], gap="large")
            
            with col9:
                st.markdown(f"""
                **📍 Journey Details**
                
                **From:** {from_stop['StopName']}  
                **To:** {to_stop['StopName']}  
                **Direction:** {selected_direction}  
                **Departure:** {hour:02d}:00 ({time_of_day})
                """)
            
            with col10:
                st.markdown(f"""
                **📊 Performance Metrics**
                
                **Model R²:** {metadata['r2_score']:.4f}  
                **MAE:** ±{metadata['mae']:.2f} min/segment  
                **Error Range:** ±{metadata['mae'] * len(journey_segments):.2f} min  
                **Confidence:** {metadata['r2_score']*100:.1f}%
                """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Segment breakdown
        if len(journey_segments) <= 15:
            st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="section-header"><span class="section-icon">📋</span>Segment Breakdown</div>', unsafe_allow_html=True)
            
            segment_df = pd.DataFrame([
                {
                    '#': i+1,
                    'From': seg['from'][:40],
                    'To': seg['to'][:40],
                    'Distance (km)': f"{seg['distance']:.2f}",
                    'Time (min)': f"{seg['time']:.2f}"
                }
                for i, seg in enumerate(segment_details)
            ])
            
            st.dataframe(segment_df, use_container_width=True, height=400)
        
        # Map visualization
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-header"><span class="section-icon">🗺️</span>Route Visualization</div>', unsafe_allow_html=True)
        
        route_map = create_route_map(journey_segments, from_stop, to_stop, from_idx, to_idx)
        st_folium(route_map, height=500, use_container_width=True, returned_objects=[])
        
        # Model info footer
        st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="alert-box alert-info">
            <strong>ℹ️ Model Intelligence:</strong> Random Forest · {metadata['num_trees']} Trees · 
            Max Depth {metadata['max_depth']} · {len(metadata['features'])} Features · 
            Training Time {metadata['training_time']:.1f}s
        </div>
        """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()
