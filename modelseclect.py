"""
Interactive Bus Runtime Prediction System
==========================================
Select stops from dropdown and get instant travel time predictions!
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

pd.set_option('display.max_columns', None)

print("="*70)
print("🚌 INTERACTIVE BUS RUNTIME PREDICTION SYSTEM")
print("="*70)

# ============================================================================
# 1. LOAD & PREPARE DATA
# ============================================================================
print("\n⏳ Loading and preparing data...")
df = pd.read_csv('combined.csv')

# Convert RunTime
df['RunTime_Minutes'] = df['RunTime'].str.extract(r'PT(\d+)M').astype(float)
df['RunTime_Minutes'].fillna(0, inplace=True)

# Calculate distance
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

df['Distance_km'] = haversine_distance(df['FromLat'], df['FromLon'], df['ToLat'], df['ToLon'])

# Extract time features
df['DepartureTime'] = pd.to_datetime(df['DepartureTime'], format='%H:%M:%S', errors='coerce')
df['Hour'] = df['DepartureTime'].dt.hour
df['IsRushHour'] = df['Hour'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 18) else 0)

# Clean data
df_clean = df[df['RunTime_Minutes'].notna()].copy()
df_clean = df_clean[df_clean['RunTime_Minutes'] > 0]

# Create unique stops reference
print("⏳ Building stops database...")
from_stops = df_clean[['FromStopRef', 'FromStopName', 'FromLat', 'FromLon']].drop_duplicates()
from_stops.columns = ['StopRef', 'StopName', 'Lat', 'Lon']

to_stops = df_clean[['ToStopRef', 'ToStopName', 'ToLat', 'ToLon']].drop_duplicates()
to_stops.columns = ['StopRef', 'StopName', 'Lat', 'Lon']

all_stops = pd.concat([from_stops, to_stops]).drop_duplicates(subset=['StopRef']).reset_index(drop=True)
all_stops = all_stops.sort_values('StopName').reset_index(drop=True)

print(f"✅ Loaded {len(df_clean):,} journey segments")
print(f"✅ Found {len(all_stops)} unique bus stops")

# ============================================================================
# 2. ENCODE & PREPARE FEATURES
# ============================================================================
print("\n⏳ Preparing ML features...")

le_line = LabelEncoder()
le_direction = LabelEncoder()
le_timing = LabelEncoder()

df_clean['LineName_Encoded'] = le_line.fit_transform(df_clean['LineName'].astype(str))
df_clean['Direction_Encoded'] = le_direction.fit_transform(df_clean['Direction'].astype(str))
df_clean['TimingStatus_Encoded'] = le_timing.fit_transform(df_clean['TimingStatus'].astype(str))

features = [
    'LineName_Encoded', 'Direction_Encoded', 'Sequence', 'TimingStatus_Encoded',
    'Distance_km', 'Hour', 'IsRushHour', 'FromLat', 'FromLon', 'ToLat', 'ToLon'
]

X = df_clean[features].copy()
y = df_clean['RunTime_Minutes'].copy()
X = X.fillna(X.median())

# ============================================================================
# 3. TRAIN MODEL
# ============================================================================
print("⏳ Training Random Forest model...")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=100, max_depth=15, min_samples_split=10,
    min_samples_leaf=5, random_state=42, n_jobs=-1, verbose=0
)

model.fit(X_train, y_train)

# Quick evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("✅ Model trained successfully!")
print(f"   Accuracy: {r2*100:.1f}% | Avg Error: {mae:.2f} minutes")

# Get unique routes and directions
unique_routes = sorted(df_clean['LineName'].unique())
unique_directions = sorted(df_clean['Direction'].unique())

# ============================================================================
# 4. INTERACTIVE PREDICTION LOOP
# ============================================================================
print("\n" + "="*70)
print("🎯 READY FOR PREDICTIONS!")
print("="*70)

def show_options(items, title):
    """Display numbered list and get user choice"""
    print(f"\n{title}")
    print("-" * 60)
    for i, item in enumerate(items, 1):
        print(f"  {i}. {item}")
    print("-" * 60)
    
    while True:
        try:
            choice = input(f"Enter choice (1-{len(items)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(items):
                return items[choice_num - 1]
            else:
                print(f"❌ Please enter a number between 1 and {len(items)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            exit(0)

def get_stop_choice(prompt):
    """Show stops with search functionality"""
    print(f"\n{prompt}")
    search = input("🔍 Search stop name (or press Enter to see all): ").strip()
    
    if search:
        filtered = all_stops[all_stops['StopName'].str.contains(search, case=False, na=False)]
        if len(filtered) == 0:
            print("❌ No stops found. Showing all stops...")
            filtered = all_stops
    else:
        filtered = all_stops
    
    # Show first 20 stops
    display_stops = filtered.head(20)
    stop_list = []
    
    print("\n" + "-" * 60)
    for i, row in display_stops.iterrows():
        display_text = f"{row['StopName']} (Ref: {row['StopRef']})"
        stop_list.append(row)
        print(f"  {len(stop_list)}. {display_text}")
    
    if len(filtered) > 20:
        print(f"  ... and {len(filtered) - 20} more stops")
        print("  💡 Use search to narrow down options")
    print("-" * 60)
    
    while True:
        try:
            choice = input(f"Enter choice (1-{len(stop_list)}): ").strip()
            choice_num = int(choice)
            if 1 <= choice_num <= len(stop_list):
                return stop_list[choice_num - 1]
            else:
                print(f"❌ Please enter a number between 1 and {len(stop_list)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except KeyboardInterrupt:
            print("\n\n👋 Exiting...")
            exit(0)

# Main prediction loop
while True:
    try:
        print("\n" + "="*70)
        print("📝 ENTER JOURNEY DETAILS")
        print("="*70)
        
        # 1. Select Route/Line
        print("\n1️⃣ SELECT BUS ROUTE")
        route_display = [f"Route {r}" for r in unique_routes]
        selected_route = show_options(route_display, "Available Routes:")
        route_num = selected_route.split()[1]
        print(f"✅ Selected: {selected_route}")
        
        # 2. Select Direction
        print("\n2️⃣ SELECT DIRECTION")
        selected_direction = show_options(unique_directions, "Available Directions:")
        print(f"✅ Selected: {selected_direction}")
        
        # 3. Select FROM stop
        print("\n3️⃣ SELECT STARTING STOP")
        from_stop = get_stop_choice("Choose where the bus will START from:")
        print(f"✅ From: {from_stop['StopName']}")
        
        # 4. Select TO stop
        print("\n4️⃣ SELECT DESTINATION STOP")
        to_stop = get_stop_choice("Choose where the bus will GO to:")
        print(f"✅ To: {to_stop['StopName']}")
        
        # 5. Enter time
        print("\n5️⃣ ENTER DEPARTURE TIME")
        while True:
            hour_input = input("Enter hour (0-23): ").strip()
            try:
                hour = int(hour_input)
                if 0 <= hour <= 23:
                    break
                else:
                    print("❌ Please enter hour between 0 and 23")
            except ValueError:
                print("❌ Please enter a valid number")
        
        # 6. Enter sequence
        print("\n6️⃣ ENTER STOP SEQUENCE")
        sequence_input = input("Stop number in route (e.g., 1, 5, 10): ").strip()
        try:
            sequence = int(sequence_input)
        except ValueError:
            sequence = 1
            print(f"⚠️ Using default: {sequence}")
        
        # Calculate distance
        distance = haversine_distance(
            from_stop['Lat'], from_stop['Lon'],
            to_stop['Lat'], to_stop['Lon']
        )
        
        is_rush_hour = 1 if (7 <= hour <= 9) or (16 <= hour <= 18) else 0
        
        # Encode inputs
        line_encoded = le_line.transform([str(route_num)])[0]
        direction_encoded = le_direction.transform([selected_direction])[0]
        timing_encoded = 0  # Default timing status
        
        # Prepare input
        input_data = pd.DataFrame({
            'LineName_Encoded': [line_encoded],
            'Direction_Encoded': [direction_encoded],
            'Sequence': [sequence],
            'TimingStatus_Encoded': [timing_encoded],
            'Distance_km': [distance],
            'Hour': [hour],
            'IsRushHour': [is_rush_hour],
            'FromLat': [from_stop['Lat']],
            'FromLon': [from_stop['Lon']],
            'ToLat': [to_stop['Lat']],
            'ToLon': [to_stop['Lon']]
        })
        
        # Make prediction
        predicted_time = model.predict(input_data)[0]
        
        # Display results
        print("\n" + "="*70)
        print("🎯 PREDICTION RESULT")
        print("="*70)
        
        print(f"\n📍 Journey Summary:")
        print(f"   Route:      {route_num}")
        print(f"   Direction:  {selected_direction}")
        print(f"   From:       {from_stop['StopName']}")
        print(f"   To:         {to_stop['StopName']}")
        print(f"   Distance:   {distance:.2f} km")
        print(f"   Time:       {hour:02d}:00 {'(Rush Hour)' if is_rush_hour else ''}")
        print(f"   Stop #:     {sequence}")
        
        print(f"\n⏱️  PREDICTED TRAVEL TIME: {predicted_time:.2f} minutes")
        
        # Convert to minutes and seconds
        mins = int(predicted_time)
        secs = int((predicted_time - mins) * 60)
        print(f"   ≈ {mins} minute(s) and {secs} seconds")
        
        # Description
        print(f"\n💡 What this means:")
        print(f"   The bus traveling from '{from_stop['StopName']}' to")
        print(f"   '{to_stop['StopName']}' on route {route_num} at {hour:02d}:00")
        print(f"   will take approximately {predicted_time:.1f} minutes.")
        
        if is_rush_hour:
            print(f"   ⚠️  This is during rush hour, so expect possible delays.")
        
        if distance < 0.5:
            print(f"   ✓ Short distance - quick journey!")
        elif distance > 3:
            print(f"   ⚠️  Long distance - allow extra time.")
        
        print(f"\n   Model confidence: {r2*100:.1f}%")
        print(f"   Average accuracy: ±{mae:.2f} minutes")
        
        print("\n" + "="*70)
        
        # Ask to continue
        print("\n")
        continue_choice = input("🔄 Make another prediction? (y/n): ").strip().lower()
        if continue_choice not in ['y', 'yes']:
            print("\n" + "="*70)
            print("👋 Thank you for using Bus Runtime Prediction System!")
            print("="*70)
            break
            
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("👋 Thank you for using Bus Runtime Prediction System!")
        print("="*70)
        break
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Let's try again...\n")
        continue
