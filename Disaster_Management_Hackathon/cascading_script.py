import zipfile
import glob
import os

# STEP 1: Define your zip filename (assuming it's in the same folder as this script)
zip_filename = "Smart_City_Dataset.zip"  

# STEP 2: Simulate your uploaded dict-style (like from Google Colab)
uploaded = {zip_filename: None}  # fake "uploaded" object like you'd get in Colab

# STEP 3: Use your original logic
zip_file = next(iter(uploaded))  # automatically gets the filename

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall("unzipped")  # Extracts all files into /unzipped

# STEP 4: List all CSV files recursively
all_csv_files = glob.glob("unzipped/**/*.csv", recursive=True)
# We merge the tables on timestamp/date
# Convert timestamp columns to datetime if needed
import pandas as pd
df_sensor = pd.read_csv(all_csv_files[6])
df_tweets = pd.read_csv(all_csv_files[7])

df_sensor['timestamp'] = pd.to_datetime(df_sensor['timestamp'])

# Filter to only include active sensors
df_sensor_clean = df_sensor[df_sensor['status'] == 'active']

# Optionally, filter to only sensor types relevant for weather comparison,
# For instance, temperature and humidity:
relevant_types = ['temp', 'humidity', 'seismic', 'flood']
df_sensor_clean = df_sensor_clean[df_sensor_clean['sensor_type'].isin(relevant_types)]
df_tweets['timestamp'] = pd.to_datetime(df_tweets['timestamp'])
# If timestamps match exactly:
# Merge based on closest timestamp (tolerance of 1 minute here; adjust as needed):
df_sensor_clean = df_sensor_clean.sort_values('timestamp')
df_weather = df_tweets.sort_values('timestamp')
df_merged = pd.merge(df_sensor_clean, df_weather, on='timestamp', how='inner')

df_merged = df_merged.dropna()
df_merged.rename(columns={'latitude_x': 'latitude_sensor'}, inplace=True)
df_merged.rename(columns={'longitude_x': 'longitude_sensor'}, inplace=True)
df_merged.rename(columns={'latitude_y': 'latitude_tweet'}, inplace=True)
df_merged.rename(columns={'longitude_y': 'longitude_tweet'}, inplace=True)

# we merge energy comsumption
df_energy = pd.read_csv(all_csv_files[3])
df_energy['timestamp'] = pd.to_datetime(df_energy['timestamp'])
df_merged = pd.merge(df_merged, df_energy, on='timestamp', how='inner')

# we change the forecast every hour to forecast every minute
df_weather = pd.read_csv(all_csv_files[8])
df_weather['timestamp'] = pd.to_datetime(df_weather['timestamp'])
df_weather.set_index('timestamp', inplace=True)
df_weather = df_weather.resample('min').ffill()
df_weather.reset_index()
df_merged = pd.merge(df_merged, df_weather, on='timestamp', how='inner')

df_disaster = pd.read_csv(all_csv_files[5])
df_disaster['timestamp'] = pd.to_datetime(df_disaster['date'])

df_merged = pd.get_dummies(df_merged, columns=['sensor_type', 'status'])

# Drop the columns and modify df_merged in place
df_merged.drop(columns=['sensor_id', 'user_id'], inplace=True)
df_merged.drop(columns=['building_id'], inplace=True)
df_disaster.drop(columns=['date'], inplace=True)

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

# Convert to GeoDataFrames
def create_gdf(df, lat_col='latitude', lon_col='longitude'):
    geometry = [Point(xy) for xy in zip(df[lon_col], df[lat_col])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
    return gdf

gdf1 = create_gdf(df_merged, lat_col='latitude_sensor', lon_col='longitude_sensor')
gdf2 = create_gdf(df_disaster)

# (Optional) Reproject to a metric CRS for accurate distance calculations (example: UTM)
gdf1 = gdf1.to_crs(epsg=32633)  # change the EPSG code as needed
gdf2 = gdf2.to_crs(epsg=32633)

# Perform a spatial join based on nearest neighbor within a maximum distance (e.g., 500 meters)
# Note: The max_distance parameter is in the CRS units (meters if using a metric CRS)
gdf_merged = gpd.sjoin_nearest(gdf1, gdf2, how="inner", max_distance=500, distance_col="dist")

gdf_merged = pd.get_dummies(gdf_merged, columns=['location'])

import numpy as np
import pandas as pd

# === Sklearn Imports ===
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

# === Hugging Face Transformers for GPT-2 ===
from transformers import GPT2LMHeadModel, GPT2Tokenizer

###################################################
# PART 0: Global Dynamic Threshold for Energy
###################################################
# Compute a dynamic power threshold based on the energy_kwh distribution in gdf_merged.
# For example, using the 10th percentile as the cutoff for a power outage.
DYNAMIC_POWER_THRESHOLD = np.percentile(gdf_merged['energy_kwh'], 10)
print("Dynamic Power Threshold (10th percentile):", DYNAMIC_POWER_THRESHOLD)

###################################################
# PART 1: DISASTER CLASSIFIER
###################################################
X_class = gdf_merged[['latitude', 'longitude', 
                      'location_Zone A', 'location_Zone B', 'location_Zone C', 'location_Zone D', 
                      'economic_loss_million_usd', 'reading_value']]
y_class = gdf_merged['disaster_type']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_class, y_class, test_size=0.2, random_state=42
)

rf_cls = RandomForestClassifier(n_estimators=200, random_state=42)
rf_cls.fit(X_train_cls, y_train_cls)

y_pred_cls = rf_cls.predict(X_test_cls)
print("=== Disaster Classifier Performance ===")
print(classification_report(y_test_cls, y_pred_cls))

###################################################
# PART 2: ENERGY REGRESSION MODEL
###################################################
# Use the classifier to predict disaster type across the dataset.
gdf_merged['predicted_disaster_type'] = rf_cls.predict(X_class)
disaster_dummies = pd.get_dummies(gdf_merged['predicted_disaster_type'], prefix='disaster')

X_reg = pd.concat([
    gdf_merged[['latitude', 'longitude', 
                'location_Zone A', 'location_Zone B', 'location_Zone C', 'location_Zone D', 
                'economic_loss_million_usd', 'reading_value']],
    disaster_dummies
], axis=1)
y_reg = gdf_merged['energy_kwh']

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)
rf_reg = RandomForestRegressor(n_estimators=200, random_state=42)
rf_reg.fit(X_train_reg, y_train_reg)

y_pred_reg = rf_reg.predict(X_test_reg)
mse_energy = mean_squared_error(y_test_reg, y_pred_reg)
print(f"Energy Regression MSE: {mse_energy:.2f}")

###################################################
# PART 3: CASUALTIES REGRESSION MODEL
###################################################
y_casualties = gdf_merged['casualties']

X_train_cas, X_test_cas, y_train_cas, y_test_cas = train_test_split(
    X_reg, y_casualties, test_size=0.2, random_state=42
)
rf_casualties = RandomForestRegressor(n_estimators=200, random_state=42)
rf_casualties.fit(X_train_cas, y_train_cas)

y_pred_cas = rf_casualties.predict(X_test_cas)
mse_cas = mean_squared_error(y_test_cas, y_pred_cas)
print(f"Casualties Regression MSE: {mse_cas:.2f}")

###################################################
# PART 4: FAKE NEWS DETECTION MODEL (Calibrated)
###################################################
# Dummy dataset for fake news detection.
data_fake = {
    'text': [
        # Real disaster-related tweets (label 0)
        "Breaking: A 6.5 magnitude earthquake has struck the region, causing widespread damage.",
        "Flood warnings issued for several counties after heavy rainfall.",
        "Wildfire under control; firefighters report minimal casualties.",
        "Hurricane making landfall; evacuation orders are in effect.",
        "Major industrial accident reported, with emergency services on site.",
        "Tornado hit the city outskirts, no severe injuries reported.",
        "Severe thunderstorm warning in effect; expect heavy winds and rains.",
        "Local authorities confirm no further aftershocks following the earthquake.",
        "Authorities report that the wildfire is contained and residents are safe.",
        "Flood levels are receding, and emergency crews are working to restore power.",
        "Shelters open citywide as storm approaches.",
        "Earthquake early-warning system triggered successfully.",
        "Emergency crews deployed to assist hurricane victims.",
        "Coast Guard rescues 12 from flooded neighborhood.",
        "Residents advised to boil water after pipe burst during quake.",
        "Wildfire containment reaches 90 percent after five days of effort.",
        "Tornado sirens activated just before touchdown.",
        "Storm surge pushes seawater half a mile inland.",
        "No damage reported after minor aftershock overnight.",
        "Disaster relief fund activated by city council.",
        "Hazmat team clears area after factory explosion.",
        "Air traffic delayed due to strong winds over disaster zone.",
        "Meteorologists confirm highest rainfall in decades.",
        "Downed power lines cause blackouts in three districts.",
        "All schools closed due to extreme heat alert.",
        "Volunteers organize community clean-up after flooding.",
        "Structural engineers assess collapsed bridge.",
        "National Guard assists with storm evacuation.",
        "Evacuation order lifted after fire threat recedes.",
        "Local hospitals prepare for surge in storm-related injuries.",
        "Disaster alert sent to residents via SMS and radio.",
        "Emergency supplies distributed at stadium shelter.",
        "Roads reopen after mudslide cleanup.",
        "Search and rescue teams locate missing hikers post-earthquake.",
        "Fire chief confirms blaze was electrical in origin.",
        "Weather center tracks second wave of the storm.",
        "Dam spillway opened to prevent overflow.",
        "Officials reassure residents after false alarm alert.",
        "Community fundraiser supports displaced families.",
        
        # Fake disaster-related tweets (label 1)
        "Exclusive: Scientists reveal that last night's earthquake was staged for political gain.",
        "Rumor: The flood warnings are fake, designed to cover up a secret military exercise.",
        "Breaking: Hurricane results have been exaggerated by media; claims say it's a hoax.",
        "Alert: Wildfire reported in downtown, but sources confirm it never existed.",
        "Unverified: An industrial accident was actually a controlled demolition misreported as disaster.",
        "Exclusive: Tornado sightings are fake, manipulated satellite images show no storm.",
        "Viral: Fake reports claim a major earthquake was caused by a secret weather weapon.",
        "Alert: Officials deny heavy rains occurred; flood warnings are a complete fabrication.",
        "Rumor: Claims of mass destruction after the hurricane are entirely false and meant to incite panic.",
        "Check this out: Reports of wildfire destruction are being debunked as a deliberate misinformation campaign.",
        "Flood water found to contain chemicals that make people compliant — experts silenced.",
        "Video shows UFO causing landslide — government in denial.",
        "Leak: Disaster sirens are actually sound weapons targeting brains.",
        "FAKE: Footage of the 'tornado' reused from a movie set.",
        "Hurricane drill used to sneak new 5G towers into schools.",
        "BREAKING: Emergency broadcasts trigger mass hypnosis.",
        "FACT CHECK: Earthquake was caused by underground missile test.",
        "Whistleblower: Disaster zones used to harvest organs secretly.",
        "City-wide blackout was staged to allow deep-state arrests.",
        "EXPOSED: No real storm, all images generated by AI.",
        "Tornado warnings fabricated to manipulate insurance markets.",
        "Wildfire never happened — no satellite evidence found.",
        "Local animals fleeing area suggest something unnatural, not weather.",
        "False flag: Explosion blamed on gas leak actually a drone strike.",
        "Fire department vehicles seen driving in circles to 'fake' response.",
        "Military operations disguised as earthquake relief.",
        "Experts silenced after revealing disaster drills as pretexts.",
        "Smoke machines create illusion of ongoing wildfire.",
        "Crisis actors spotted giving interviews about non-existent flood.",
        "Storm photos proven to be years old — used again by media.",
        "No hurricane damage in neighborhoods despite official reports.",
        "Conspiracy: Flood victims paid to exaggerate losses.",
        "Meteorologist caught faking live disaster report with green screen.",
        "Helicopter footage of earthquake is entirely CGI.",
        "Quake epicenter linked to secret underground prison break.",
        "Storm warnings reused from last year to manipulate elections.",
        "Wildfire zones cordoned off to hide government experiments.",
        "Multiple disaster alerts sent by mistake — or were they?",
        "City map altered to exaggerate fire spread for FEMA funding."
    ],
    'fake': [0] * 39 + [1] * 39  # 40 real, 40 fake
}
print(len(data_fake['text']), len(data_fake['fake']))  # Should both print 80

df_fake = pd.DataFrame(data_fake)


# Build a pipeline: TF-IDF vectorizer + Logistic Regression.
fake_news_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LogisticRegression(random_state=42))
])
# Optionally calibrate to get better probability estimates.
from sklearn.calibration import CalibratedClassifierCV
calibrated_fake_news_model = CalibratedClassifierCV(estimator=fake_news_pipeline, cv=3)
calibrated_fake_news_model.fit(df_fake['text'], df_fake['fake'])

def is_fake_news(tweet):
    """
    Determines if a tweet is fake news using our calibrated model.
    
    Parameters:
        tweet (str): The tweet text.
        
    Returns:
        tuple: (is_fake (bool), probability_fake (float))
    """
    prob_fake = calibrated_fake_news_model.predict_proba([tweet])[0][1]
    prediction = calibrated_fake_news_model.predict([tweet])[0]
    return bool(prediction), prob_fake


###################################################
# PART 5: LLM-BASED MESSAGE ENHANCEMENT USING GPT-2
###################################################
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_final_message(prompt, max_length=50):
    """
    Enhance emergency instructions by first providing context on what makes them effective.
    This version is inspired by 'generate_emergency_instructions', instructing the model to add
    additional context, detailed guidance, and a clear structure.
    """
    # Prepend additional context to guide GPT-2
    context = ("Please enhance the following emergency instructions. "
               "Provide clear, step-by-step guidance and make the message concise but detailed. "
               "Ensure it remains actionable and easy to understand.\n\n")
    full_prompt = context + prompt

    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")
    outputs = gpt2_model.generate(
        input_ids,
        max_length=max_length,
        do_sample=True,         # Allow sampling for variation.
        top_k=40,               # Lower top_k value.
        top_p=0.9,              # Slightly reduce top_p.
        temperature=0.8,        # Lower temperature for less creativity.
        num_return_sequences=1,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id
    )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text



###################################################
# PART 6: HELPER FUNCTIONS
###################################################
def detect_power_outage(energy_kwh, threshold=DYNAMIC_POWER_THRESHOLD):
    """
    Determines if the predicted energy usage indicates a power outage.
    
    Uses a dynamic threshold (default: 10th percentile of historical energy usage).
    """
    return energy_kwh < threshold

def generate_emergency_instructions(disaster, power_outage, casualties, tweet_info):
    """
    Generates base emergency instructions with detailed explanations.
    
    Parameters:
      - disaster (str): Predicted disaster type.
      - power_outage (bool): True if a power outage is indicated.
      - casualties (float): Predicted casualty count.
      - tweet_info (tuple): (is_fake (bool), probability (float))
      
    Returns:
      str: Base emergency instructions.
    """
    is_fake, prob_fake = tweet_info
    instructions = f"Disaster Detected: {disaster}.\n"
    instructions += ("Our models, which incorporate geospatial, economic, and historical data, "
                     "indicate that this disaster is occurring.\n")
    
    if power_outage:
        instructions += ("The predicted energy usage is critically low (below our dynamic threshold), "
                         "suggesting a power outage affecting the area.\n")
    else:
        instructions += "Energy usage predictions indicate that the power infrastructure is stable.\n"
    
    if casualties > 100:
        instructions += ("High casualty estimates signal severe impact; immediate evacuation and emergency response are essential.\n")
    elif casualties > 50:
        instructions += ("Moderate casualty predictions advise caution and preparing for potential evacuation.\n")
    else:
        instructions += "Casualty estimates are low, but remaining vigilant is crucial.\n"
    
    if is_fake:
        instructions += (f"Attention: Local tweet analysis suggests possible misinformation (fake probability: {prob_fake:.2f}). "
                         "Verify details with multiple sources before acting.\n")
    else:
        instructions += (f"Local tweet credibility appears high (fake probability: {prob_fake:.2f}), "
                         "supporting our model outputs.\n")
    
    instructions += "Follow instructions from local emergency services and remain updated through official channels."
    return instructions

###################################################
# PART 7: CASCADING PREDICTION FUNCTION
###################################################
def cascade_prediction(lat, lon, economic_loss=0, reading_value=0, zone=None, tweet_text=""):
    # Build the zone one-hot vector.
    zone_cols = ['location_Zone A', 'location_Zone B', 'location_Zone C', 'location_Zone D']
    zone_data = [0, 0, 0, 0]
    if zone in zone_cols:
        zone_data[zone_cols.index(zone)] = 1

    # Prepare a DataFrame for the classification input
    cls_columns = ['latitude', 'longitude', 'location_Zone A', 'location_Zone B', 
                   'location_Zone C', 'location_Zone D', 'economic_loss_million_usd', 'reading_value']
    cls_input = pd.DataFrame([[lat, lon, *zone_data, economic_loss, reading_value]], columns=cls_columns)
    
    # --- Stage 1: Disaster Prediction ---
    predicted_disaster = rf_cls.predict(cls_input)[0]
    print("\n=== Cascading Prediction ===")
    print("Predicted Disaster Type:", predicted_disaster)
    
    # One-hot encode disaster for regression input.
    all_disaster_types = list(rf_cls.classes_)
    disaster_onehot = [1 if dt == predicted_disaster else 0 for dt in all_disaster_types]
    
    # Prepare the regression input. Note that your regression input was built with the same
    # original features plus the one-hot encoded disaster type.
    reg_columns = cls_columns + [f'disaster_{dt}' for dt in all_disaster_types]
    reg_input = pd.DataFrame([[lat, lon, *zone_data, economic_loss, reading_value, *disaster_onehot]], columns=reg_columns)
    
    # --- Stage 2: Energy & Casualties Prediction ---
    predicted_energy = rf_reg.predict(reg_input)[0]
    predicted_casualties = rf_casualties.predict(reg_input)[0]
    print(f"Predicted Energy Usage (kWh): {predicted_energy:.2f}")
    print(f"Predicted Casualties: {predicted_casualties:.2f}")
    
    # --- Stage 3: Power Outage Detection ---
    power_outage = detect_power_outage(predicted_energy)
    
    # --- Stage 4: Fake News Detection ---
    if tweet_text:
        fake_flag, fake_prob = is_fake_news(tweet_text)
    else:
        fake_flag, fake_prob = (False, 0.0)
    print(f"Tweet Fake News Flag: {fake_flag}, Fake Probability: {fake_prob:.2f}")
    
    # --- Stage 5: Generate Base Emergency Instructions ---
    base_instructions = generate_emergency_instructions(
        disaster=predicted_disaster,
        power_outage=power_outage,
        casualties=predicted_casualties,
        tweet_info=(fake_flag, fake_prob)
    )
    
    # --- Stage 6: Enhance the Message via GPT-2 ---
    prompt = (f"Improve the following emergency instructions with additional context and detailed, "
              f"step-by-step guidance:\n\n{base_instructions}\n\nFinal Message:")
    enhanced_instructions = generate_final_message(prompt, max_length=250)
    
    print("\n=== Emergency Instructions ===")
    print(enhanced_instructions)


###################################################
# PART 8: EXAMPLE USAGE
###################################################
cascade_prediction(
    lat=37.7749,
    lon=-122.4194,
    economic_loss=400,
    reading_value=90,
    zone='location_Zone C',   # Ensure this matches one of your zone columns.
    tweet_text="For everyone impacted by the flood, please evacuate now"
)
