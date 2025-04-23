# 🌪️ Cascade Crisis AI: Predicting Disasters, Filtering Fakes, Guiding Response

**Tools:** Python, Scikit-Learn, GeoPandas, Shapely, DBSCAN, TF-IDF, Logistic Regression, Folium, Streamlit

This project builds a real-time AI platform that predicts cascading disasters, filters misinformation from tweets, and generates infrastructure response plans. Using smart city sensor data, social inputs, and spatial analytics, the system enables responders to act faster with confidence and clarity.

## 🔍 Key Highlights

- Conducted **geospatial clustering** using **DBSCAN** to detect escalating disaster zones within proximity thresholds.
- Mapped incidents and infrastructure using **GeoPandas** and **Shapely**, and calculated response reach distances to hospitals, shelters, and fire stations.
- Used **TF-IDF with Calibrated Logistic Regression** to classify real vs. fake tweets, achieving **80%+ accuracy** in live simulations.
- Trained **Random Forest classifiers and regressors** to predict disaster type, casualties, and power disruption based on sensor inputs.
- Deployed a user-friendly **Streamlit dashboard** for real-time scenario simulation, risk visualization, and AI-generated instructions.
- Visualized risk clusters and infrastructure recommendations using **Folium** with color-coded markers, severity scaling, and escalation flags.

## 🚀 Future Enhancements

- Integrate **live weather, traffic, and satellite APIs** for real-time predictions and context enrichment.
- Deploy to **edge computing devices** for offline-first response in disaster-prone zones.
- Upgrade misinformation detection with **BERT-style transformer models** for deeper tweet context understanding.
- Extend multilingual AI output for SMS/email/alert compatibility across regions.

📍 **Check out the full notebook, mapping engine, and Streamlit app to explore the crisis simulation in action!**
