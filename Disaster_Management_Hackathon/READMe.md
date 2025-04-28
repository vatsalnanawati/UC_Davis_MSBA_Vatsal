# 🌪️ Cascade Crisis AI - Hackathon Project

## 🚀 Project Overview
**Cascade Crisis AI** is designed to be the thinking layer when urban disasters strike — anticipating cascading failures, filtering misinformation, and suggesting rapid, explainable decisions.  
Our mission: **Predict the next collapse before it happens**.

---

## 🛠️ What We Built
**Cascading Disaster Prediction**  
→ Detects disaster chains in motion (e.g., flood → power outage → hospital overload).

**Fake News Filtering**  
→ Uses NLP to verify social media updates in real-time and catch misinformation before panic spreads.

**Hotzone Mapping**  
→ Clusters nearby incidents using **geospatial DBSCAN**, revealing emerging high-risk zones.

**Smart Emergency Routing**  
→ Suggests closest safe locations (hospital, fire station, shelter) with alerts if too far (>4 km).

---

## 🔥 Key Components
- **Random Forest Classifier** — Disaster Type Prediction  
- **Random Forest Regressor** — Predict Energy Consumption and Casualties  
- **TF-IDF + Calibrated Logistic Regression** — Fake News Detection on tweets  
- **GPT-2 (HuggingFace Transformers)** — LLM-enhanced emergency message generation  
- **GeoPandas + Shapely** — Spatial joining for disaster localization  
- **Streamlit Dashboard** — (Optional) for real-time crisis monitoring interface  

---

## 📊 Datasets Used
Extracted from smart city systems:
- Sensor readings (temperature, humidity, seismic activity)
- Energy consumption logs
- Historical disaster event data
- Simulated Twitter data for real-time updates  

We **merged multimodal data** (geospatial + textual + time series) to power our predictive models.

---

## 🧠 How Our AI Thinks
1. **Analyze** sensor chaos and social noise.  
2. **Predict** disaster type at the given location.  
3. **Estimate** energy disruptions and casualty levels.  
4. **Detect** possible misinformation from social media.  
5. **Generate** clear, improved emergency instructions using LLM.

**Tools:** Python, Scikit-Learn, GeoPandas, Shapely, DBSCAN, TF-IDF, Logistic Regression, Folium, Streamlit

## 🚀 Future Enhancements

- Integrate **live weather, traffic, and satellite APIs** for real-time predictions and context enrichment.
- Deploy to **edge computing devices** for offline-first response in disaster-prone zones.
- Upgrade misinformation detection with **BERT-style transformer models** for deeper tweet context understanding.
- Extend multilingual AI output for SMS/email/alert compatibility across regions.

📍 **Check out the full notebook, mapping engine, and Streamlit app to explore the crisis simulation in action!**
---

## 🧪 How to Run It
```bash
# Step 1: Clone Repo
git clone https://github.com/your-repo/cascade-crisis-ai.git

# Step 2: Install Requirements
pip install -r requirements.txt

# Step 3: Run Core Script
python cascading_script.py

# (Optional) Step 4: Launch Streamlit Dashboard
streamlit run dashboard.py

