# 🚌 SFMTA Bus Citation AI - Smarter Transit Through Data

---

## 🎯 Goal: Decode Citation Patterns, Empower City Ops

This project leverages real-time and historical parking citation data from SFMTA to analyze spatial and temporal patterns of violations that affect public bus movement. The goal is to provide operational intelligence for smarter officer deployment and reduce transit inefficiencies caused by parking infractions.

**Mission:** Decode patterns in urban traffic enforcement to inform smarter transit operations.

---

## 🔍 What We Developed

- **📅 Time Feature Engineering**  
  → Extracted hour, weekday, weekend, and seasonal trends from citation timestamps.

- **📍 Hotspot Detection via Clustering**  
  → Applied `DBSCAN` on lat/lon data to discover high-density citation zones.

- **🗺️ Spatial Join by Neighborhood**  
  → Merged citation data with SF neighborhood shapefiles for area-wise violation mapping.

- **🧼 Cleaned & Encoded Categorical Features**  
  → One-hot encoded `violation_desc` and `analysis_neighborhood` to prep for ML.

- **🧠 Citation Risk Predictor (Gradio Interface)**  
  → Accepts location (`latitude`, `longitude`) and time (`hour`, `weekday`) as inputs.  
  → Returns likelihood of receiving a citation at that time and place.  
  → Built using `Gradio` for simple, no-code interaction with the model.

---

## 🧩 Core Building Blocks

- **Socrata API Ingestion** — Retrieved 500,000+ citations using paginated API requests.
- **Datetime Parsing** — Engineered weekday, hour, and weekend indicators from `date_added`.
- **One-Hot Encoding** — Categorical features like violations and neighborhoods transformed for ML readiness.
- **DBSCAN Clustering** — Identified geographic hotspots using geospatial density.
- *(Planned)*: Supervised models (XGBoost, RF) for citation location prediction.
- *(Planned)*: Streamlit dashboard for interactive spatial/temporal exploration.

---

## 📂 Data Sources at a Glance

- **SFMTA Parking Citations**  
  API Endpoint: `https://data.sfgov.org/resource/8pxu-u28x.json`  
  Fields include:
  - `violation_desc`, `latitude`, `longitude`, `fine_amount`, `analysis_neighborhood`, `date_added`

- *(Planned)*: Bus route & delay data to study transit impact

---

## 🧠 Our AI Workflow

1. **Extract** temporal features like hour, weekday, and seasonality  
2. **Cluster** dense violation locations using DBSCAN  
3. **Encode** categorical data for analysis and ML training  
4. *(Upcoming)*: **Train** classifiers for citation hotspot prediction  
5. *(Upcoming)*: **Visualize** violations by location & time in a dynamic dashboard

**Tech Stack**: Python, Pandas, GeoPandas, Scikit-learn, DBSCAN, Shapely, Seaborn, Streamlit, Gradio

---

## 🚀 What’s Next: Expanding the Impact

- Train classifiers (e.g., XGBoost, RF) to predict future citation-prone areas  
- Integrate live bus delay data for correlation with citation clusters  
- Build an interactive **Streamlit dashboard** for real-time monitoring  
- Fuse external data: event schedules, road closures, and weather conditions

---

## 🌐 Try the Interactive Predictor

Launch the real-time citation prediction tool powered by Gradio:

[![Open in Gradio](https://img.shields.io/badge/Gradio-Live--Demo-blue?style=for-the-badge&logo=gradio)](https://5aae85e178ecba54db.gradio.live)

**Inputs:**
- `Latitude`, `Longitude` (geo-coordinates of location)
- `Hour of Day` (0–23)
- `Weekday` (0 = Monday, 6 = Sunday)

**Output:**  
A real-time prediction of citation risk at that location and time.

> *Hosted via public URL from Colab – expires in 1 week. For long-term access, deploy via `gradio deploy` to HuggingFace Spaces.*

---

## 🖥️ Try It Yourself: Run Instructions

```bash
# Step 1: Clone Repo
git clone https://github.com/your-repo/sfmta-bus-citation.git

# Step 2: Install Dependencies
pip install -r requirements.txt

# Step 3: Launch Jupyter Notebook
jupyter notebook SFMTA_Parking_Citation_Project.ipynb

# (Optional) Step 4: Run Dashboard
streamlit run dashboard.py
