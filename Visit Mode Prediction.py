import os
import json
import streamlit as st
import pandas as pd
from pipeline import run_pipeline, load_artifacts, build_single_feature_row, vectorize_row, _find_col

base_dir = os.path.dirname(os.path.dirname(__file__))
metrics_path = os.path.join(base_dir, "outputs", "metrics.json")
if not os.path.exists(metrics_path):
    run_pipeline(base_dir)
data, cols, reg, clf, sim = load_artifacts(base_dir)
if data is None or cols is None:
    run_pipeline(base_dir)
    data, cols, reg, clf, sim = load_artifacts(base_dir)

st.set_page_config(page_title="Visit Mode Prediction", layout="wide")
st.markdown("""
<style>
.main {background-color:#0b22ff;}
div.stButton>button {background-color:#ff5c5c;color:white;border-radius:8px;height:48px;font-weight:600;}
.panel {background:#111827;border:1px solid #1f2937;border-radius:12px;padding:16px;color:#e5e7eb;}
.section {background:#0b22ff;padding:16px;border-radius:12px;color:white;}
.title {background:#0a165e;color:white;padding:12px 16px;border-radius:12px;font-size:28px;font-weight:800;}
.input-card {background:#0a165e22;border-radius:8px;padding:8px;}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div class='panel'><h3>About This Application</h3><ul><li>Visit Mode Prediction System</li><li>Enter user and attraction context</li><li>Predict most likely visit mode</li><li>Powered by Streamlit</li></ul></div>", unsafe_allow_html=True)

st.markdown("<div class='title'>🚀 VISIT MODE PREDICTION</div>", unsafe_allow_html=True)

user_col = _find_col(data, ["user_id", "userid", "user"])
item_col = _find_col(data, ["attractionid", "item_id", "itemid", "attraction_id", "item"])
continent_col = _find_col(data, ["continent_id", "continentid"])
region_col = _find_col(data, ["region_id", "regionid"])
type_col = _find_col(data, ["attraction_type", "type_name", "type"])
type_id_col = _find_col(data, ["attractiontypeid", "type_id", "typeid"])
name_col = _find_col(data, ["attraction_name", "item_name", "name"])

user_ids = sorted(list(set(data[user_col].tolist()))) if user_col else []
item_ids = sorted(list(set(data[item_col].tolist()))) if item_col else []
years_available = data["year"].dropna().unique().tolist() if "year" in data.columns else [2000]
months_available = data["month"].dropna().unique().tolist() if "month" in data.columns else [1]
continents = sorted(data[continent_col].dropna().unique().tolist()) if continent_col else []
regions = sorted(data[region_col].dropna().unique().tolist()) if region_col else []
types = sorted(data[type_col].dropna().unique().tolist()) if type_col else []

c1, c2 = st.columns(2)
uid = c1.number_input("User ID", value=int(user_ids[0]) if user_ids else 1, step=1)
cont = c2.selectbox("Continent ID", continents if continents else [1])
yr = c1.selectbox("Visit Year", years_available)
regv = c2.selectbox("Region ID", regions if regions else [1])
mn = c1.selectbox("Visit Month", months_available)
aname = c2.text_input("Attraction Name", "")
vm_enc = c1.number_input("Visit Mode (Encoded)", value=1, step=1)
atype = c2.text_input("Attraction Type", types[0] if types else "")
aid = c1.number_input("Attraction ID", value=int(item_ids[0]) if item_ids else 1, step=1)
atype_id = c2.number_input("Attraction Type ID", value=1, step=1)

if st.button("Predict Visit Mode"):
    if clf is None:
        st.warning("Classifier not available; please run pipeline.")
    else:
        row = build_single_feature_row(data, uid, aid)
        if row is None:
            st.warning("Could not build feature row.")
        else:
            if "year" in row.columns:
                row["year"] = yr
            if "month" in row.columns:
                row["month"] = mn
            if continent_col:
                row[continent_col] = cont
            if region_col:
                row[region_col] = regv
            if type_id_col:
                row[type_id_col] = atype_id
            if type_col:
                row[type_col] = atype
            if name_col:
                row[name_col] = aname
            Xrow = vectorize_row(row, cols)
            pred = clf.predict(Xrow)[0]
            st.success(f"Predicted visit mode: {pred}")

st.subheader("Preview of Tourism Data")
st.dataframe(data.head(20)) 
