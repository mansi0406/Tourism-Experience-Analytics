import os
import json
import streamlit as st
import pandas as pd
import joblib
from pipeline import run_pipeline, load_artifacts, build_single_feature_row, vectorize_row, recommend_for_user, _find_col

base_dir = os.path.dirname(__file__)
metrics_path = os.path.join(base_dir, "outputs", "metrics.json")
if not os.path.exists(metrics_path):
    run_pipeline(base_dir)
data, cols, reg, clf, sim = load_artifacts(base_dir)
if data is None or cols is None:
    run_pipeline(base_dir)
    data, cols, reg, clf, sim = load_artifacts(base_dir)
if sim is None and data is not None:
    from pipeline import build_user_item_matrix, build_item_similarity
    _mat = build_user_item_matrix(data)
    if _mat is not None:
        sim = build_item_similarity(_mat)
if sim is not None:
    sim.index = sim.index.astype(str)
    sim.columns = sim.columns.astype(str)
metrics = {}
if os.path.exists(metrics_path):
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
st.set_page_config(page_title="Tourism Experience Analytics", layout="wide")
st.markdown("""
<style>
.main {background-color:#0b22ff;}
div.stButton>button {background-color:#ff5c5c;color:white;border-radius:8px;height:48px;font-weight:600;}
.metric-container {background:#0b22ff;color:white;padding:8px;border-radius:8px;}
.card {background:#111827;border:1px solid #1f2937;border-radius:8px;padding:12px;margin-bottom:8px;color:#e5e7eb;}
.section {background:#0b22ff;padding:16px;border-radius:12px;color:white;}
</style>
""", unsafe_allow_html=True)
st.title("Tourism Experience Analytics")
col1, col2, col3 = st.columns(3)
if "regression" in metrics:
    col1.metric("R²", f'{metrics["regression"].get("r2", 0):.3f}')
    col2.metric("RMSE", f'{metrics["regression"].get("rmse", 0):.3f}')
if "classification" in metrics:
    col3.metric("Accuracy", f'{metrics["classification"].get("accuracy", 0):.3f}')
user_col = _find_col(data, ["user_id", "userid", "user"])
item_col = _find_col(data, ["item_id", "itemid", "attraction_id", "attractionid", "item"])
st.sidebar.header("About")
st.sidebar.success("Discover personalized attractions using your history and similarities.")
st.sidebar.info("Choose a user, click Find Attractions, explore Top Picks.")
user_ids = sorted(list(set(data[user_col].tolist()))) if user_col else []
item_ids = sorted(list(set(data[item_col].tolist()))) if item_col else []
selected_user = st.sidebar.selectbox("User", user_ids) if user_ids else None
selected_item = st.sidebar.selectbox("Attraction", item_ids) if item_ids else None
top_k = st.sidebar.slider("Top K recommendations", min_value=3, max_value=10, value=5)
tab1, tab2, tab3 = st.tabs(["Predict Rating", "Visit Mode Prediction", "Recommendations"])
with tab1:
    if st.button("Predict Rating"):
        if selected_user is not None and selected_item is not None and reg is not None:
            row = build_single_feature_row(data, selected_user, selected_item)
            if row is not None and cols is not None:
                Xrow = vectorize_row(row, cols)
                pred = reg.predict(Xrow)[0]
                st.success(f"Predicted rating: {pred:.2f}")
with tab2:
    conts = [c for c in data.columns if "continent" in str(c).lower()]
    regs = [c for c in data.columns if "region" in str(c).lower()]
    years_available = data["year"].dropna().unique().tolist() if "year" in data.columns else []
    months_available = data["month"].dropna().unique().tolist() if "month" in data.columns else []
    c1, c2 = st.columns(2)
    uid = c1.number_input("User ID", value=int(user_ids[0]) if user_ids else 1, step=1)
    yv = c1.selectbox("Visit Year", years_available if years_available else [2000])
    mv = c1.selectbox("Visit Month", months_available if months_available else [1])
    aid = c2.number_input("Attraction ID", value=int(item_ids[0]) if item_ids else 1, step=1)
    cont_val = c2.selectbox("Continent", sorted(data[conts[0]].dropna().unique().tolist())) if conts else None
    reg_val = c2.selectbox("Region", sorted(data[regs[0]].dropna().unique().tolist())) if regs else None
    if st.button("Predict Visit Mode"):
        if clf is not None:
            row = build_single_feature_row(data, uid, aid)
            if row is not None and cols is not None:
                if "year" in row.columns:
                    row["year"] = yv
                if "month" in row.columns:
                    row["month"] = mv
                if conts:
                    row[conts[0]] = cont_val
                if regs:
                    row[regs[0]] = reg_val
                Xrow = vectorize_row(row, cols)
                pred = clf.predict(Xrow)[0]
                st.success(f"Predicted visit mode: {pred}")
    st.subheader("Preview of Tourism Data")
    st.dataframe(data.head(20))
with tab3:
    st.subheader("Select User ID for Recommendations")
    sel_user = st.selectbox("Choose a User ID", user_ids) if user_ids else None
    if st.button("Get Recommendations"):
        pop_col = _find_col(data, ["attraction_popularity"])
        if sim is None and data is not None:
            from pipeline import build_user_item_matrix, build_item_similarity
            _mat = build_user_item_matrix(data)
            if _mat is not None:
                sim = build_item_similarity(_mat)
                sim.index = sim.index.astype(str)
                sim.columns = sim.columns.astype(str)
        if sim is not None:
            uid = sel_user
            if uid is None:
                uid = data[user_col].iloc[0] if user_col else None
            recs = recommend_for_user(data, sim, uid, top_k=top_k) if uid is not None else []
            if recs:
                st.subheader("Recommended Attractions")
                for r in recs:
                    st.markdown(f"<div class='card'>✨ {r}</div>", unsafe_allow_html=True)
                st.subheader("Preview of Tourism Data")
                st.dataframe(data.head(20))
            elif pop_col and item_col:
                st.subheader("Popular Attractions")
                popular = data.groupby(item_col)[pop_col].max().sort_values(ascending=False)
                for r in list(popular.index[:top_k]):
                    st.markdown(f"<div class='card'>✨ {r}</div>", unsafe_allow_html=True)
                st.subheader("Preview of Tourism Data")
                st.dataframe(data.head(20))
            else:
                if item_col:
                    st.subheader("Popular Attractions")
                    popular = data[item_col].value_counts()
                    for r in list(popular.index[:top_k]):
                        st.markdown(f"<div class='card'>✨ {r}</div>", unsafe_allow_html=True)
                    st.subheader("Preview of Tourism Data")
                    st.dataframe(data.head(20))
                else:
                    st.info("Attraction column not detected; please run pipeline to rebuild data.")
        else:
            if item_col:
                st.subheader("Popular Attractions")
                popular = data[item_col].value_counts()
                for r in list(popular.index[:top_k]):
                    st.markdown(f"<div class='card'>✨ {r}</div>", unsafe_allow_html=True)
                st.subheader("Preview of Tourism Data")
                st.dataframe(data.head(20))
            else:
                st.info("Attraction column not detected; please run pipeline to rebuild data.")
