import os
import json
import math
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def _norm(s):
    return str(s).strip().lower().replace(" ", "_")

def _find_col(df, candidates):
    cols = {_norm(c): c for c in df.columns}
    for cand in candidates:
        k = _norm(cand)
        if k in cols:
            return cols[k]
    for c in df.columns:
        nc = _norm(c)
        for cand in candidates:
            if _norm(cand) in nc:
                return c
    return None

def load_tables(base_dir):
    def xl(path):
        return pd.read_excel(os.path.join(base_dir, path))
    tables = {}
    tables["transaction"] = xl("Transaction.xlsx")
    tables["user"] = xl("User.xlsx")
    tables["item"] = xl("Item.xlsx")
    tables["type"] = xl("Type.xlsx")
    tables["mode"] = xl("Mode.xlsx")
    tables["city"] = xl("City.xlsx")
    tables["country"] = xl("Country.xlsx")
    tables["region"] = xl("Region.xlsx")
    tables["continent"] = xl("Continent.xlsx")
    add_path = os.path.join(base_dir, "Additional_Data_for_Attraction_Sites", "Updated_Item.xlsx")
    if os.path.exists(add_path):
        tables["item_additional"] = pd.read_excel(add_path)
    return tables

def merge_tables(tables):
    tx = tables["transaction"].copy()
    usr = tables["user"].copy()
    itm = tables["item"].copy()
    typ = tables["type"].copy()
    md = tables["mode"].copy()
    city = tables["city"].copy()
    country = tables["country"].copy()
    region = tables["region"].copy()
    continent = tables["continent"].copy()
    if "item_additional" in tables:
        ia = tables["item_additional"].copy()
        item_id_col = _find_col(itm, ["item_id", "itemid", "attraction_id", "id"])
        ia_id_col = _find_col(ia, ["item_id", "itemid", "attraction_id", "id"])
        if item_id_col and ia_id_col:
            itm = itm.merge(ia, left_on=item_id_col, right_on=ia_id_col, how="left")
    user_id_tx = _find_col(tx, ["user_id", "userid", "user id", "user"])
    user_id_usr = _find_col(usr, ["user_id", "userid", "id"])
    item_id_tx = _find_col(tx, ["item_id", "itemid", "attraction_id", "attractionid", "item"])
    item_id_itm = _find_col(itm, ["item_id", "itemid", "attraction_id", "attractionid", "id"])
    type_id_itm = _find_col(itm, ["type_id", "typeid"])
    type_id_typ = _find_col(typ, ["type_id", "typeid", "id"])
    mode_id_tx = _find_col(tx, ["visit_mode_id", "mode_id", "modeid"])
    mode_id_md = _find_col(md, ["visit_mode_id", "mode_id", "modeid", "id"])
    city_id_itm = _find_col(itm, ["city_id"])
    city_id_city = _find_col(city, ["city_id", "id"])
    country_id_city = _find_col(city, ["country_id"])
    country_id_country = _find_col(country, ["country_id", "id"])
    region_id_country = _find_col(country, ["region_id"])
    region_id_region = _find_col(region, ["region_id", "id"])
    continent_id_region = _find_col(region, ["continent_id"])
    continent_id_continent = _find_col(continent, ["continent_id", "id"])
    if user_id_tx and user_id_usr:
        tx = tx.merge(usr, left_on=user_id_tx, right_on=user_id_usr, how="left", suffixes=("", "_user"))
    if item_id_tx and item_id_itm:
        tx = tx.merge(itm, left_on=item_id_tx, right_on=item_id_itm, how="left", suffixes=("", "_item"))
    if type_id_typ:
        type_id_in_tx = _find_col(tx, ["type_id", "typeid", "attractiontypeid"])
        if type_id_in_tx:
            tx = tx.merge(typ, left_on=type_id_in_tx, right_on=type_id_typ, how="left", suffixes=("", "_type"))
    if mode_id_tx and mode_id_md:
        tx = tx.merge(md, left_on=mode_id_tx, right_on=mode_id_md, how="left", suffixes=("", "_mode"))
    if city_id_city:
        city_id_in_tx = _find_col(tx, ["city_id"])
        if city_id_in_tx:
            tx = tx.merge(city, left_on=city_id_in_tx, right_on=city_id_city, how="left", suffixes=("", "_city"))
    if country_id_country:
        country_id_in_tx = _find_col(tx, ["country_id"])
        if country_id_in_tx:
            tx = tx.merge(country, left_on=country_id_in_tx, right_on=country_id_country, how="left", suffixes=("", "_country"))
    if region_id_region:
        region_id_in_tx = _find_col(tx, ["region_id"])
        if region_id_in_tx:
            tx = tx.merge(region, left_on=region_id_in_tx, right_on=region_id_region, how="left", suffixes=("", "_region"))
    if continent_id_continent:
        continent_id_in_tx = _find_col(tx, ["continent_id"])
        if continent_id_in_tx:
            tx = tx.merge(continent, left_on=continent_id_in_tx, right_on=continent_id_continent, how="left", suffixes=("", "_continent"))
    return tx

def clean_data(df):
    df = df.copy()
    if "rating" in [c.lower() for c in df.columns]:
        rating_col = [c for c in df.columns if c.lower() == "rating"][0]
    else:
        rating_col = _find_col(df, ["rating", "score"])
    if rating_col is not None:
        df[rating_col] = pd.to_numeric(df[rating_col], errors="coerce")
        df = df[df[rating_col].notna()]
        df[rating_col] = df[rating_col].clip(1, 5)
    year_col = _find_col(df, ["year"])
    month_col = _find_col(df, ["month"])
    date_col = _find_col(df, ["date", "visit_date"])
    if year_col and month_col:
        def mkdate(y, m):
            try:
                return datetime(int(y), int(m), 1)
            except:
                return pd.NaT
        df["date"] = [mkdate(a, b) for a, b in zip(df[year_col], df[month_col])]
    elif date_col:
        df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.drop_duplicates()
    df = df.replace({True: 1, False: 0})
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].astype(str).str.strip()
    return df

def season_from_month(m):
    if m in [12, 1, 2]:
        return "winter"
    if m in [3, 4, 5]:
        return "spring"
    if m in [6, 7, 8]:
        return "summer"
    if m in [9, 10, 11]:
        return "autumn"
    return "unknown"

def feature_engineering(df):
    df = df.copy()
    user_id = _find_col(df, ["user_id", "userid", "user"])
    item_id = _find_col(df, ["item_id", "itemid", "attraction_id", "attractionid", "item"])
    rating_col = _find_col(df, ["rating", "score"])
    mode_name_col = _find_col(df, ["visit_mode_name", "mode_name", "mode", "visit_mode"])
    type_name_col = _find_col(df, ["type_name", "type"])
    city_name_col = _find_col(df, ["city_name", "city"])
    country_name_col = _find_col(df, ["country_name", "country"])
    region_name_col = _find_col(df, ["region_name", "region"])
    continent_name_col = _find_col(df, ["continent_name", "continent"])
    if user_id and item_id:
        visits_per_user = df.groupby(user_id).size().rename("user_total_visits")
        df = df.merge(visits_per_user, left_on=user_id, right_index=True, how="left")
        if rating_col:
            avg_rating_user = df.groupby(user_id)[rating_col].mean().rename("user_avg_rating")
            df = df.merge(avg_rating_user, left_on=user_id, right_index=True, how="left")
        popularity = df.groupby(item_id).size().rename("attraction_popularity")
        df = df.merge(popularity, left_on=item_id, right_index=True, how="left")
    if "date" in df.columns:
        df["month"] = df["date"].dt.month
        df["season"] = df["month"].apply(season_from_month)
        df["year"] = df["date"].dt.year
    cols_keep = []
    for c in [user_id, item_id, rating_col, mode_name_col, type_name_col, city_name_col, country_name_col, region_name_col, continent_name_col, "season", "month", "year", "user_total_visits", "user_avg_rating", "attraction_popularity", "date"]:
        if c and c in df.columns:
            cols_keep.append(c)
    df = df[cols_keep].copy()
    return df

def eda(df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rating_col = _find_col(df, ["rating", "score"])
    mode_col = _find_col(df, ["visit_mode_name", "mode_name", "mode", "visit_mode"])
    item_id = _find_col(df, ["item_id", "itemid", "attraction_id", "attractionid", "item"])
    country_name_col = _find_col(df, ["country_name", "country"])
    if rating_col:
        plt.figure()
        sns.histplot(df[rating_col], kde=False, bins=10)
        plt.title("Rating distribution")
        plt.savefig(os.path.join(out_dir, "rating_distribution.png"), bbox_inches="tight")
        plt.close()
    if mode_col:
        plt.figure()
        df[mode_col].value_counts().head(20).plot(kind="bar")
        plt.title("Visit mode distribution")
        plt.savefig(os.path.join(out_dir, "visit_mode_distribution.png"), bbox_inches="tight")
        plt.close()
    if item_id:
        plt.figure()
        df[item_id].value_counts().head(20).plot(kind="bar")
        plt.title("Top attractions by visits")
        plt.savefig(os.path.join(out_dir, "top_attractions.png"), bbox_inches="tight")
        plt.close()
    if country_name_col:
        plt.figure()
        df[country_name_col].value_counts().head(20).plot(kind="bar")
        plt.title("Top countries by visits")
        plt.savefig(os.path.join(out_dir, "top_countries.png"), bbox_inches="tight")
        plt.close()
    if "month" in df.columns and rating_col:
        plt.figure()
        df.groupby("month")[rating_col].mean().plot(kind="line")
        plt.title("Seasonal rating trends")
        plt.savefig(os.path.join(out_dir, "seasonal_trends.png"), bbox_inches="tight")
        plt.close()

def prepare_datasets(df):
    y_reg_col = _find_col(df, ["rating", "score"])
    y_clf_col = _find_col(df, ["visit_mode_name", "mode_name", "mode", "visit_mode"])
    non_feature = set([y_reg_col, y_clf_col, "date"])
    feature_df = df.drop(columns=[c for c in non_feature if c in df.columns])
    cat_cols = feature_df.select_dtypes("object").columns.tolist()
    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_df[num_cols] = feature_df[num_cols].fillna(0)
    X = pd.get_dummies(feature_df, columns=cat_cols, drop_first=False)
    X = X.fillna(0)
    y_reg = None
    y_clf = None
    if y_reg_col:
        y_reg = df[y_reg_col].astype(float)
    if y_clf_col:
        y_clf = df[y_clf_col].astype(str)
    if y_reg is not None:
        mask = y_reg.notna()
        X = X.loc[mask]
        y_reg = y_reg.loc[mask]
    if y_clf is not None:
        mask = y_clf.notna()
        X = X.loc[mask]
        y_clf = y_clf.loc[mask]
    return X, y_reg, y_clf

def train_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = math.sqrt(mean_squared_error(y_test, preds))
    return model, {"r2": r2, "rmse": rmse}

def train_classification(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="weighted")
    return model, {"accuracy": acc, "f1": f1}

def build_user_item_matrix(df):
    user_id = _find_col(df, ["user_id", "userid", "user"])
    item_id = _find_col(df, ["item_id", "itemid", "attraction_id", "attractionid", "item"])
    rating_col = _find_col(df, ["rating", "score"])
    if not (user_id and item_id):
        return None
    if rating_col:
        mat = df.pivot_table(index=user_id, columns=item_id, values=rating_col, aggfunc="mean").fillna(0)
    else:
        tmp = df[[user_id, item_id]].dropna().copy()
        tmp["_interaction"] = 1.0
        mat = tmp.pivot_table(index=user_id, columns=item_id, values="_interaction", aggfunc="sum").fillna(0)
    return mat

def build_item_similarity(user_item):
    cols = list(map(str, user_item.columns.tolist()))
    user_item = user_item.copy()
    user_item.columns = cols
    sim = cosine_similarity(user_item.T)
    return pd.DataFrame(sim, index=cols, columns=cols)

def recommend_for_user(df, sim_df, user_id_value, top_k=5):
    user_id = _find_col(df, ["user_id", "userid", "user"])
    item_id = _find_col(df, ["item_id", "itemid", "attraction_id", "attractionid", "item"])
    rating_col = _find_col(df, ["rating", "score"])
    if not user_id or not item_id:
        pop_col = _find_col(df, ["attraction_popularity"])
        alt_item = None
        for c in df.columns:
            s = str(c).lower()
            if ("attraction" in s and "id" in s) or ("item" in s and "id" in s):
                alt_item = c
                break
        if pop_col and alt_item:
            popular = df.groupby(alt_item)[pop_col].max().sort_values(ascending=False)
            return [str(i) for i in popular.index[:top_k]]
        if alt_item:
            vc = df[alt_item].value_counts()
            return [str(i) for i in vc.index[:top_k]]
        return []
    df_items = df[df[user_id] == user_id_value]
    seen = set(map(str, df_items[item_id].dropna().tolist()))
    user_ratings = df_items[[item_id, rating_col]].dropna() if rating_col else pd.DataFrame()
    scores = {}
    def _accumulate_from_item(iid_str, weight):
        if iid_str in sim_df.index:
            sims = sim_df.loc[iid_str]
            for j, s in sims.items():
                if j in seen:
                    continue
                scores[j] = scores.get(j, 0) + s * float(weight)
    if not user_ratings.empty:
        for _, row in user_ratings.iterrows():
            _accumulate_from_item(str(row[item_id]), float(row[rating_col]))
    else:
        for iid in df_items[item_id].dropna().unique().tolist():
            _accumulate_from_item(str(iid), 1.0)
    recs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if not recs:
        pop_col = _find_col(df, ["attraction_popularity"])
        if pop_col:
            popular = df.groupby(item_id)[pop_col].max().sort_values(ascending=False)
            popular = [str(i) for i in popular.index if str(i) not in seen]
            return popular[:top_k]
    return [x[0] for x in recs[:top_k]]

def save_artifacts(base_dir, df, X_cols, reg_model, clf_model, metrics_reg, metrics_clf, sim_df):
    os.makedirs(os.path.join(base_dir, "outputs"), exist_ok=True)
    df.to_csv(os.path.join(base_dir, "outputs", "cleaned_master.csv"), index=False)
    with open(os.path.join(base_dir, "outputs", "feature_columns.json"), "w") as f:
        json.dump(X_cols, f)
    joblib.dump(reg_model, os.path.join(base_dir, "outputs", "regression_model.joblib"))
    joblib.dump(clf_model, os.path.join(base_dir, "outputs", "classification_model.joblib"))
    with open(os.path.join(base_dir, "outputs", "metrics.json"), "w") as f:
        json.dump({"regression": metrics_reg, "classification": metrics_clf}, f)
    if sim_df is not None:
        sim_df.to_csv(os.path.join(base_dir, "outputs", "item_similarity.csv"))

def run_pipeline(base_dir):
    tables = load_tables(base_dir)
    merged = merge_tables(tables)
    cleaned = clean_data(merged)
    features = feature_engineering(cleaned)
    eda(features, os.path.join(base_dir, "outputs", "eda"))
    X, y_reg, y_clf = prepare_datasets(features)
    reg_model, reg_metrics = train_regression(X, y_reg) if y_reg is not None else (None, {})
    clf_model, clf_metrics = train_classification(X, y_clf) if y_clf is not None else (None, {})
    user_item = build_user_item_matrix(features)
    sim_df = build_item_similarity(user_item) if user_item is not None else None
    save_artifacts(base_dir, features, X.columns.tolist(), reg_model, clf_model, reg_metrics, clf_metrics, sim_df)
    return {"regression": reg_metrics, "classification": clf_metrics}

def load_artifacts(base_dir):
    paths = {
        "cleaned": os.path.join(base_dir, "outputs", "cleaned_master.csv"),
        "features": os.path.join(base_dir, "outputs", "feature_columns.json"),
        "reg_model": os.path.join(base_dir, "outputs", "regression_model.joblib"),
        "clf_model": os.path.join(base_dir, "outputs", "classification_model.joblib"),
        "sim": os.path.join(base_dir, "outputs", "item_similarity.csv"),
    }
    data = None
    cols = None
    reg = None
    clf = None
    sim = None
    if os.path.exists(paths["cleaned"]):
        data = pd.read_csv(paths["cleaned"], parse_dates=["date"], infer_datetime_format=True)
    if os.path.exists(paths["features"]):
        with open(paths["features"], "r") as f:
            cols = json.load(f)
    if os.path.exists(paths["reg_model"]):
        reg = joblib.load(paths["reg_model"])
    if os.path.exists(paths["clf_model"]):
        clf = joblib.load(paths["clf_model"])
    if os.path.exists(paths["sim"]):
        sim = pd.read_csv(paths["sim"], index_col=0)
    return data, cols, reg, clf, sim

def build_single_feature_row(df, user_id_value, item_id_value):
    user_id = _find_col(df, ["user_id", "userid", "user"])
    item_id = _find_col(df, ["item_id", "itemid", "attraction_id", "item"])
    if not (user_id and item_id):
        return None
    sample = df[(df[user_id] == user_id_value) & (df[item_id] == item_id_value)]
    if sample.empty:
        any_user = df[user_id].iloc[0]
        sample = df[df[user_id] == any_user].iloc[[0]].copy()
        sample[item_id] = item_id_value
    sample = sample.iloc[[0]].copy()
    return sample

def vectorize_row(row_df, feature_columns):
    non_feature = set([_find_col(row_df, ["rating", "score"]), _find_col(row_df, ["visit_mode_name", "mode_name", "mode", "visit_mode"]), "date"])
    feat = row_df.drop(columns=[c for c in non_feature if c in row_df.columns])
    cat_cols = feat.select_dtypes("object").columns.tolist()
    X = pd.get_dummies(feat, columns=cat_cols, drop_first=False)
    X = X.reindex(columns=feature_columns, fill_value=0)
    return X

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    metrics = run_pipeline(base_dir)
    print(json.dumps(metrics)) 
