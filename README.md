
# Tourism Experience Analytics: Classification, Prediction & Recommendation

- End-to-end data science project that predicts attraction ratings, predicts visit mode, and recommends attractions to users
- Built with pandas, numpy, scikit-learn, cosine similarity, and Streamlit

## Quick Start
- Install dependencies (Windows PowerShell):

```bash
python -m pip install pandas numpy scikit-learn streamlit seaborn matplotlib openpyxl joblib
```

- Generate artifacts (cleaned data, models, metrics, EDA charts):

```bash
python pipeline.py
```

- Launch the app:

```bash
streamlit run app.py
```

- Open http://localhost:8501/

## Project Goals
- Predict Rating (Regression): estimate how much a user will like an attraction
- Predict Visit Mode (Classification): infer trip type (Business, Family, Couples, etc.)
- Recommend Attractions (Recommender): suggest top attractions based on history and similarity

## Data Sources
- Transaction: user visits, ratings, month/year
- User: demographics
- Item: attraction details (+ Updated_Item if provided)
- Type: attraction type/category
- Mode: visit mode dictionary
- City, Country, Region, Continent: location hierarchy

## Pipeline Overview
- Code: [pipeline.py](file:///c:/Users/HP/OneDrive/Desktop/data%20scientist/Labmentix%20Trainingship/Tourism%20Project/Tourism%20Dataset/pipeline.py)
- Steps:
  - Load tables from Excel files
  - Robust column detection to handle naming differences across files
  - Merge Transaction with User, Item, Type, Mode, and location hierarchy
  - Clean data: numeric ratings, date building, duplicates removal, trims/normalization
  - Feature engineering:
    - user_total_visits
    - user_avg_rating
    - attraction_popularity
    - season, month, year derived from date
  - EDA chart generation and save to outputs/eda/
  - Prepare datasets: one-hot encode categoricals, fill numeric NaNs, align targets
  - Train models:
    - RandomForestRegressor for Rating (R², RMSE)
    - RandomForestClassifier for VisitMode (Accuracy, F1)
  - Build user–item matrix and cosine item similarity
  - Save artifacts to outputs/

## Models
- Regression: RandomForestRegressor
  - Metrics saved to outputs/metrics.json
- Classification: RandomForestClassifier
  - Metrics saved to outputs/metrics.json
- Recommendation: item-based collaborative filtering with cosine similarity
  - Cold-start fallback using attraction_popularity or value_counts
  - Similarity saved to outputs/item_similarity.csv

## Streamlit App
- Code: [app.py](file:///c:/Users/HP/OneDrive/Desktop/data%20scientist/Labmentix%20Trainingship/Tourism%20Project/Tourism%20Dataset/app.py)
- Pages:
  - Predict Rating: select user + attraction and view predicted rating
  - Visit Mode Prediction: styled inputs for context; shows predicted visit mode
  - Recommendations: select user; get personalized top-k attractions or popular fallback
- Additional page:
  - [Visit Mode Prediction.py](file:///c:/Users/HP/OneDrive/Desktop/data%20scientist/Labmentix%20Trainingship/Tourism%20Project/Tourism%20Dataset/pages/Visit%20Mode%20Prediction.py)



![My Dashboard](Screenshot2026-02-24230233.png)

![My Dashboard](Screenshott2026-02-24230407.png)
## EDA Outputs
- Generated automatically to outputs/eda/:
  - rating_distribution.png
  - visit_mode_distribution.png
  - top_attractions.png
  - top_countries.png
  - seasonal_trends.png

## Outputs Folder
- outputs/cleaned_master.csv
- outputs/feature_columns.json
- outputs/regression_model.joblib
- outputs/classification_model.joblib
- outputs/item_similarity.csv
- outputs/metrics.json
- outputs/eda/*.png

## Folder Structure
```
Tourism Dataset/
  app.py
  pipeline.py
  pages/
    Visit Mode Prediction.py
  outputs/
    cleaned_master.csv
    feature_columns.json
    regression_model.joblib
    classification_model.joblib
    item_similarity.csv
    metrics.json
    eda/
      rating_distribution.png
      visit_mode_distribution.png
      top_attractions.png
      top_countries.png
      seasonal_trends.png
  City.xlsx
  Continent.xlsx
  Country.xlsx
  Item.xlsx
  Mode.xlsx
  Region.xlsx
  Transaction.xlsx
  Type.xlsx
  Additional_Data_for_Attraction_Sites/Updated_Item.xlsx
```

## How It Works (Short)
- Data is loaded and merged; columns are auto-detected even with varying names
- Cleaned and engineered features are produced for modeling and recommendation
- Models are trained and evaluated; similarity is computed for recommender
- The app loads artifacts; predicts rating/visit mode and recommends attractions

## Notes
- The app rebuilds similarity on the fly if it’s missing
- If user history is sparse, the app shows popular attractions as fallback

## Future Enhancements
- Show attraction names alongside IDs
- Add filters (country, type, season) and CSV export of recommendations
- Hyperparameter tuning and richer features (text embeddings, price bands)

