#!/usr/bin/env python
# coding: utf-8

# ### BSAN 6070_Spring2026_ML_Final_Project
# # Flight Delay Prediction
# ### Anthony's Individual Model: Logistic Regression - Top 6 Airport Route Version
# 
# **Goal:** Train a Logistic Regression model only on flights where both origin and destination are one of `ORD`, `ATL`, `DEN`, `DFW`, `CLT`, and `LAX`. This version excludes 2020 and uses the same feature groups as Prince and Alex, including engineered congestion features.
# 

# ## 1. Setup & Imports
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_score, recall_score, f1_score, accuracy_score
)
import joblib

RANDOM_STATE = 42
SAMPLE_FRAC = 0.40

print("Imports loaded successfully.")


# ## 2. Data Loading - Top 6 Airport OD Routes
# This notebook uses only routes where both `Origin` and `Dest` are in:
# 
# `ORD, ATL, DEN, DFW, CLT`
# 
# It uses the retained non-2020 years: **2018, 2019, 2021, and 2022**.
# 
# The file loaded below was pre-filtered from the full Parquet files and includes two leakage-free congestion proxy features:
# - `origin_hourly_departures`
# - `dest_hourly_arrivals`
# 

# In[2]:


DATA_PATH = "/Users/anthonyhanna/Documents/New project/team_data_6_airports/top6_airport_routes_excl2020_with_congestion.parquet"
TOP_AIRPORTS = ['ORD', 'ATL', 'DEN', 'DFW', 'CLT', 'LAX']
TARGET = 'ArrDel15'

df = pd.read_parquet(DATA_PATH)
df[TARGET] = df[TARGET].astype(int)

print(f"Loaded top 6 airport route dataset shape: {df.shape}")
print(f"Airport set: {TOP_AIRPORTS}")
print(f"Overall delay rate: {df[TARGET].mean():.4f}")
print()
print("Rows by year:")
print(df['Year'].value_counts().sort_index())

df.head(3)


# ## 3. Shared Target and Feature Set
# This mirrors Prince and Alex's feature grouping so the model comparison stays fair.
# 
# The model uses temporal, schedule, route, airline, and congestion features. Departure features like `DepDelay`, `DepDel15`, and `TaxiOut` are listed as conditional features but are not included in this pre-departure prediction setup.
# 

# In[3]:


# CANDIDATE FEATURE GROUPS
TEMPORAL_FEATURES   = ['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek']
SCHEDULE_FEATURES   = ['CRSDepTime', 'CRSArrTime', 'DepTimeBlk']
ROUTE_FEATURES      = ['Origin', 'Dest', 'Distance', 'DistanceGroup']
AIRLINE_FEATURES    = ['Marketing_Airline_Network', 'Operating_Airline']
CONGESTION_FEATURES = ['origin_hourly_departures', 'dest_hourly_arrivals']
DEPARTURE_FEATURES  = ['DepDelay', 'DepDel15', 'TaxiOut']  # conditional; not used here

ALL_CANDIDATES = (
    TEMPORAL_FEATURES + SCHEDULE_FEATURES +
    ROUTE_FEATURES + AIRLINE_FEATURES +
    CONGESTION_FEATURES
)

available_candidates = [c for c in ALL_CANDIDATES if c in df.columns]
X = df[available_candidates].copy()
y = df[TARGET].copy().astype(int)

print("Candidate features by group:")
for grp, feats in [
    ('Temporal', TEMPORAL_FEATURES),
    ('Schedule', SCHEDULE_FEATURES),
    ('Route', ROUTE_FEATURES),
    ('Airline', AIRLINE_FEATURES),
    ('Congestion', CONGESTION_FEATURES),
    ('Departure conditional', DEPARTURE_FEATURES)
]:
    print(f"  {grp}: {feats}")

print()
print("Actual features used:")
print(available_candidates)
print()
print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Delay rate: {y.mean():.4f}")
print()
print("Missing values per feature:")
print(X.isnull().sum()[X.isnull().sum() > 0].sort_values(ascending=False))


# ## 4. Exploratory Data Analysis Before Modeling
# Before training Logistic Regression, we check the distribution across years, routes, airports, target class, and congestion features.
# 

# In[4]:


print("Rows by year:")
year_counts = df['Year'].value_counts().sort_index()
print(year_counts)

print()
print("Delay rate by year:")
delay_by_year = df.groupby('Year')[TARGET].mean().round(4)
print(delay_by_year)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

year_counts.plot(kind='bar', ax=axes[0], color='steelblue')
axes[0].set_title('Rows by Year - Top 6 Airport Routes')
axes[0].set_xlabel('Year')
axes[0].set_ylabel('Rows')

delay_by_year.plot(kind='bar', ax=axes[1], color='darkorange')
axes[1].set_title('Arrival Delay 15+ Rate by Year')
axes[1].set_xlabel('Year')
axes[1].set_ylabel('Delay Rate')
axes[1].set_ylim(0, max(delay_by_year.max() * 1.25, 0.25))

plt.tight_layout()
plt.show()


# In[5]:


print("Origin-Destination matrix:")
od_matrix = pd.crosstab(df['Origin'], df['Dest']).reindex(index=TOP_AIRPORTS, columns=TOP_AIRPORTS, fill_value=0)
display(od_matrix)

fig, ax = plt.subplots(figsize=(7, 5))
sns.heatmap(od_matrix, annot=True, fmt=',d', cmap='Blues', cbar=False, ax=ax)
ax.set_title('Route Counts Among Top 6 Airports')
ax.set_xlabel('Destination')
ax.set_ylabel('Origin')
plt.tight_layout()
plt.show()

print()
print("Top airlines in this subset:")
print(df['Marketing_Airline_Network'].value_counts().head(10))


# In[6]:


numeric_cols_for_eda = [
    'Distance', 'CRSDepTime', 'CRSArrTime',
    'origin_hourly_departures', 'dest_hourly_arrivals'
]

numeric_summary = df[numeric_cols_for_eda].describe().T
print("Numeric and congestion feature summary:")
display(numeric_summary)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

df['origin_hourly_departures'].plot(kind='hist', bins=20, ax=axes[0], color='seagreen')
axes[0].set_title('Origin Hourly Departures')
axes[0].set_xlabel('Scheduled departures in same origin-hour')

df['dest_hourly_arrivals'].plot(kind='hist', bins=20, ax=axes[1], color='slateblue')
axes[1].set_title('Destination Hourly Arrivals')
axes[1].set_xlabel('Scheduled arrivals in same destination-hour')

plt.tight_layout()
plt.show()

missing_summary = df[[TARGET] + available_candidates].isna().sum().sort_values(ascending=False)
missing_summary = missing_summary[missing_summary > 0]
print()
print("Missing values in modeling data:")
print(missing_summary if len(missing_summary) else "No missing values in selected modeling columns.")


# ## 5. Train / Test Split
# Use the same split settings as the team.
# 
# This six-airport subset is small enough to train Logistic Regression on the full training split, but the same capped training safeguard remains in the code.
# 

# In[7]:


X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

# Training cap stays here as a safety guard, but this subset should train on all rows.
TRAIN_CAP = 500_000

train_pool = X_train_full.copy()
train_pool[TARGET] = y_train_full.values

if len(train_pool) > TRAIN_CAP:
    train_frac = TRAIN_CAP / len(train_pool)
    train_sample = (
        train_pool
        .groupby(['Year', TARGET], group_keys=False)
        .sample(frac=train_frac, random_state=RANDOM_STATE)
    )
else:
    train_sample = train_pool

X_train = train_sample.drop(columns=[TARGET])
y_train = train_sample[TARGET].astype(int)

print(f"Full train pool size : {X_train_full.shape[0]:,}")
print(f"LR train subset size : {X_train.shape[0]:,}")
print(f"Full test size       : {X_test.shape[0]:,}")
print(f"Train delay rate     : {y_train.mean():.4f}")
print(f"Test delay rate      : {y_test.mean():.4f}")
print()
print("LR train subset by year:")
print(X_train.assign(ArrDel15=y_train).groupby('Year')['ArrDel15'].agg(['count', 'mean']).round(4))


# ## 6. Logistic Regression Preprocessing
# For Logistic Regression, we need:
# - median imputation for numeric columns
# - one-hot encoding for categorical columns
# - feature scaling for numeric columns
# 
# The congestion features are numeric and are scaled with the other numeric variables.
# 

# In[8]:


categorical_cols = ['Marketing_Airline_Network', 'Operating_Airline', 'Origin', 'Dest', 'DepTimeBlk']
categorical_cols = [c for c in categorical_cols if c in X_train.columns]
numeric_cols = [c for c in X_train.columns if c not in categorical_cols]

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

print("Numeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)


# ## 7. Model Choice Justification
# Fill this in for your report after you run the model.
# 
# Suggested direction:
# - Logistic Regression is a strong baseline for binary classification.
# - It is interpretable and easy to explain.
# - It lets us see direction and relative importance of features through coefficients.
# - It works well as a comparison model against tree-based methods like Random Forest and XGBoost.
# - This version tests whether a focused six-airport route network can be modeled more cleanly than the full national dataset.
# - Congestion features are included because they use scheduled flight volume and do not rely on post-flight outcome information.
# 

# ## 8. Baseline Logistic Regression Model
# This uses stochastic gradient descent with logistic loss. It is still a Logistic Regression model, but it trains quickly on the six-airport route subset.
# 

# In[9]:


lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', SGDClassifier(
        loss='log_loss',
        max_iter=5,
        tol=1e-3,
        class_weight='balanced',
        average=True,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=2,
        n_jobs=-1,
        random_state=RANDOM_STATE
    ))
])

lr_pipeline.fit(X_train, y_train)

y_pred_lr = lr_pipeline.predict(X_test)
y_prob_lr = lr_pipeline.predict_proba(X_test)[:, 1]

print("=== Logistic Regression Results ===")
print(classification_report(y_test, y_pred_lr, target_names=['On Time', 'Delayed']))
print(f"Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred_lr):.4f}")
print(f"F1 Score : {f1_score(y_test, y_pred_lr):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob_lr):.4f}")


# ## 9. Confusion Matrix
# 

# In[10]:


cm = confusion_matrix(y_test, y_pred_lr)

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['On Time', 'Delayed'],
            yticklabels=['On Time', 'Delayed'], ax=ax)
ax.set_title('Logistic Regression - Confusion Matrix', fontsize=13, fontweight='bold')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.show()


# ## 10. ROC Curve
# 

# In[11]:


fpr, tpr, _ = roc_curve(y_test, y_prob_lr)
auc_score = roc_auc_score(y_test, y_prob_lr)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Logistic Regression (AUC = {auc_score:.3f})')
ax.plot([0, 1], [0, 1], 'k--', lw=1.5)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curve - Logistic Regression', fontsize=13, fontweight='bold')
ax.legend(loc='lower right')
plt.tight_layout()
plt.show()


# ## 11. Coefficient Interpretation
# This is one of the advantages of Logistic Regression.
# 

# In[12]:


feature_names = lr_pipeline.named_steps['preprocessor'].get_feature_names_out()
coefficients = lr_pipeline.named_steps['model'].coef_[0]

coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'AbsCoefficient': np.abs(coefficients)
}).sort_values('AbsCoefficient', ascending=False)

print("Top 15 features by absolute coefficient:")
print(coef_df[['Feature', 'Coefficient']].head(15).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
top_coef = coef_df.head(15).sort_values('Coefficient')
ax.barh(top_coef['Feature'], top_coef['Coefficient'], color='steelblue')
ax.set_title('Top 15 Logistic Regression Coefficients', fontsize=13, fontweight='bold')
ax.set_xlabel('Coefficient Value')
plt.tight_layout()
plt.show()


# ## 12. Save Artifacts
# 

# In[13]:


joblib.dump(lr_pipeline, 'logistic_flight_delay_model.sav')
joblib.dump(available_candidates, 'logistic_feature_names.sav')

print("Saved: logistic_flight_delay_model.sav")
print("Saved: logistic_feature_names.sav")

