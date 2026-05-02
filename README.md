# ML Individual Final: Flight Delay Prediction

This repository contains my individual Logistic Regression model for predicting whether a U.S. domestic flight will arrive at least 15 minutes late.

The final model focuses on the six major airports used for the condensed version of the project:

| Airport Code | Airport |
|---|---|
| ORD | Chicago O'Hare International Airport |
| ATL | Hartsfield-Jackson Atlanta International Airport |
| DEN | Denver International Airport |
| DFW | Dallas/Fort Worth International Airport |
| CLT | Charlotte Douglas International Airport |
| LAX | Los Angeles International Airport |

The notebook filters the data so that both the departure airport and arrival airport are within this six-airport group. The year 2020 is excluded because COVID-19 heavily disrupted normal airline operations.

## Project Files

| File | Purpose |
|---|---|
| `anthony_logistic_regression_6_airports.ipynb` | Fully executed notebook with results shown under each cell |
| `anthony_logistic_regression_6_airports.py` | Python script version of the notebook |
| `logistic_flight_delay_model.sav` | Saved trained Logistic Regression pipeline |
| `logistic_feature_names.sav` | Saved list of model feature names |
| `requirements.txt` | Python packages needed to run the notebook/script |

## Predictive Question

Can we predict whether a flight between the selected major airports will arrive 15 minutes or more late using schedule, route, airline, date, and congestion-related features?

## Data Scope

The model uses the filtered six-airport dataset with these retained years:

| Year | Rows |
|---:|---:|
| 2018 | 118,388 |
| 2019 | 180,364 |
| 2021 | 148,091 |
| 2022 | 85,941 |
| **Total** | **532,784** |

Overall delay rate in this dataset: **20.18%**

## Features Used

| Feature Group | Variables |
|---|---|
| Temporal | `Year`, `Quarter`, `Month`, `DayofMonth`, `DayOfWeek` |
| Schedule | `CRSDepTime`, `CRSArrTime`, `DepTimeBlk` |
| Route | `Origin`, `Dest`, `Distance`, `DistanceGroup` |
| Airline | `Marketing_Airline_Network`, `Operating_Airline` |
| Congestion | `origin_hourly_departures`, `dest_hourly_arrivals` |

The congestion variables are engineered from scheduled flight volume only, so they do not use actual arrival delay information.

## Model

My individual model is Logistic Regression, implemented with `SGDClassifier(loss="log_loss")`. This keeps the model comparable to standard Logistic Regression while making it much faster on the larger airline dataset.

## Model Choice Justification

I chose **Logistic Regression** because our prediction is a simple yes/no outcome: whether a flight arrives 15 minutes or more late. It is a good fit because it is easy to explain, runs quickly, and gives us a strong baseline to compare against Prince's and Alex's more advanced models.

Logistic Regression also helps us understand which features are connected to delays, such as airport route, airline, scheduled time, day of week, and congestion. Since my model focuses on the six-airport route network, it gives us a cleaner and more focused version of the delay prediction problem.

Overall, this model is useful because it is practical, interpretable, and fair to compare against Random Forest and XGBoost/Gradient Boosting.

The preprocessing pipeline handles:

| Data Type | Processing |
|---|---|
| Numeric features | Median imputation and standard scaling |
| Categorical features | Most-frequent imputation and one-hot encoding |
| Target imbalance | Balanced class weights |

## Results

The model was evaluated on a held-out 20% test set.

| Metric | Score |
|---|---:|
| Accuracy | 0.5730 |
| Precision | 0.2629 |
| Recall | 0.6188 |
| F1 Score | 0.3691 |
| ROC-AUC | 0.6217 |

The model is better at identifying delayed flights than a simple baseline, but it still creates many false positives. This makes sense because arrival delays are difficult to predict using schedule and route data alone. Weather, crew availability, aircraft maintenance, and real-time airport conditions would likely improve performance.

## How To Run

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the notebook:

```bash
jupyter notebook anthony_logistic_regression_6_airports.ipynb
```

Or run the script version:

```bash
python anthony_logistic_regression_6_airports.py
```

## Note About Data

The original airline data files are not included in this GitHub repository because they are too large. The notebook expects the prepared local data file to exist at:

```text
/Users/anthonyhanna/Documents/New project/team_data_6_airports/top6_airport_routes_excl2020_with_congestion.parquet
```

If the data file is moved, update the `CACHE_FILE` path near the top of the notebook/script.
