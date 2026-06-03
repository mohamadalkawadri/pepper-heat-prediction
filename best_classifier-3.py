import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import KBinsDiscretizer


data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

target_column = 'Scoville Heat Units (SHU)'

def one_encode(df, feature):
    uniques = df[feature].unique()

    for value in uniques:
        df[f"{feature}_{value}"] = np.where(df[feature] == value, 1, 0)

def prepprocess(df):
    # Encoding all categories.
    one_encode(df, "color")
    one_encode(df, "Harvest Time")
    one_encode(df,'Average Temperature During Storage (celcius)')
    # Replacing missing valeus with zero.
    df.fillna(0, inplace=True)

def drop_outliers(df):
    z_scores = (df - df.mean()) / df.std()
    # Removing outliers, by using the z-score, by deleting all those that are more than 3 std
    df[np.abs(z_scores) < 3].dropna(inplace=True)

prepprocess(data)
prepprocess(test_data)

# This is the best classifier (uses bins) and can be found in best_classifier.py
best_pipeline = Pipeline([('classifier', RandomForestClassifier(n_estimators=100))])
best_features = ['Seed Count', 'Capsaicin Content', 'Pericarp Thickness (mm)', 'Moisture Content', 'Firmness', 'Sugar Content', 'Vitamin C Content (mg)', 'Length (cm)']
best_num_bins = 5
current_df = data[best_features + [target_column]].copy()
drop_outliers(current_df)
new_target, bin_edges = pd.qcut(current_df[target_column], best_num_bins, duplicates='drop', labels=False, retbins=True)
current_df[target_column] = new_target
best_pipeline.fit(current_df[best_features], current_df[target_column])
predictions = best_pipeline.predict(test_data.fillna(0)[best_features])

# Transforming bins back to numbers
for i in range(len(bin_edges)):
    predictions[predictions == i] = bin_edges[i]

df = pd.DataFrame(predictions, columns=[target_column])
df.index.name = 'index'
df.to_csv("predictions.csv")