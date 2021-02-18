import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, make_scorer

# Load the file
path = "online_shoppers.csv"
data = pd.read_csv(path)
# Check first 5 rows
data.head()
