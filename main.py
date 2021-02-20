import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import plot_confusion_matrix

# Load the file
path = "online_shoppers.csv"
data = pd.read_csv(path)

# Define label = in our case, revenue (true,false)
label = data['Revenue']


# Check possible null values in the dataset
def check_null(data):
    total = data.isnull().sum().sort_values(ascending=False)  # order by desc
    calc1 = data.isnull().sum() / (data.isnull().count()) * 100
    percent = (np.round(calc1, 1)).sort_values(ascending=False)
    null_data = pd.concat([total, percent], axis=1, keys=['total', '%'])
    return null_data


check_null(data)
print(check_null(data))

# Check the categorical variables
print(data.info())

# Encode categorical variables
le = LabelEncoder()


def LabelEncoder(df):
    for i in df.columns:
        if df.dtypes[i] == object or df.dtypes[i] == bool:
            le.fit(df[i].astype(str))
            df[i] = le.transform(df[i].astype(str))
    return df


modified_data = LabelEncoder(data)
# modified_data['Weekend']= le.fit_transform(modified_data['Weekend'])
print(modified_data.info())

# Define input variables and target (label)
cols = ['Session', 'Revenue']
x = modified_data.drop(cols, axis=1)

# Check variables correlation
# But first, create subplots to display multiple plots
f1, ax1 = plt.subplots()
f2, ax2 = plt.subplots()
f3, ax3 = plt.subplots()

correlation = x.corr()
sns.heatmap(correlation, annot=None, fmt="d", cmap="Blues", ax=ax1)
# plt.show()

# After checking strong correlation between 'ProductRelated' and 'ProductRelated_Duration', and 'ExitRates'
# and 'BounceRates', reduce further the number of columns
cols2 = ['Session', 'Revenue', 'ProductRelated_Duration', 'ExitRates']
x = modified_data.drop(cols2, axis=1)
y = modified_data['Revenue']

# Check distribution of label classes
sns.countplot(y, ax=ax2)
# print(sorted((y)).value_counts())

# SMOTE (oversampling)
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = SMOTE().fit_resample(x, y)
# print
# print(sorted((y_sm).value_counts()))

# graph
sns.countplot(y_sm, ax=ax3)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.3, random_state=33)

# Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# XGBoost classifier
xgb_clf = XGBClassifier(eta=0.1, max_depth=20, n_estimators=200)
xgb_clf.fit(X_train, y_train)
xgb_score = xgb_clf.score(X_test, y_test)
xgb_training_score = xgb_clf.score(X_train, y_train)
predictions_xgb = xgb_clf.predict(X_test)
# CHECK OVERFITTING
print("XGB Accuracy score:", xgb_score)
print("XGB TRAINING Accuracy score:", xgb_training_score)
print("XGBoost Classification Report")
print(classification_report(y_test, predictions_xgb))

# Rearrange the dataset columns
label='Revenue'
cols = modified_data.columns.tolist()
colIdx = modified_data.columns.get_loc(label)
# Do nothing if the label is in the 0th position
# Otherwise, change the order of columns to move label to 0th position
if colIdx != 0:
    cols = cols[colIdx:colIdx + 1] + cols[0:colIdx] + cols[colIdx + 1:]
# Change the order of data so that label is in the 0th column
gcp_input = modified_data[cols]

# Convert the data to float
gcp_input = gcp_input.astype('float')
gcp_input['Revenue'] = gcp_input['Revenue'].astype('int')
# Write the dataset as a csv
gcp_input.to_csv('online_shoppers_input.csv', index=False, header=False)

# Confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(xgb_clf, X_test, y_test,
                                 display_labels=[1, 0],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()


