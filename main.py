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
from sklearn.ensemble import GradientBoostingClassifier
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
        if df.dtypes[i] == object:
            le.fit(df[i].astype(str))
            df[i] = le.transform(df[i].astype(str))
    return df


modified_data = LabelEncoder(data)
modified_data['Weekend']= le.fit_transform(modified_data['Weekend'])  #also convert Weekend that is a bool variable
print(modified_data.info())

# Define input variables and target (label)
cols = ['Revenue']
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
cols2 = ['Revenue', 'ProductRelated_Duration', 'ExitRates']
x = modified_data.drop(cols2, axis=1)
y = modified_data['Revenue']

# Check distribution of label classes
sns.countplot(y, ax=ax2)


# SMOTE (oversampling) - to solve class imbalance
smote = SMOTE(sampling_strategy='minority')
X_sm, y_sm = SMOTE().fit_resample(x, y)

# graph
sns.countplot(y_sm, ax=ax3)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=33)

# Scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# XGBoost classifier
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)
xgb_score = xgb_clf.score(X_test, y_test)
xgb_training_score = xgb_clf.score(X_train, y_train)
predictions_xgb = xgb_clf.predict(X_test)

# Gradient Boosting Classifier
gb_clf = GradientBoostingClassifier()
gb_clf.fit(X_train, y_train)
gb_score=gb_clf.score(X_test,y_test)
gb_training_score = gb_clf.score(X_train, y_train)
predictions_gb= gb_clf.predict(X_test)

# Check scores
print("XGBoost Accuracy score:", (round((xgb_score),3))*100,"%")
print("XGBoost Training Accuracy score:", (round((xgb_training_score),3))*100,"%")

print("Gradient Boosting Accuracy score:", (round((gb_score),3))*100,"%")
print("Gradient Boosting Training Accuracy score:", (round((gb_training_score),3))*100,"%")

# Classification reports
print("XGBoost Classification Report")
print(classification_report(y_test, predictions_xgb))
print("Gradient Boosting Classification Report")
print(classification_report(y_test, predictions_gb))

# Rearrange the dataset that will be used as an input for the training job on GCP
label='Revenue'
useless_columns=['ProductRelated_Duration', 'ExitRates']
modified_data=modified_data.drop(useless_columns, axis=1)
cols = modified_data.columns.tolist()
colIdx = modified_data.columns.get_loc(label)
# Check if the label is in the first column. Otherwise, rearranges it
if colIdx != 0:
    cols = cols[colIdx:colIdx + 1] + cols[0:colIdx] + cols[colIdx + 1:]
gcp_input = modified_data[cols]

# Final transformations required by GCP
gcp_input.iloc[:,1:16] = gcp_input.iloc[:,1:16].astype('float')
gcp_input[label] = gcp_input[label].astype('string')
# Writes the final dataset as a csv file
gcp_input.to_csv('online_shoppers_gcp_input.csv', index=False, header=False)

# Confusion matrix
titles_options = [("XGBoost confusion matrix, without normalization", None),
                  ("XGBoost normalized confusion matrix", 'true')]
for title, normalize in titles_options:
    disp = plot_confusion_matrix(xgb_clf, X_test, y_test,
                                 display_labels=[True, False],
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)
    print(title)
    print(disp.confusion_matrix)
plt.show()


