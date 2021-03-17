# Online-shoppers-XGBoost

Medium post: https://towardsdatascience.com/xgboost-deployment-on-gcp-with-google-ai-platform-ccf2e5b4d6ea.

The aim of this project, as it's explained in the link above, was to first develop a classification (supervised learning PoC), and after deploy the solution on Google AI Platform.
During the PoC phase, two models were compared: XGBoost and Gradient Boosting Classifier, through Acurracy and the rest of the metrics available in the classification report.
Since the goal was to deploy the best performing one (XGBoost), the code also contains a data transformation to shape the data in the required format to first train a job on GCP.

The original dataset can be found in https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset#. It contains 12.330 rows of sessions from distinct users during the period of one year. The target variable is 'Revenue'. 


