from sklearn.model_selection import GridSearchCV
#GridSearch for hyper-parameter tuning
parameters= {
    'eta': np.array([0.01,0.1]),
    'n_estimators': np.array([100,200]),
    'max_depth':np.array([10,20])
}


grid_check=GridSearchCV(estimator=XGBClassifier(),param_grid=parameters)
grid_check.fit(X_train,y_train)
#Print out best parameters
print(grid_check.best_params_)