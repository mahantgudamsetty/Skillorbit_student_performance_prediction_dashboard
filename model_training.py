from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np
import pickle

with open('model/processed_data.pkl', 'rb') as file:
    x_train, x_test, y_train, y_test = pickle.load(file)
parameter_grid=param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True]
}

rfr=RandomForestRegressor()

# we will try to find the best set of parameters for the data we took using 5 fold grid search
# verbose lets me know the progress of the grid search, n_jobs=-1 means use all processors to speed up the process.

grid_search=GridSearchCV(estimator=rfr,param_grid=parameter_grid,cv=5,verbose=1,n_jobs=-1,scoring='r2')
grid_search.fit(x_train,y_train)
print(grid_search.best_params_)
print(grid_search.best_score_)

# the gird has already trained the model with the best parameters, so we can use it directly to predict the test data
rfr_best=grid_search.best_estimator_


y_pred=rfr_best.predict(x_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print("R2 Score for GPA:", r2)
print("MAE for GPA:", mae)
print("MSE for GPA:", mse)
print("RMSE for GPA:", rmse)

# for future use, we will save the models in a pickle file so that we can use them without training again.
with open("Skillorbit_student_performance_prediction_dashboard/model/regressor.pkl", "wb") as f:
    pickle.dump(rfr_best, f)
