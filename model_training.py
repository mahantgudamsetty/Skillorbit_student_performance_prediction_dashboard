from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import numpy as np
import pickle

with open('Skillorbit_student_performance_prediction_dashboard/model/processed_data.pkl', 'rb') as file:
    x_train, x_test, y_train, y_test, y1_train, y1_test = pickle.load(file)
parameter_grid=param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"],
    "bootstrap": [True]
}
rfc=RandomForestClassifier()
rfr=RandomForestRegressor()

# we will try to find the best set of parameters for the data we took using 5 fold grid search
# verbose lets me know the progress of the grid search, n_jobs=-1 means use all processors to speed up the process.
grid_search=GridSearchCV(estimator=rfc,param_grid=parameter_grid,cv=5,verbose=1,n_jobs=-1,scoring='accuracy')
grid_search.fit(x_train,y_train)
grid1_search=GridSearchCV(estimator=rfr,param_grid=parameter_grid,cv=5,verbose=1,n_jobs=-1,scoring='r2')
grid1_search.fit(x_train,y1_train)
best_parameters=grid_search.best_params_
best1_parameters=grid1_search.best_params_
print(grid_search.best_params_)
print(grid_search.best_score_)
print(grid1_search.best_params_)
print(grid1_search.best_score_)

# the gird has already trained the model with the best parameters, so we can use it directly to predict the test data
rfr_best=grid1_search.best_estimator_
rfc_best=grid_search.best_estimator_ 
y_pred=rfc_best.predict(x_test)
y1_pred=rfr_best.predict(x_test)

acc=accuracy_score(y_test, y_pred)
print("Accuracy Score for GradeClass:", acc)
class_report=classification_report(y_test, y_pred)
print("Classification Report for GradeClass:", class_report)
conf_matrix=confusion_matrix(y_test, y_pred)
print("Confusion Matrix for GradeClass:", conf_matrix)
r2_1 = r2_score(y1_test, y1_pred)
mae1 = mean_absolute_error(y1_test, y1_pred)
mse1 = mean_squared_error(y1_test, y1_pred)
rmse1 = np.sqrt(mse1)
print("R2 Score for GPA:", r2_1)
print("MAE for GPA:", mae1)
print("MSE for GPA:", mse1)
print("RMSE for GPA:", rmse1)

# for future use, we will save the models in a pickle file so that we can use them without training again.
with open("Skillorbit_student_performance_prediction_dashboard/model/classifier.pkl", "wb") as f:
    pickle.dump(rfc_best, f)

with open("Skillorbit_student_performance_prediction_dashboard/model/regressor.pkl", "wb") as f:
    pickle.dump(rfr_best, f)