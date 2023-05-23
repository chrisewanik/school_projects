# %%
# #adding the optuna for google colab 
# !pip install --quiet optuna
# import optuna
# optuna.__version__

# %%
# Importing Libraries
import numpy as np
import pandas as pd
import sklearn
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import optuna
import xgboost as xgb
from sklearn.linear_model import ElasticNet
import time
import numpy as np
from sklearn.utils import check_random_state
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# Set Random Seeds
np.random.seed(37)
rng = check_random_state(37)

# %%
# Load the Dataset in
# Colab
# Data = pd.read_csv('/content/drive/My Drive/COS598_Project/auto_cleaned.csv')
# Local
Data = pd.read_csv('auto_cleaned.csv')

# Spliting  into features and target
target = Data['mpg']
features = Data.drop(['mpg'], axis=1)

# %%
#Train Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=37)

# %% [markdown]
# First we train Kernel Ridge Regression

# %%
# Starting timing of KRR with Optuna
start_time = time.time()

# %%
# KRR Objective Function for Optuna
def objective(trial):    
    """Kernel Ridge Regression objective function for Optuna to minimize.

    Args:
        trial (optuna.trial.Trial): A trial object 

    Returns:
        float: The mean squared error of the model on the testing data
    """    
    # Hyperparameters we want to tune

    # Set the range for alpha to be uniform between 0.0 and 1.0, alpha is the regularization parameter
    alpha = trial.suggest_uniform('alpha', 0.0, 1.0)

    # Set the range for gamma to be uniform between 0.0 and 1.0, gamma is the RBF, laplacian, polynomial, exponential chi2 and sigmoid kernels
    gamma = trial.suggest_uniform('gamma', 0.0, 1.0)

    # Set the range for degree to be uniform between 0 and 5, degree is Degree of the polynomial kernel
    degree = trial.suggest_int('degree', 0, 5)

    # Set the range for coef0 to be uniform between 0.0 and 1.0, coef0 is Zero coefficient for polynomial and sigmoid kernels. Ignored by other kernels.
    coef0 = trial.suggest_uniform('coef0', 0.0, 1.0)

    # Set the range for kernel to be either linear, poly, rbf, sigmoid
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])

    # Create the KernelRidge model with the specified hyperparameters
    model = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=37)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Get the MSE of the model on the testing data
    mse = mean_squared_error(y_test, model.predict(X_test))

    # Return mse as the objective value
    return mse

# %%
# Create the Optuna study, we maximize the objective function (the score)
# model.score() returns the coefficient of determination R^2 of the prediction 
# i.e. the percentage of the variance in the target variable that is predictable from the feature variables
study = optuna.create_study(direction='minimize', sampler = TPESampler(seed=37), pruner=MedianPruner())
study.optimize(objective, n_trials=100)

# Print the number of finished trials
print("Number of finished trials: ", len(study.trials))

# Print the best trial and save as a variable
print("Best trial:")
trial = study.best_trial

# Print the value of the final trial and the best parameters
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %%
# Fit Kernel Ridge Regression using the optimal hyperparameters found by Optuna
kr1 = KernelRidge(alpha=study.best_params['alpha'], kernel=study.best_params['kernel'], gamma=study.best_params['gamma'], degree=study.best_params['degree'], coef0=study.best_params['coef0'])

# Fit the trainng data to the model
kr1.fit(X_train, y_train)

# Get the training and testing mse
train_mse = mean_squared_error(y_train, kr1.predict(X_train))
test_mse = mean_squared_error(y_test, kr1.predict(X_test))

# Get the time it took to run the model
end_time = time.time()
duration = end_time - start_time

# Print the results
print(f"Tuned Training set score: {train_mse:}")
print(f"Tuned Test set score: {test_mse:}")
print(f"Training duration: {duration:.2f} seconds")

# %%
# Starting timing of KRR
start_time = time.time()

# Train a Kernel Ridge Regression model with the default hyperparameters
kr2 = KernelRidge()

# Fit the trainng data to the model
kr2.fit(X_train, y_train)

# Get the training and testing mse
train_mse = mean_squared_error(y_train, kr2.predict(X_train))
test_mse = mean_squared_error(y_test, kr2.predict(X_test))

# Get the time it took to run the model
end_time = time.time()
duration = end_time - start_time

# Print the results
print(f"Default Training set score: {train_mse:}")
print(f"Default Test set score: {test_mse:}")
print(f"Training duration: {duration:.2f} seconds")

# %% [markdown]
# Next we train Elastic Net Regression

# %%
# Starting timing of Elastic Net with Optuna
start_time = time.time()

# %%
# Elastic Net Objective Function for Optuna
def objective(trial):
    """Elastic Net objective function for Optuna to minimize.

    Args:
        trial (optuna.trial.Trial): A trial object

    Returns:
        float: The mean squared error of the model on the testing data
    """    
    
    # Hyperparameters we want to tune

    # Set the range for alpha to be uniform between 0.0 and 1.0, alpha is the regularization parameter
    alpha = trial.suggest_uniform('alpha', 0.0, 1.0)

    # Set the range for l1_ratio to be uniform between 0.0 and 1.0, l1_ratio is the ElasticNet mixing parameter
    l1_ratio = trial.suggest_uniform('l1_ratio', 0.0, 1.0)

    # Set the range for max_iter to be between 100 and 1000, max_iter is the maximum number of iterations
    max_iter = trial.suggest_int('max_iter', 100, 1000)

    # Create the ElasticNet model with the specified hyperparameters
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=37)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Get the MSE of the model on the testing data
    mse = mean_squared_error(y_test, model.predict(X_test))

    # Return mse as the objective value
    return mse

# %%
# Create the Optuna study, we maximize the objective function (the score)
# model.score() returns the coefficient of determination R^2 of the prediction 
# i.e. the percentage of the variance in the target variable that is predictable from the feature variables
study = optuna.create_study(direction='minimize', sampler = TPESampler(seed=37), pruner=MedianPruner())
study.optimize(objective, n_trials=100)

# Print the number of finished trials
print("Number of finished trials: ", len(study.trials))

# Print the best trial and save as a variable
print("Best trial:")
trial = study.best_trial

# Print the value of the final trial and the best parameters
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %%
# Create the model using the best parameters
model = ElasticNet(alpha=study.best_params['alpha'], l1_ratio=study.best_params['l1_ratio'], max_iter=study.best_params['max_iter'])

# Fit the model
model.fit(X_train, y_train)

# Get the training and testing mse
train_mse = mean_squared_error(y_train, model.predict(X_train))
test_mse = mean_squared_error(y_test, model.predict(X_test))

# Get the time it took to run the model
end_time = time.time()
duration = end_time - start_time

# Print the score on the training and testing sets
print(f"Tuned Training set MSE: {train_mse:}")
print(f"Tuned Test set MSE: {test_mse:}")
print(f"Training duration: {duration:.2f} seconds") 

# %%
# Starting timing of Elastic Net
start_time = time.time()

# Train an ElasticNet model with the default hyperparameters
model = ElasticNet()

# Fit the model
model.fit(X_train, y_train)

# Get the training and testing mse
train_mse = mean_squared_error(y_train, model.predict(X_train))
test_mse = mean_squared_error(y_test, model.predict(X_test))

# Get the time it took to run the model
end_time = time.time()
duration = end_time - start_time

# Print the score on the training and testing sets
print(f"Default Training set score: {train_mse:}")
print(f"Default Test set score: {test_mse:}")
print(f"Training duration: {duration:.2f} seconds") 

# %% [markdown]
# Finally we train an xgboost model

# %%
# Starting timing of XGBoost Regression with Optuna
start_time = time.time()

# %%
# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# %%
# XGBoost Regression Objective Function for Optuna
def objective(trial):
    """XGBoost objective function for Optuna to minimize.

    Args:
        trial (optuna.trial.Trial): A trial object

    Returns:
        float: The mean squared error of the model on the testing data
    """    

    # Hyperparameters we want to tune
    param = {
        "verbosity": 0,
        "objective": "reg:squarederror",
        # use exact for small dataset.
        "tree_method": "exact",
        # defines booster, gblinear for linear functions.
        "booster": trial.suggest_categorical("booster", ["gbtree", "dart"]),
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        # sampling ratio for training data.
        "subsample": trial.suggest_float("subsample", 0.2, 1.0),
        # sampling according to each tree.
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
    }
    # If the booster is dart or gbtree, set additional parameters
    if param["booster"] in ["gbtree", "dart"]:
        # maximum depth of the tree, signifies complexity of the tree.
        param["max_depth"] = trial.suggest_int("max_depth", 3, 9, step=2)
        # minimum child weight, larger the term more conservative the tree.
        param["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
        # eta value, step size shrinkage, similar to learning rate.
        param["eta"] = trial.suggest_float("eta", 1e-8, 1.0, log=True)
        # defines how selective algorithm is.
        param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)
        # grow policy determines the pruning of the trees.
        param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
    # If the booster is dart, set additional parameters
    if param["booster"] == "dart":
        # Sample type is either uniform or weighted,
        param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
        # normalize type is either tree or forest.
        param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
        # rate_drop is dropout rate
        param["rate_drop"] = trial.suggest_float("rate_drop", 1e-8, 1.0, log=True)
        # one drop is probability of skipping dropout
        param["skip_drop"] = trial.suggest_float("skip_drop", 1e-8, 1.0, log=True)

    # Train the model with the specified parameters
    bst = xgb.train(param, dtrain_reg)

    # Get the predictions
    preds = bst.predict(dtest_reg)

    # Calculate mse
    mse = sklearn.metrics.mean_squared_error(y_test, preds)
    
    # Return the mse as the objective value
    return mse


# %%
# Create the Optuna study, we maximize the objective function (the score)
# model.score() returns the coefficient of determination R^2 of the prediction 
# i.e. the percentage of the variance in the target variable that is predictable from the feature variables
study = optuna.create_study(direction='minimize', sampler = TPESampler(seed=37), pruner=MedianPruner())
study.optimize(objective, n_trials=100)

# Print the number of finished trials
print("Number of finished trials: ", len(study.trials))

# Print the best trial and save as a variable
print("Best trial:")
trial = study.best_trial

# Print the value of the final trial and the best parameters
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# %%
# Create the model using the best parameters with the objective as the squared error (which is MSE) and the exact tree method
best_params = study.best_trial.params

# Change the booster key to Booster to avoid an error
best_params["Booster"] = best_params.pop("booster")

# Create the model
model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', tree_method='exact')

# Fit the model
model.fit(X_train, y_train)

# Predict the test set
y_pred = model.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, model.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)

# Get the time it took to run the model
end_time = time.time()
duration = end_time - start_time

# Print the results
print("Tuned Train MSE: %f" % (train_mse))
print("Tuned Test MSE: %f" % (test_mse))
print(f"Training duration: {duration:.2f} seconds") 

# %%
# Starting timing of XGBoost Regression
start_time = time.time()

# Create the model using the default parameters
model = xgb.XGBRegressor(objective='reg:squarederror', tree_method='exact')

# Fit the model
model.fit(X_train, y_train)

# Predict the test set
y_pred = model.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, model.predict(X_train))
test_mse = mean_squared_error(y_test, y_pred)

# Get the time it took to run the model
end_time = time.time()
duration = end_time - start_time

# Print the results
print("Default Train MSE: %f" % (train_mse))
print("Default Test MSE: %f" % (test_mse))
print(f"Training duration: {duration:.2f} seconds") 


