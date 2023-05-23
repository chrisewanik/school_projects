# %%
# #adding the optuna for google colab 
# !pip install --quiet optuna
# import optuna
# optuna.__version__

# %%
# Importing Libraries
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics
import optuna
from sklearn.utils import check_random_state
import time
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb

# %%
# Set Random Seeds
np.random.seed(37)
rng = check_random_state(37)

# %%
# from google.colab import drive
# drive.mount('/content/drive')

# %%
# Load the Dataset in
# Colab
# Data = pd.read_csv('/content/drive/My Drive/COS598_Project/adult_cleaned.csv')#.to_numpy()
# Local
Data = pd.read_csv('adult_cleaned.csv')

#data_target = Data[:,-1]
target = Data['income']

# Remove the income column from the dataset
features = Data.drop(['income'], axis=1)

# %%
# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3,random_state=37) # 70% training and 30% test

# %% [markdown]
# First the SVM

# %%
# Starting timing of SVM with Optuna
start_time = time.time()

# %%
# SVM Objective Function for Optuna
def objective(trial):
  """Optuna SVM objective function.

  Args:
      trial (optuna.trial.Trial): A trial object

  Returns:
      float: Accuracy of the model.
  """  
  # Hyperparameters we want to tune
  kernel=trial.suggest_categorical('kernel',['rbf','poly','linear','sigmoid'])
  c=trial.suggest_float("C",0.1,3.0,log=True)
  gamma=trial.suggest_categorical('gamma',['auto','scale'])
  degree=trial.suggest_int("degree",1,3,log=True)

  #Create a svm Classifier
  o_clf = SVC(kernel=kernel, C=c, gamma=gamma, degree=degree, random_state=37)

  #Train the model using the training sets
  o_clf.fit(X_train, y_train)

  #Predict the response for test dataset
  o_y_pred = o_clf.predict(X_test)

  # Return accuracy score of prediction
  return metrics.accuracy_score(y_test, o_y_pred)

# %%
# Create the Optuna study, we maximize the objective function (accuracy)
study = optuna.create_study(direction='maximize', sampler = TPESampler(seed=37), pruner=MedianPruner())
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

# Create a svm Classifier with the best parameters
clf = SVC(**best_params, random_state=37)

# Fit the model
clf.fit(X_train, y_train)

# Predict the test set
y_pred = clf.predict(X_test)

# Evaluate the model
train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

# Get the time it took to run the model
end_time = time.time()
duration = end_time - start_time

# Print the results
print("Tuned Train accuracy: %f" % (train_acc))
print("Tuned Test accuracy: %f" % (test_acc))
print(f"Training duration: {duration:.2f} seconds")

# %%
# Starting timing of SVM
start_time = time.time()

# Create a SVM Classifier with the default parameters
clf = SVC(random_state=37)

# Fit the model
clf.fit(X_train, y_train)

# Predict the test set
y_pred = clf.predict(X_test)

# Evaluate the model
train_acc = accuracy_score(y_train, clf.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

# Get the time it took to run the model
end_time = time.time()
duration = end_time - start_time


# Print the results
print("Default Train accuracy: %f" % (train_acc))
print("Default Test accuracy: %f" % (test_acc))
print(f"Training duration: {duration:.2f} seconds")

# %% [markdown]
# Next the Naive Bayes

# %%
# Starting timing of Naive Bayes with Optuna
start_time = time.time()

# %%
# Naive Bayes Objective Function for Optuna
def objective(trial):
    """Optuna Naive Bayes objective function.

    Args:
        trial (optuna.trial.Trial): A trial object 

    Returns:
        float: Accuracy of the model.
    """
    # Set the range for var_smoothing to be uniform between 1e-10 and 1e-8, var_smoothing is the smoothing parameter
    var_smoothing = trial.suggest_uniform('var_smoothing', 1e-10, 1e-8)

    # Create the Gaussian Naive Bayes model with the specified hyperparameters
    model = GaussianNB(var_smoothing=var_smoothing)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=37)

    # Fit the model to the training data
    model.fit(X_train, y_train)

    # Return the score of the model on the testing data
    return model.score(X_test, y_test)

# %%
# Create the Optuna study, we maximize the objective function (accuracy)
study = optuna.create_study(direction='maximize', sampler = TPESampler(seed=37), pruner=MedianPruner())
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


# Store the best parameters in a variable
best_params = study.best_trial.params

# Create a Gaussian Naive Bayes Classifier with the best parameters
nb = GaussianNB(**best_params)

# Fit the model
nb.fit(X_train, y_train)

# Predict the test set
y_pred = nb.predict(X_test)

# Evaluate the model
train_acc = accuracy_score(y_train, nb.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

# Get the time it took to run the model
end_time = time.time()
duration = end_time - start_time

# Print the results
print("Tuned Train accuracy: %f" % (train_acc))
print("Tuned Test accuracy: %f" % (test_acc))
print(f"Training duration: {duration:.2f} seconds")

# %%
# Starting timing of Naive Bayes 
start_time = time.time()

# Fit a Gaussian Naive Bayes Classifier with the default parameters
nb = GaussianNB()

# Fit the model
nb.fit(X_train, y_train)

# Predict the test set
y_pred = nb.predict(X_test)

# Evaluate the model
train_acc = accuracy_score(y_train, nb.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

# Get time it took to run the model
end_time = time.time()
duration = end_time - start_time

# Print the results
print("Default Train accuracy: %f" % (train_acc))
print("Default Test accuracy: %f" % (test_acc))
print(f"Training duration: {duration:.2f} seconds")

# %% [markdown]
# Finally the XGBoost

# %%
# Starting timing of XGBoost Classification with Optuna
start_time = time.time()

# %%
# Set up classification matricies
dtrain = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, y_test, enable_categorical=True)

# %%
# XGBoost Classification Objective Function for Optuna
def objective(trial):
      """XGBoost Classification objective function for Optuna.

      Args:
          trial (optuna.trial.Trial): A trial object

      Returns:
            float: Accuracy of the model.
      """  

      # Hyperparameters we want to tune
      # Choosing binary:logistic because it is a binary-classification problem (above/equal to 50K or below 50K)
      param = {
      "objective": "binary:logistic",
      "tree_method": "exact",
      "eval_metric": "auc",
      "booster": trial.suggest_categorical("booster", ["gbtree", "gblinear", "dart"]),
      "lambda": trial.suggest_loguniform("lambda", 1e-8, 1.0),
      "alpha": trial.suggest_loguniform("alpha", 1e-8, 1.0),
      }

      # Setting up specific hyperparameters based off booster method chosen
      if param["booster"] == "gbtree" or param["booster"] == "dart":
            param["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            param["eta"] = trial.suggest_loguniform("eta", 1e-8, 1.0)
            param["gamma"] = trial.suggest_loguniform("gamma", 1e-8, 1.0)
            param["grow_policy"] = trial.suggest_categorical("grow_policy", ["depthwise", "lossguide"])
      if param["booster"] == "dart":
            param["sample_type"] = trial.suggest_categorical("sample_type", ["uniform", "weighted"])
            param["normalize_type"] = trial.suggest_categorical("normalize_type", ["tree", "forest"])
            param["rate_drop"] = trial.suggest_loguniform("rate_drop", 1e-8, 1.0)
            param["skip_drop"] = trial.suggest_loguniform("skip_drop", 1e-8, 1.0)


      # Train the model 
      bst = xgb.train(param,dtrain)

      # Make a prediction
      preds = bst.predict(dtest)
      pred_labels = np.rint(preds)

      # Return the accuracy of the prediction
      return metrics.accuracy_score(y_test, pred_labels)


# %%
# Create the Optuna study, we maximize the objective function (accuracy)
study = optuna.create_study(direction='maximize', sampler = TPESampler(seed=37), pruner=MedianPruner())
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
# best_params["Booster"] = best_params.pop("booster")

# Create the model with the best parameters
xgb_classifier = xgb.XGBClassifier(**best_params)

# Fit the model
xgb_classifier.fit(X_train, y_train)

# Predict the test set
y_pred = xgb_classifier.predict(X_test)

# Evaluate the model
train_acc = accuracy_score(y_train, xgb_classifier.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

# Get the time it took to run the model
end_time = time.time()
duration = end_time - start_time

# Print the results
print("Tuned Train accuracy: %f" % (train_acc))
print("Tuned Test accuracy: %f" % (test_acc))
print(f"Training duration: {duration:.2f} seconds")

# %%
# Starting timing of XGBoost Classification
start_time = time.time()

# Create the model using the default parameters
xgb_classifier = xgb.XGBClassifier()

# Fit the model
xgb_classifier.fit(X_train, y_train)

# Predict the test set
y_pred = xgb_classifier.predict(X_test)

# Evaluate the model
train_acc = accuracy_score(y_train, xgb_classifier.predict(X_train))
test_acc = accuracy_score(y_test, y_pred)

# Get time it took to run the model
end_time = time.time()
duration = end_time - start_time

# Print the results
print("Default Train accuracy: %f" % (train_acc))
print("Default Test accuracy: %f" % (test_acc))
print(f"Training duration: {duration:.2f} seconds")


