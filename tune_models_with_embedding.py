from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
import lightgbm as lgb
from scipy.stats import randint
import xgboost as xgb
import math
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
from utils import evaluate_model

import argparse
import sys
parser = argparse.ArgumentParser()
parser.add_argument('--setting', default=3, required=True, type=int, help='dataset setting')

args = parser.parse_args()

if (args.setting > 3) or (args.setting < 2):
    sys.exit('Only setting 2(II) and 3(III) are implemented in this code.')

df_training = pd.read_pickle(f'./saved_files/train_setting_{args.setting}.pkl')
df_test = pd.read_pickle(f'./saved_files/eval_setting_{args.setting}.pkl')
df_training = df_training.fillna(-1)
df_test = df_test.fillna(-1)

X = df_training.drop(columns=['score'])
y = df_training['score']

X_test = df_test.drop(columns=['score'])
y_test = df_test['score']


max_feature_1 = math.ceil(math.sqrt(len(X.columns)))
max_feature_2 = math.ceil(math.log2(len(X.columns)))

# Define the parameter grid for each model
param_grid_rf = {
    "n_estimators": randint(60, 1000),
    "max_depth": randint(3, 21),
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 11),
    "max_features": [max_feature_1, max_feature_2, None],
    "bootstrap": [True, False],
    "random_state": [0]
}

param_grid_lgbm = {
    "n_estimators": randint(60, 1000),
    "max_depth": randint(3, 21),
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.5, 1],
    "num_leaves": randint(10, 100),
    "reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100],
    "reg_lambda": [1e-5, 1e-2, 0.1, 1, 10, 100],
    "random_state": [0]
}

param_grid_et = {
    "n_estimators": randint(60, 1000),
    "max_depth": randint(3, 21),
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 11),
    "max_features": [max_feature_1, max_feature_2, None],
    "bootstrap": [True, False],
    "random_state": [0]
}

param_grid_gbc = {
    "n_estimators": randint(60, 1000),
    "max_depth": randint(3, 21),
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.5, 1],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 11),
    "max_features": [max_feature_1, max_feature_2, None],
    "random_state": [0]
}

xgb_grid = {
    # Learning rate shrinks the weights to make the boosting process more conservative
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.5, 1],
    # Maximum depth of the tree, increasing it increases the model complexity.
    "max_depth": range(3, 21, 2),
    # Gamma specifies the minimum loss reduction required to make a split.
    "gamma": [i / 10.0 for i in range(0, 5)],
    # Percentage of columns to be randomly samples for each tree.
    "colsample_bytree": [i / 10.0 for i in range(3, 10)],
    # reg_alpha provides l1 regularization to the weight, higher values result in more conservative models
    "reg_alpha": [1e-5, 1e-2, 0.1, 1, 10, 100],
    # reg_lambda provides l2 regularization to the weight, higher values result in more conservative models
    "reg_lambda": [1e-5, 1e-2, 0.1, 1, 10, 100], 'n_estimators': range(60, 1000, 50), }

# Set up the models
rf_model = RandomForestClassifier()
lgbm_model = lgb.LGBMClassifier()
et_model = ExtraTreesClassifier()
gbc_model = GradientBoostingClassifier()
xgb_model = xgb.XGBClassifier()

scaler = StandardScaler()
# Fit the scaler on the data and transform it
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Set up the parameter grid for random search
param_grid = {
    'rf': param_grid_rf,
    "lgbm": param_grid_lgbm,
    "et": param_grid_et,
    "gbc": param_grid_gbc,
    "xgb": xgb_grid,
}

all_test_probs = {}

for model_name, model in zip(['rf', 'lgbm', 'et', 'gbc', "xgb"],
                             [rf_model, lgbm_model, et_model, gbc_model, xgb_model]):
    search = RandomizedSearchCV(model, param_grid[model_name], scoring="roc_auc", cv=5, n_iter=30, random_state=0,
                                n_jobs=-1)
    search.fit(X_scaled, y)
    best_random = search.best_estimator_
    # best_params[model_name] = best_random.get_params()

    pickle.dump(best_random, open(f'./models/{model_name}_classifier_setting_{args.setting}.sav', 'wb'))
    print(f"Results for {model_name}:")
    print(f"Best params: {best_random.get_params()}")

    test_probs = model.predict_proba(X_test_scaled)

    all_test_probs[model_name] = test_probs[:, 1]

    test_predictions = model.predict(X_test_scaled)

    evaluate_model(y_test, test_predictions, test_probs[:, 1])

all_test_probs_arr = np.array(list(all_test_probs.values()))
test_mean_probs = all_test_probs_arr.mean(axis=0)

print('Results for Ensemble Model:')

test_predictions = (test_mean_probs >= 0.5).astype(int)

evaluate_model(y_test, test_predictions, test_mean_probs)
