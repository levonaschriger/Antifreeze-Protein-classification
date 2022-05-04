import numpy as np
import pandas as pd
import xgboost
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

#loading data:\n
train_data = pd.read_csv('train.csv',low_memory=False)
test_data = pd.read_csv('test.csv',low_memory=False)

#removing the labels from the features:
y_train = train_data["label"]
train_data.drop(labels="label", axis=1, inplace=True)

# create a concatenated new data set:
full_data = train_data.append(test_data)

#drop columns that aren't necessary:
drop_columns = ["Feature0"]
full_data.drop(labels=drop_columns, axis=1, inplace=True)

#str type to int
full_data = pd.get_dummies(full_data, columns=["Id"])
full_data.fillna(value=0.0, inplace=True)

#split the data into training and testing sets:
X_train = full_data.values[0:3884]
X_test = full_data.values[3884:]\
 
#scale our data by creating an instance of the scaler and scaling it:
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
  
#split the data into training and testing sets\
state = 12 
test_size = 0.40
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=test_size, random_state=state)

XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                importance_type='gain', interaction_constraints='',
                learning_rate=0.300000012, max_delta_step=0, max_depth=6,
                min_child_weight=1, missing=nan, monotone_constraints='()',
                n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
                tree_method='exact', validate_parameters=1, verbosity=None)

xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)

score = xgb_clf.score(X_val, y_val)
print(score)
y_pred = xgb_clf.predict_proba(X_test) 
