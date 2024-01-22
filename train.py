# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import pickle

#Parameters
C = 0.01
n_splits = 5
output_file = f'model_C={C}.bin'

#Data Preparation

df = pd.read_csv('capstone2.csv')
df.head().T

df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')

diabetes = {False: 0, True: 1}
df['diabetes'] = df['diabetes'].replace(diabetes)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
y_test = df_test.diabetes.values

categorical = ['gender', 'smoking_history']

numerical = [
    'age',
    'hypertension',
    'heart_disease',
    'bmi',
    'hba1c_level',
    'blood_glucose_level',
    'diabetes'
]
#Training

def train(df_train, y_train, C=0.01):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    # Create a StandardScaler and scale the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model

def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred

print(f'doing validation C = {C}')

kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

scores = []
fold = 0

for train_idx, val_idx in kfold.split(df_full_train):
    df_train = df_full_train.iloc[train_idx]
    df_val = df_full_train.iloc[val_idx]

    y_train = df_train.diabetes.values
    y_val = df_val.diabetes.values

    dv, model = train(df_train, y_train, C=C)
    y_pred = predict(df_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f'auc on fold {fold} is {auc}')
    fold = fold + 1

print('Validation results:')
print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))

scores

#Final model
print('Training the final model')

dv, model = train(df_full_train, df_full_train.diabetes.values, C=0.01)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc

print(f'auc:{auc}')


f_out = open(output_file, 'wb')
pickle.dump((dv, model), f_out)
f_out.close()


with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'The model is save to {output_file}')