import os, sys, pathlib, yaml , json 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from dvclive import Live
from joblib import dump

params=yaml.safe_load(open('params.yaml'))
proc_dir=pathlib.Path(params['data']['processed_dir'])
model_dir=pathlib.Path(params['paths']['model_dir'])
metrics_dir=pathlib.Path(params['paths']['metrics_dir'])
model_dir.mkdir(parents=True, exist_ok=True)
metrics_dir.mkdir(parents=True, exist_ok=True)

train=pd.read_csv(proc_dir /'train.csv')
X=train.drop(columns=['species'])
y=train['species']

steps=[]
if params['features']['scale']:
    steps.append(('scale',StandardScaler()))

steps.append(('rf',RandomForestClassifier(
            n_estimators=params['train']['n_estimators'],
            random_state=params['train']['random_state'],
            max_depth=params['train']['max_depth']

        )))

pipe=Pipeline(steps)

with Live(dir=str(metrics_dir), save_dvc_exp=True) as live:
    pipe.fit(X,y)
    preds=pipe.predict(X)
    acc=accuracy_score(y, preds)
    live.log_metric('train_accuracy', float(acc))

dump(pipe, model_dir / "model.joblib")
