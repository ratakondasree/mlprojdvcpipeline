import os, pandas as pd , pathlib, yaml , json
from  sklearn.model_selection import train_test_split

params=yaml.safe_load(open('params.yaml'))
rawdir=pathlib.Path(params['data']['raw_dir'])
processeddir=pathlib.Path(params['data']['processed_dir'])
processeddir.mkdir(parents=True, exist_ok=True)

df=pd.read_csv(rawdir/'iris.csv')
train_ratio=params['data']['train_ratio']

train_df, test_df=train_test_split(df, train_size=train_ratio, shuffle=True, stratify=df['species'], random_state=42)

train_df.to_csv(processeddir/'train.csv', index=False)
test_df.to_csv(processeddir/'test.csv', index=False)

print('Preprocess done', len(train_df), len(test_df))
