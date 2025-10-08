import os , sys, shutil, urllib.request, yaml, pathlib
params=yaml.safe_load(open('params.yaml'))
rawdir=pathlib.Path(params['data']['raw_dir'])
rawdir.mkdir(parents=True, exist_ok=True)


url="https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
dst=rawdir/'iris.csv'
urllib.request.urlretrieve(url, dst.as_posix())
print(f"Downloaded -> {dst}")