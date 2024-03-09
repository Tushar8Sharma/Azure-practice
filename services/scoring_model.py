from azureml.core import Workspace, Model
import joblib
import json
import pandas as pd

def init():
    global refcol,model,encoders
    refcol,model,encoders = joblib.load(Model.get_model_path('Titanic_RFC_model'))

def run(data):
    data_dict = json.loads(data)['data']
    df = pd.DataFrame.from_dict(data_dict)
    df = df[refcol]
    i = 0
    for col in df.columns:
        if 'obj' in str(df[col].dtypes):
            df[col] = encoders[i].transform(df[col])
            i += 1
    score = list(model.predict(df))
    if score[0] == 0:
        return json.dumps('dead')
    else:
        return json.dumps('survived')