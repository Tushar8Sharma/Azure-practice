import pandas as pd
from azureml.core import Run
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from azureml.widgets import RunDetails

run = Run.get_context()

dataset = run.input_datasets['raw_data']

df = dataset.to_pandas_dataframe()

print(df.head())

model = LogisticRegression()

X_train,X_test,y_train,y_test = train_test_split(df.drop(['Column1','survived'],axis=1),df['survived'],test_size=0.3,stratify=df['survived'])
model = model.fit(X_train,y_train)
y_pred = model.predict(X_test)

score = accuracy_score(y_pred,y_test)

run.log('accuracy',score)

run.complete()