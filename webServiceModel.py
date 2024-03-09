from azureml.core import Workspace, Experiment, Environment, Dataset
from sklearn.ensemble import RandomForestClassifier
from azureml.core.environment import CondaDependencies
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

ws = Workspace.from_config(path = './config/config.json')

env = Environment(workspace = ws,name = 'custom_environment')
env_dep = CondaDependencies.create(conda_packages=['pip','pandas','joblib','scikit-learn'],pip_packages=['azureml-defaults'])
env.python.conda_dependencies = env_dep
env.register(ws)

exp = Experiment(workspace = ws,name = 'kubernetes_exp_02')
run = exp.start_logging(snapshot_directory=None)

df = Dataset.get_by_name(ws,'titanic dataset').to_pandas_dataframe()

print(df.columns)

df = df[['sex','age','sibsp','parch','fare','embarked','body','survived']]
encode = []
for col in df.columns:
    if 'obj' in str(df[col].dtypes):
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encode.append(encoder)
    else:
        df[col] = df[col].fillna(df[col].mean())
    df[col] = df[col].fillna(0)

X,y = df[['sex','age','sibsp','parch','fare','embarked','body']],df['survived']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=df['survived'])

rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)
pred = rfc.predict(X_test)

col = X_train.columns
model = rfc
encoders = encode

joblib.dump(value=[col,model,encoders],filename='outputs/classifier1.pkl')#always save your model in outputs folder

run.complete()

new_run = ws.get_run(list(exp.get_runs())[0].id)

new_run.register_model(
    model_path='outputs/classifier1.pkl',
    model_name='Titanic_RFC_model',
    tags={'source':'sdk_run','algorithm':'RandomForest'},
    properties={'Accuracy':'checking'},
    description='Combined Models from the run'
)