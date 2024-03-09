from azureml.core import Workspace,Dataset,Environment,Experiment
from azureml.interpret import ExplanationClient
from interpret.ext.blackbox import TabularExplainer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

ws = Workspace.from_config(path='./config/config.json')

df = Dataset.get_by_name(workspace=ws,name = 'new titanic dataset').to_pandas_dataframe()

exp = Experiment(workspace=ws, name = 'exp01')

run = exp.start_logging(snapshot_directory=None)

for col in df.columns:
    print(df[col].dtypes)
    df[col]=df[col].fillna(0)

X,y = df[['sex','age','sibsp','parch','fare','embarked','body']],df['survived']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=df['survived'])

rfc = RandomForestClassifier()

rfc.fit(X_train,y_train)
pred = rfc.predict(X_test)

print(accuracy_score(pred,y_test))

path = 'models/random.pkl'
joblib.dump(rfc,path)

features = X_train.columns
classes = ['lived','died']

tab_explainer = TabularExplainer(rfc,X_train,features = features,classes = classes)
global_explain = tab_explainer.explain_global(X_train)

print(global_explain.get_feature_importance_dict())

local_explain = tab_explainer.explain_local(X_train.head())
print(local_explain.get_ranked_local_names())
print(local_explain.get_ranked_local_values())

run.register_model(
    model_name = 'Random_forest_classifier',
    model_path = path,
    # tags = [],
    descriptions = 'Contain classifier for titanic model'
)

run.complete()

explanation_client = ExplanationClient.from_run(run)
explanation_client.upload_model_explanation(global_explain,comment='This is a practice explainer')


new_run = ws.get_run('exp01')
new_explanation_client = ExplanationClient.from_run(new_run)
explainer = new_explanation_client.download_model_explanation()