from azureml.core import Workspace
import json

ws = Workspace.from_config('./config/config.json')

webService = ws.webservices['titanic-survivors']
# sex','age','sibsp','parch','fare','embarked','body'
data = {"data" : {
    "sex": ["male"],
    "age": [25],
    "sibsp" : [0],
    "parch" : [0],
    "embarked" : ["S"],
    "fare" : [5],
    "body" : [0]
}}

json_data = json.dumps(data)

res = webService.run(json_data)

print(json.loads(res))