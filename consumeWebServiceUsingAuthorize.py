import requests
import json

primary_key = 'LfvqqPZsMUFvwMgU35GXClCsnc55HgiQ'
url =  'http://40.88.204.153:80/api/v1/service/titanic-survivors/score'

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

header = {'Content-Type':'application/json','Authorization':f'Bearer {primary_key}'}
resp = requests.post(url,json_data,headers=header)

print(json.loads(resp.json()))