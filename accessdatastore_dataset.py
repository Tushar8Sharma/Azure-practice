from azureml.core import Workspace, Datastore, Dataset

ws = Workspace.from_config(path = './config/config.json')

print(ws.datastores)
print(ws.datasets.keys())