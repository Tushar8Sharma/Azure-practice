from azureml.core import Workspace, Datastore, Dataset

ws = Workspace.from_config(path = './config/config.json')

az_store = Datastore.get(ws,'customdatastore')

dataset_path = [(az_store,'titanic1.csv')]

titanic_dataset = Dataset.Tabular.from_delimited_files(dataset_path)

titanic_dataset.register(ws, 'titanic dataset', create_new_version=True)