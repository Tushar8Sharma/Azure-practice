from azureml.core import Workspace, Datastore

ws = Workspace.from_config(path='./config/config.json')

# ws.write_config()

az_store = Datastore.register_azure_blob_container(
    workspace = ws,
    datastore_name = 'customdatastore',
    account_name = 'practicedatastore',
    container_name = 'practice-container',
    account_key = 'n/vodJQs/N56FTp+MlApRVY1ocoFPlN86ZuQPcERJ+ZrPcp3RV8pzImprar3JQM7jcZVHf5+j55W+AStzpnK1A=='
)