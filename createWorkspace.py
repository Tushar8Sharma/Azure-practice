from azure.ai.ml import MLClient
from azure.ai.ml.entities import Workspace
from azure.identity import DefaultAzureCredential

subscription_id = 'f70adb95-4144-4799-9bc6-d5fb84c6e7ec'
resource_name = 'Practice'
mlclient = MLClient(DefaultAzureCredential(),subscription_id,resource_name)

ws = Workspace(
    name = 'practice-sdk01',
    location = 'eastus',
    description = 'For practicing workspace',
    hbi_workspace=False
)

mlclient.workspaces.begin_create(ws)