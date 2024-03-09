from azureml.core import Workspace, Experiment, Environment
from azureml.core.compute import AksCompute, ComputeTarget
from azuremlcore.model import Model, InferenceConfig
from azureml.core.webservice import AksWebservice

ws = Workspace.from_config('./config/config.json')

cluster_name = 'kubernetes_cluster'
if cluster_name not in ws.compute_targets:
    cluster_config = AksCompute.provisoning_cluster(vm_size='STANDARD_D11_V2',agent_count=2,cluster_purpose='DevTest',location='eastus')
    aks_cluster = ComputeTarget.create(ws,cluster_name,cluster_config)
    aks_cluster.wait_for_completion()
else:
    aks_cluster = ws.compute_targets[cluster_name]

infernece_config = InferenceConfig()