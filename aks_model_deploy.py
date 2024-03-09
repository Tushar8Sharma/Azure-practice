from azureml.core import Workspace, Environment, Experiment
from azureml.core.compute import AksCompute,ComputeTarget
from azureml.core.environment import CondaDependencies
from azureml.core.webservice import AksWebservice
from azureml.core.model import InferenceConfig, Model

ws = Workspace.from_config('./config/config.json')

cluster_name = 'aks-cluster'

if cluster_name not in ws.compute_targets:
    compute_config = AksCompute.provisioning_configuration(
        location='eastus', vm_size='STANDARD_D11_V2',agent_count = 1, cluster_purpose='DevTest')
    cluster = ComputeTarget.create(ws,cluster_name,compute_config)
    cluster.wait_for_completion()
else:
    cluster = ws.compute_targets[cluster_name]

env = Environment(workspace=ws,name= 'kubernetes_environment')
env_dep = CondaDependencies.create(conda_packages=['pip','pandas','scikit-learn','joblib'],pip_packages=['azureml-defaults'])
env.python.conda_dependencies = env_dep
env.register(ws)

inf_config = InferenceConfig(environment=env,source_directory = 'services',entry_script='scoring_model.py')

dep_config = AksWebservice.deploy_configuration(cpu_cores = 1, memory_gb=1)

model = ws.models['Titanic_RFC_model']

service = Model.deploy(
    workspace=ws,
    name = 'titanic-survivors',
    inference_config = inf_config,
    deployment_config = dep_config,
    deployment_target = cluster,
    models = [model]
)

service.wait_for_deployment(show_output = True)