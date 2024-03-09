from azureml.core import Workspace, Dataset, ScriptRunConfig, Experiment, Environment
from azureml.core.environment import CondaDependencies
from azureml.core.compute import AmlCompute,ComputeTarget

ws = Workspace.from_config('./config/config.json')

env = Environment(workspace = ws,name = 'custom_environment')
env_dep = CondaDependencies.create(conda_packages=['scikit-learn','pip','pandas'],pip_packages=['azureml-defaults'])
env.python.conda_dependencies = env_dep

cluster_name = 'practiceCluster'
if cluster_name not in ws.compute_targets:
    cluster_config = AmlCompute.provisioning_configuration(
        vm_size = 'STANDARD_D11_V2',
        max_nodes = 2
    )
    az_cluster = ComputeTarget.create(ws,cluster_name,cluster_config)

    az_cluster.wait_for_completion()
else:
    az_cluster = ws.compute_targets[cluster_name]

run_config = ScriptRunConfig(
    arguments = ['--input_data',Dataset.get_by_name(ws,'new titanic dataset').as_named_input('raw_data')],
    source_directory='training',
    script = 'training.py',
    compute_target = az_cluster,
    environment = env
)


exp = Experiment(ws,'practice_exp02')

# run = exp.start_logging(snapshot_directory=None)
run = exp.submit(run_config)

run.log('sample','successfull')

run.wait_for_completion(show_output=True)