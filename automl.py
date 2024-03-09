from azureml.core import Workspace, Dataset, Datastore, Environment, Experiment
from azureml.train.automl import AutoMLConfig
from azureml.core.compute import AmlCompute, ComputeTarget

ws = Workspace.from_config(path = './config/config.json')

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

env = Environment(workspace = ws,name = 'custom_environment')
# env_dep = CondaDependencies.create(conda_packages=['scikit-learn','pip','pandas'],pip_packages=['azureml-defaults'])
# env.python.conda_dependencies = env_dep

auto_ml = AutoMLConfig(
    task = 'classification',
    compute_target = cluster_name,
    training_data = Dataset.get_by_name(ws,'new titanic dataset'),
    label_column_name = 'survived',
    validation_size = 0.3,
    primary_metric = 'accuracy',
    iterations = 18,
    max_concurrent_iterations = 2,
    iteration_timeout_minutes = 20
)

exp = Experiment(workspace = ws,name='automl_practice_01')

run = exp.submit(auto_ml)

run.wait_for_completion(show_output=True)

print(run.get_best_child())

print(run.get_children())