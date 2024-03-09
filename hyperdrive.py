from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig, Dataset
from azureml.train.hyperdrive import GridParameterSampling, choice, HyperDriveConfig, PrimaryMetricGoal
from azureml.core.environment import CondaDependencies

ws = Workspace.from_config(path = './config/config.json')

env = Environment(Workspace=ws, name='custom_hyperdrive_environment')
env_dep = CondaDependencies.create(conda_packages=['pip','scikit-learn','pandas','joblib'],pip_packages=['azureml-defaults'])
env.python.conda_dependencies = env_dep

env.register(ws)

parameters = GridParameterSampling({
    '--n_estimators':choice(50,100,200),
    '--min_samples_leaf':choice(2,4,8)
})

az_cluster = ws.compute_targets['practiceCluster']

run_config = ScriptRunConfig(
    arguments = ['--input_data',Dataset.get_by_name(ws,'titanic dataset').as_named_input('raw_data')],
    source_directory = 'training',
    script = 'hyperdrive_training_script.py',
    compute_target = az_cluster,
    environment = env
)

hyper_config = HyperDriveConfig(
    hyperparameter_sampling=parameters,
    run_config = run_config,
    primary_metric_name='accuracy',
    primary_metric_goal = PrimaryMetricGoal.MAXIMIZE,
    max_total_runs = 18,
    max_concurrent_runs = 2,
    max_duration_minutes = 20,
    policy=None
)

exp = Experiment(workspace=ws, name='hyperdrive_practice01')
run = exp.submit(hyper_config)

run.wait_for_completion(show_output=True)