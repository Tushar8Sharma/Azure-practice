from azureml.core import Workspace, Dataset, Datastore, Experiment, Environment, ScriptRunConfig
from azureml.core.environment import CondaDependencies
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.core.compute import AmlCompute,ComputeTarget
from azureml.pipeline.steps import PythonScriptStep
from azureml.core.runconfig import RunConfiguration

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


input_ds = Dataset.get_by_name(ws,'new titanic dataset')
dataFolder = PipelineData('dataFolder',datastore = ws.get_default_datastore())
run_config = RunConfiguration()
run_config.environment = env
run_config.target = az_cluster

dataPrep01 = PythonScriptStep(
    name = '01_Data_preparation',
    source_directory = 'training',
    script_name = 'Dataprep_pipeline.py',
    inputs = [input_ds.as_named_input('raw_data')],
    outputs = [dataFolder],
    arguments = ['--dataFolder',dataFolder],
    runconfig = run_config
)

trainModel02 = PythonScriptStep(
    name = '02_training_model',
    source_directory = 'training',
    script_name = 'training_model.py',
    inputs = [dataFolder],
    runconfig = run_config,
    arguments = ['--dataFolder',dataFolder]
)

pipe_list = [dataPrep01,trainModel02]

pipe_config = Pipeline(workspace = ws, steps = pipe_list)

exp = Experiment(ws,'practice_exp02')

run = exp.submit(pipe_config)

run.wait_for_completion(show_output = True)