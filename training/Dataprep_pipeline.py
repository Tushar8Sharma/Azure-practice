from azureml.core import Run,Dataset
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder
from argparse import ArgumentParser as AP

#get the context of experiment run
run = Run.get_context()
ws = run.experiment.workspace #best practice to use the specific workspace not some other workspace if you have multiple
titanic = run.input_datasets['raw_data'].to_pandas_dataframe()

#Load the titanic_dataset
print('loading......')
# titanic = pd.read_csv('titanic.csv').drop('Unnamed: 0',axis=1)
# titanic = Dataset.get_by_name(ws,'titanic survivor predictor using SDK').to_pandas_dataframe()

#preprocessing the data set
# label = LabelEncoder()
# titanic['sex'] = label.fit_transform(titanic['sex'])
# titanic['embarked'] = label.fit_transform(titanic['embarked'])

# #filling nan values
# titanic['body'] = titanic.body.fillna('0')
# titanic['boat'] = titanic.boat.fillna('0')
# titanic['age'] = titanic.age.fillna(titanic.age.mean())
# titanic['fare'] = titanic.fare.fillna(titanic.fare.mean())

# #making boat column a integer type
# boatDict = {'A' : 17,'B' : 18, 'C' : 19, 'D' : 20, '5 9' : 21,'5 7' : 22, '8 10' : 23,'13 15 B':24, 'C D': 25, '15 16':26,'13 15' : 27}

# for index,rows in titanic.iterrows():
#     try:
#         titanic.loc[index,'body'] = int(rows['body'])
#         titanic.loc[index,'boat'] = int(boatDict[rows['boat']])
#     except:
#         titanic.loc[index,'body'] = int(rows['boat'])
#         pass

#get the arguments from pipeline job

parser = AP()
parser.add_argument('--dataFolder',type=str)
args = parser.parse_args()

#create the folder is it does not exist
os.makedirs(args.dataFolder,exist_ok=True)
path = os.path.join(args.dataFolder,'defaults_prep.csv')
titanic.to_csv(path)

run.complete()