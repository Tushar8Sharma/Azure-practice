from azureml.core import Workspace, Datastore, Dataset
from sklearn.preprocessing import LabelEncoder

ws = Workspace.from_config(path = './config/config.json')

az_store = Datastore.get(ws,'customdatastore')
titanic = Dataset.get_by_name(ws,'titanic dataset')
df = Dataset.get_by_name(ws,'titanic dataset').to_pandas_dataframe()


for col in df.columns:
    if 'obj' in str(df[col].dtypes):
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
    else:
        df[col] = df[col].fillna(df[col].mean())

df.drop(['home.dest','boat'],axis=1,inplace=True)

df.to_csv('./data/new.csv')

titanic.Tabular.register_pandas_dataframe(
        dataframe = df,
        target = az_store,
        name = 'new titanic dataset',
        description = 'preprocessed dataset'
)

files = ['./data/new.csv']

# az_store.upload_files([files],target_path='newTitanic',overwrite=True)

az_store.upload('./data',overwrite = True)