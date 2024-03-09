from azureml.core import Workspace, Dataset, Datastore
from sklearn.preprocessing import LabelEncoder

ws = Workspace.from_config()

titanic_dataset = Dataset.get_by_name(ws,'titanic dataset')

df = titanic_dataset.to_pandas_dataframe()

# print(df.isnull().sum())

for col in df.columns:
    if 'obj' in str(df[col].dtypes):
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
    else:
        df[col] = df[col].fillna(df[col].mean())
# print('----')
# print(df.isnull().sum())
# print(df.dtypes)

df.drop(['home.dest','boat'],axis=1,inplace=True)

# df.to_csv('new_titanic_dataset.csv')

# print(df.head())
az_store = Datastore.get(ws,'practicedatastore')
# Dataset.Tabular.register_pandas_dataframe(dataframe = df, target = az_store,name = 'modified titanic dataset')


# az_store.upload_files(['new_titanic_dataset.csv'],overwrite=True)
az_store.upload('./data/',overwrite=True)