#create hyper drive training script

#import libraries
from azureml.core import Run
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder

run = Run.get_context()
ws = run.experiment.workspace

#get the argument data
parser = argparse.ArgumentParser()
parser.add_argument('--input_data',type=str)
parser.add_argument('--n_estimators',type=int)
parser.add_argument('--min_samples_leaf',type=int)

args = parser.parse_args()

ne = args.n_estimators
msi = args.min_samples_leaf

titanic = run.input_datasets['raw_data'].to_pandas_dataframe()

label = LabelEncoder()
titanic['sex'] = label.fit_transform(titanic['sex'])
titanic['embarked'] = label.fit_transform(titanic['embarked'])

#filling nan values
titanic['body'] = titanic.body.fillna('0')
titanic['boat'] = titanic.boat.fillna('0')
titanic['age'] = titanic.age.fillna(titanic.age.mean())
# titanic['fare'] = titanic.fare.fillna(titanic.fare.mean())

#making boat column a integer type
boatDict = {'A' : 17,'B' : 18, 'C' : 19, 'D' : 20, '5 9' : 21,'5 7' : 22, '8 10' : 23,'13 15 B':24, 'C D': 25, '15 16':26,'13 15' : 27}

for index,rows in titanic.iterrows():
    try:
        titanic.loc[index,'body'] = int(rows['body'])
        titanic.loc[index,'boat'] = int(boatDict[rows['boat']])
    except:
        titanic.loc[index,'body'] = int(rows['boat'])
        pass


# Create X and Y - Similar to "edit columns" in Train Module
Y = titanic[['survived']]
X = titanic.drop(['survived','Column1','name','home.dest'], axis=1)


# Split Data - X and Y datasets are training and testing sets

X_train, X_test, Y_train, Y_test = \
train_test_split(X, Y, test_size = 0.3, random_state = 1234, stratify=Y)


# Build the Random Forest model

rfc = RFC(n_estimators=ne, min_samples_leaf=msi)


# Fit the data to the Random Forest object - Train Model
rfc.fit(X_train, Y_train)


# Predict the outcome using Test data - Score Model 
# Scored Label
Y_predict = rfc.predict(X_test)

# Get the probability score - Scored Probabilities
Y_prob = rfc.predict_proba(X_test)[:, 1]

# Get Confusion matrix and the accuracy/score - Evaluate
cm    = confusion_matrix(Y_test, Y_predict)
score = rfc.score(X_test, Y_test)


# Always log the primary metric
run.log("accuracy", score)

run.complete()






