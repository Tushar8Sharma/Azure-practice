from azureml.core import Run,Dataset
import pandas as pd
import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from argparse import ArgumentParser as AP

#get the context of experiment run
run = Run.get_context()
ws = run.experiment.workspace #best practice to use the specific workspace not some other workspace if you have multiple

#get the arguments from pipeline job

parser = AP()
parser.add_argument('--dataFolder',type=str)
args = parser.parse_args()

#get the data
path = os.path.join(args.dataFolder,'defaults_prep.csv')
titanic = pd.read_csv(path)

#Separating independent variable and target variable
X,y = titanic[['sex','age','sibsp','parch','fare','embarked','body']],titanic['survived']
# Split data into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)



# train the model
model = LogisticRegression()
model.fit(X_train,y_train)

# calculate accuracy
y_hat = model.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))

# calculate AUC
y_scores = model.predict_proba(X_test)
auc = roc_auc_score(y_test,y_scores[:,1])
print('AUC: ' + str(auc))
run.log('AUC', np.float(auc))

#calculate confusion_matirx
cm = confusion_matrix(y_test,y_hat)
#confusion_matrix can't be logged like a regular matrix. So, azure has a special way to do it.
#first create dictionary
cm_dict = {"schema_type": "confusion_matrix",
            "schema_version" : "v1",
            "data": {"class_labels":['0','1'],
                     "matrix":cm.tolist()}
            }
#now log confusion matrix with special method
run.log_confusion_matrix("confusionMatrix",cm_dict)

# Save the trained model in the outputs folder
os.makedirs('outputs', exist_ok=True)
joblib.dump(value=model, filename='outputs/titanic_model.pkl')

run.complete()