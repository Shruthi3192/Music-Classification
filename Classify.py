from google.colab import drive
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pandas import DataFrame
import seaborn as sns
from sklearn import tree
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
import sklearn.model_selection as skl_ms
import sklearn.preprocessing as preprocess
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
from sklearn.metrics import classification_report, confusion_matrix, plot_confusion_matrix,recall_score ,f1_score,make_scorer , accuracy_score , precision_score

def MountDrive():
  drive.mount('/content/drive')
  %cd '/content/drive/MyDrive/'

def CreateClassifier():
  
  model_randomforest=RandomForestClassifier(n_estimators=400,min_samples_split=5,max_features='sqrt',max_depth=90,min_samples_leaf=1)
  return model_randomforest

def VisualiseTrainingData(model):
  np.random.seed(1)
  song_data=pd.read_csv("training_data.csv")
  song_data.describe()
  song_data.hist()
  plt.show()
  return song_data

def GridSearch()
  
  # Number of trees in random forest
  n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
  # Number of features to consider at every split
  max_features = ['auto', 'sqrt']
  # Maximum number of levels in tree
  max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
  max_depth.append(None)
  # Minimum number of samples required to split a node
  min_samples_split = [2,5,10]
  # Minimum number of samples required at each leaf node
  min_samples_leaf = [1,2,4]
  # Method of selecting samples for training each tree
  bootstrap = [True, False]
  # Create the random grid
  random_grid = {'n_estimators': n_estimators,
                'max_features': max_features,
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf,
                'bootstrap': bootstrap}
  pprint(random_grid)
  # First create the base model to tune
  rf = RandomForestClassifier()
  # Random search of parameters, using 3 fold cross validation, 
  # search across 100 different combinations, and use all available cores
  rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
  # Fit the random search model
  song_data=pd.read_csv("training_data.csv")
  train_features=song_data.drop(columns=['label'])
  train_labels=song_data['label']
  rf_random.fit(train_features, train_labels)
  rf_random.best_params_


def CheckFeatureImportance(data_set,model):
  %matplotlib inline
  # Creating a bar plot
  feature_imp = pd.Series(model.feature_importances_,index=['acousticness','danceability','duration','energy','instrumentalness','key','liveness','loudness','mode','speechiness','tempo','time_signature','valence']).sort_values(ascending=False)
  feature_imp
  sns.barplot(x=feature_imp, y=feature_imp.index)
  #Add labels to your graph
  plt.xlabel('Feature Importance Score')
  plt.ylabel('Features')
  plt.title("Visualizing Important Features")
  plt.legend()
  plt.show()
  

def DataPreProcessing(data_set,train=True):
  scaler=preprocess.MinMaxScaler()
  if train==True:
    train_set=data_set.drop(columns=['label'])
    train_label=data_set['label']
    #train_set = scaler.fit_transform(train_set)
    return train_set,train_label
  else:
    test_set=data_set.drop(columns=[])
    #test_set = scaler.fit_transform(test_set)
    return test_set

def SplitData(data_set,data_label):
  X_train, X_test, y_train, y_test = skl_ms.train_test_split(data_set, data_label, test_size=0.25, random_state=1, shuffle = False)
  return X_train,X_test,y_train,y_test


def ApplyModel(model,X_train,X_test,y_train,y_test):
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  print(confusion_matrix(y_test,y_pred))
  print(classification_report(y_test,y_pred))
  print(accuracy_score(y_test, y_pred))
  plot_confusion_matrix(model, X_test, y_test,cmap=plt.cm.Blues)
  plt.show()
  return model

def KfoldValidation(model,train_set,train_label):
  cv=skl_ms.RepeatedKFold(n_splits=7,random_state=4,n_repeats=10)
  accuracyList = []
  model.fit(train_set,train_label)
  scoring = {'accuracy' : make_scorer(accuracy_score),'precision' : make_scorer(precision_score),'recall' : make_scorer(recall_score),'f1_score' : make_scorer(f1_score)}
  score=skl_ms.cross_validate(model,train_set,train_label,scoring=scoring,cv=cv,n_jobs=-1)
  print('Accuracy : %.3f' % (np.mean(score['test_accuracy'])))
  print('Precision : %.3f' % (np.mean(score['test_precision'])))
  print('Recall : %.3f' % (np.mean(score['test_recall'])))
  print('F1  Score : %.3f' % (np.mean(score['test_f1_score'])))
  return model
 
def PrepareTestData():
  Test_song_data=pd.read_csv("songs_to_classify.csv")
  test_set=DataPreProcessing(Test_song_data,False)
  return test_set

def Predict(data_set,model):
  prediction=model.predict(data_set)
  submission = ''. join (str (e) for e in prediction)
  print (submission)

def StartHere():
  #def GridSearch()
  MountDrive()
  model=CreateClassifier()
  trainingData=VisualiseTrainingData(model)
  preProcessData,trainingLabel=DataPreProcessing(trainingData,True)
  trainX,testX,trainY,testY=SplitData(preProcessData,trainingLabel)
  #regular train test split
  #trainedModel=ApplyModel(model,trainX,testX,trainY,testY)

  #using kfold for validation
  trainedModel=KfoldValidation(model,preProcessData,trainingLabel)

  #used to visualise the important features
  #CheckFeatureImportance(preProcessData,trainedModel)

  unseenTestdata=PrepareTestData()
  Predict(unseenTestdata,trainedModel)


if __name__ == '__main__':
  StartHere()
