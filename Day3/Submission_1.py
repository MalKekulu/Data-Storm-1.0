#for numarical operations
import numpy as np
#for ploting
import matplotlib.pyplot as plt
#for importing and managing data sets
import pandas as pd


#for prepare X and y
dataset=pd.read_csv('credit_card_default_train.csv')
#preprocessing balance limits
dataset['Balance_Limit_V1'] = dataset['Balance_Limit_V1'].str.replace('M','000000')
dataset['Balance_Limit_V1'] = dataset['Balance_Limit_V1'].str.replace('K','000')
dataset['Balance_Limit_V1'] = dataset['Balance_Limit_V1'].astype(float)

X = dataset.iloc[:, 1:24]
y = dataset.iloc[:, 24].values



#Get the corelation matrix
corr=dataset.corr()

#Finiding missing data
missing_data=dataset[dataset.isnull().any(axis=1)]

# Encoding categorical data for training set
X = pd.get_dummies( X,columns =['Gender','EDUCATION_STATUS','MARITAL_STATUS','AGE','PAY_JULY','PAY_AUG','PAY_SEP','PAY_OCT','PAY_NOV','PAY_DEC'] )

X.insert(71, 'PAY_NOV_1', np.zeros(shape=(len(X),1)))
X.insert(82, 'PAY_DEC_1', np.zeros(shape=(len(X),1)))


cols_to_drop = [ 'Gender_F','EDUCATION_STATUS_Graduate','MARITAL_STATUS_Other','AGE_31-45','PAY_JULY_-2','PAY_AUG_-2','PAY_SEP_-2','PAY_OCT_-2','PAY_NOV_-2','PAY_DEC_-2' ]
X = X.drop( cols_to_drop, axis = 1 )

X=X.iloc[:,:].values

# Splitting the data into training and Testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=0)
    
#Feature scalling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

#Fitting the data set using XGboost
'''from xgboost import XGBClassifier
classifier_Xgboost=XGBClassifier(n_estimators=130, max_depth=5,learning_rate=0.1, colsample_bytree=0.8)   #polsample 0.8 
classifier_Xgboost.fit(X_train, y_train)'''

#Fitting data using Catboost
from catboost import CatBoostRegressor
classifier_CatBoost=CatBoostRegressor(iterations=100, depth=5, learning_rate=0.1)
classifier_CatBoost.fit(X_train, y_train)
    
#predict the test set
y_pred=classifier_CatBoost.predict(X_test)

y_pred=(y_pred>0.5)


#Fitting the Training set (Logistic Regression)
from sklearn.linear_model import LogisticRegression
classifier_Logistic=LogisticRegression(random_state=0)
classifier_Logistic.fit(X_train,y_train)

#Fitting the data set using XGboost
from xgboost import XGBClassifier
classifier_Xgboost=XGBClassifier(n_estimators=270, max_depth=4,learning_rate=0.1, colsample_bytree=0.8)   #polsample 0.8  n_estimator = 350
classifier_Xgboost.fit(X_train, y_train)

#Fitting whole test (Random Forest)
from sklearn.ensemble import RandomForestClassifier
classifier_forest=RandomForestClassifier(n_estimators=10, n_jobs=2, criterion='entropy',random_state=0)
classifier_forest.fit(X_train,y_train)

#make the network
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU

#Initializing the ANN
classifier=Sequential()

#Adding input layer and the first hidden layer
classifier.add(Dense(output_dim=40, init= 'uniform',activation='relu', input_dim=80))

#adding the second hidden layer
classifier.add(Dense(output_dim=20 , init= 'uniform',activation='relu'))

#adding the third hidden layer
#classifier.add(Dense(output_dim=10 , init= 'uniform',activation='relu'))

#Addning the output layer
classifier.add(Dense(output_dim=1 , init= 'uniform',activation='sigmoid'))

#compiling the nural network
classifier.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

#fitting the ANN to the training set
classifier.fit(X_train,y_train,batch_size=10,nb_epoch=100)

#Predicting the test test
y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)



#predict the test set
y_pred=classifier_Xgboost.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
print("Accuracy is ",accuracy_score(y_test,y_pred)*100)



from sklearn.preprocessing import LabelEncoder
labelEncoder_X=LabelEncoder()
y_pred=labelEncoder_X.fit_transform(y_pred)

tt=classifier_CatBoost.predict(X_train)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#tt_cm=confusion_matrix(y_train,tt)
    
acc_test=(cm[0,0]+cm[1,1])/sum(sum(cm))*100
print(acc_test)
    
F1_acc=(cm[1,1])/(cm[1,0]+cm[1,1])*100
print(F1_acc)
    
acc_train=(tt_cm[0,0]+tt_cm[1,1])/sum(sum(tt_cm))*100
print(acc_train)




    
############################## Train whole data set ################################## 

#Feature scalling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_tot=sc_X.fit_transform(X)


#Fitting the Training set (Logistic Regression)
from sklearn.linear_model import LogisticRegression
classifier1=LogisticRegression(random_state=0)
classifier1.fit(X_tot,y)

#Fitting whole test (Random Forest)
from sklearn.ensemble import RandomForestClassifier
classifier_forest1=RandomForestClassifier(n_estimators=12, criterion='entropy',random_state=0)
classifier_forest1.fit(X_tot,y)


#Fitting the data set using XGboost
from xgboost import XGBClassifier
classifier_Xgboost1=XGBClassifier(n_estimators=270, max_depth=4,learning_rate=0.1, colsample_bytree=0.8)   #polsample 0.8  n_estimator = 350
classifier_Xgboost1.fit(X_tot,y)


#Fitting data using Catboost
from catboost import CatBoostRegressor
classifier_CatBoost1=CatBoostRegressor(iterations=270, depth=3, learning_rate=0.1, loss_function='RMSE')
classifier_CatBoost1.fit(X_tot,y)


######################################################################################


#for prepare X_testing
test_set=pd.read_csv('credit_card_default_test.csv')
X_testing = test_set.iloc[:, 1:]


#preprocessing balance limits
X_testing['Balance_Limit_V1'] = X_testing['Balance_Limit_V1'].str.replace('M','000000')
X_testing['Balance_Limit_V1'] = X_testing['Balance_Limit_V1'].str.replace('K','000')
X_testing['Balance_Limit_V1'] = X_testing['Balance_Limit_V1'].astype(float)

#Finiding missing data
missing_data_testset=test_set[test_set.isnull().any(axis=1)]



# Encoding categorical data for training set
X_testing = pd.get_dummies( X_testing,columns =['Gender','EDUCATION_STATUS','MARITAL_STATUS','AGE','PAY_JULY','PAY_AUG','PAY_SEP','PAY_OCT','PAY_NOV','PAY_DEC'] )

#X_testing['PAY_AUG_8']=pd.DataFrame(np.zeros(shape=(len(X_testing),1)))

X_testing.insert(45, 'PAY_AUG_8', np.zeros(shape=(len(X_testing),1)))
X_testing.insert(49, 'PAY_SEP_1', np.zeros(shape=(len(X_testing),1)))
X_testing.insert(60, 'PAY_OCT_1', np.zeros(shape=(len(X_testing),1)))
X_testing.insert(67, 'PAY_OCT_8', np.zeros(shape=(len(X_testing),1)))
X_testing.insert(71, 'PAY_NOV_1', np.zeros(shape=(len(X_testing),1)))

X_testing.insert(78, 'PAY_NOV_8', np.zeros(shape=(len(X_testing),1)))
X_testing.insert(82, 'PAY_DEC_1', np.zeros(shape=(len(X_testing),1)))
X_testing.insert(89, 'PAY_DEC_8', np.zeros(shape=(len(X_testing),1)))



cols_to_drop = [ 'Gender_F','EDUCATION_STATUS_Graduate','MARITAL_STATUS_Other','AGE_31-45','PAY_JULY_-2','PAY_AUG_-2','PAY_SEP_-2','PAY_OCT_-2','PAY_NOV_-2','PAY_DEC_-2' ]
X_testing = X_testing.drop( cols_to_drop, axis = 1 )

X_testing=X_testing.iloc[:,:].values

#Feature scalling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_testing=sc_X.fit_transform(X_testing)

#Random Forest
y_testing=classifier_forest1.predict(X_testing)

#XGBoost
y_testing=classifier_Xgboost1.predict(X_testing)

#CatBoost
y_testing=classifier_CatBoost1.predict(X_testing)

#CatBoost
y_testing=(y_testing>0.5)


