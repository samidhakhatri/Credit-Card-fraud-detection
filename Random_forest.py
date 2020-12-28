# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 20:06:03 2019

@author: lenovo
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import time as t


from sklearn.ensemble import RandomForestClassifier

data=pd.read_csv('creditcard.csv')


predictors=data.iloc[:,0:30]
target=data.iloc[:,30]

predictors_train,predictors_test,target_train,target_test=train_test_split(predictors,target,test_size=0.2,random_state=0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
predictors_train = sc.fit_transform(predictors_train)
predictors_test=   sc.transform(predictors_test)






rf=RandomForestClassifier()

start_time=t.time()
rf.fit(predictors_train,target_train)
print("------%s seconds------"%(t.time()-start_time))

start_time=t.time()

prediction=rf.predict(predictors_test)

print("----%s seconds-----"%(t.time()-start_time))

from sklearn import metrics

confusion=metrics.confusion_matrix(target_test,prediction)
print("CONFUSION MATRIX 1:\n %s"%(confusion))

#row,column
TP=confusion[1,1]
TN=confusion[0,0]
FP=confusion[0,1]
FN=confusion[1,0]

sensitivity=TP/float(FN+TP)
print("SENSITIVITY: %s"%(sensitivity))
#print("SENSITIVITY:"+metrics.recall_score(target_test,prediction))

'''specitivity=TN/(TN+FP)
print("SPECITIVITY:"+specitivity)

false_positive_rate=FP/float(TN+FP)
print("FALSE POSITIVE RATE:"+false_positive_rate)
print(1-specitivity)'''

precision=TP/float(TP+FP)
print("PRECISION: %s"%(precision))
print("PRECISION: %s"%(metrics.precision_score(target_test,prediction)))


rf.predict(predictors_test)[0:10] #prints the first 10 predicted responses


rf.predict_proba(predictors_test)[0:10]

#print the first 10 predicted probalities for class 1
rf.predict_proba(predictors_test)[0:10,1]

#store the predicted probalities for class 1
y_pred_prob=rf.predict_proba(predictors_test)[:,1]



from sklearn.preprocessing import binarize
y_pred_prob=y_pred_prob.reshape(-1,1)
y_pred_class=binarize(y_pred_prob,0.4)

#print the first 10 predicted probabilities
y_pred_prob[0:10]

y_pred_class[0:10]

print("CONFUSION MATRIX 1:\n%s"%(confusion))

# new confusion matrix
confusion_2=metrics.confusion_matrix(target_test,y_pred_class)
print("CONFUSION MATRIX 2:\n%s"%(confusion_2))

TP_2=confusion_2[1,1]
TN_2=confusion_2[0,0]
FP_2=confusion_2[0,1]
FN_2=confusion_2[1,0]

sensitivity_2=TP_2/float(FN_2+TP_2)
print("SENSITIVITY :%s"%(sensitivity_2))

print("PRECISION: %s"%(metrics.precision_score(target_test,y_pred_class)))
