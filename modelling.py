import pandas as pd
import numpy as np
import sklearn as sk
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

df = pd.read_csv("train.csv")

df.head()
label = df['label']
images = df.drop('label',axis=1)


# plotting random digits
def digit_plotter(images,sample):
    sample = [1,2,3,500]
    for x in sample:
        
        image = images.iloc[x].values
        image = image.reshape(28,28)
        # plot the sample
        fig = plt.figure
        plt.imshow(image, cmap='gray')
        plt.title(f"Label: {label.iloc[x]}")
        plt.show()

#splitting 'images' to train and dev set
from sklearn.model_selection import train_test_split
trainx,devx,trainy,devy = train_test_split(images,label,test_size=0.05,random_state=12)

pickle.dump(trainx,open('trainx.sav','wb'))
pickle.dump(trainy,open('trainy.sav','wb'))
pickle.dump(devx,open('devx.sav','wb'))
pickle.dump(devy,open('devy.sav','wb'))

#trying out several models

from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier 
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression



def evaluator(train_pred, dev_pred, true_train, true_dev):
    print(f"Accuracy on train set: {accuracy_score(true_train, train_pred)} and dev set: {accuracy_score(true_dev, dev_pred)}")
    print(f"Precision on train set: {precision_score(true_train, train_pred, average='weighted')} and dev set: {precision_score(true_dev, dev_pred, average='weighted')}")
    print(f"Recall on train set: {recall_score(true_train, train_pred, average='weighted')} and dev set: {recall_score(true_dev, dev_pred, average='weighted')}")
    print(f"f-1 score on train set: {f1_score(true_train, train_pred, average='weighted')} and dev set: {f1_score(true_dev, dev_pred, average='weighted')}")

lg = LogisticRegression(multi_class='multinomial', random_state=12)
lg.fit(trainx, trainy)
#train_pred = lg.predict(trainx)
#dev_pred = lg.predict(devx)

pickle.dump(lg,open('logistic_regressor.sav','wb'))

rf = RandomForestClassifier(random_state=12,max_depth=10)
rf.fit(trainx, trainy)
#train_pred = rf.predict(trainx)
#dev_pred = rf.predict(devx)
pickle.dump(rf,open('randomforest.sav','wb'))
#evaluator(train_pred, dev_pred, trainy, devy)

xgb = XGBClassifier()
xgb.fit(trainx, trainy)
# train_pred = xgb.predict(trainx)
# dev_pred = xgb.predict(devx)
# evaluator(train_pred, dev_pred, trainy, devy)
pickle.dump(xgb,open('xgb.sav','wb'))



   
   