from utils import loadData
import pandas as pd
import numpy as np
from os.path import join
import matplotlib.pyplot as plt

import recruit_config

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import xgboost as xgb

print('\014')

#===config===
dataDir=join(recruit_config.DATADIR,'processed_data')
figDir='/home/arash/MEGA/MEGAsync/Machine Learning/Kaggle/Recruit/Figures'
fittedModelDir='/home/arash/MEGA/MEGAsync/Machine Learning/'+\
                'Kaggle/Recruit/Fitted models'              
                
feature_names=[u'dow', u'holiday_flg', u'gldn_flg',u'year', u'month',u'day',
               u'air_genre_name',u'air_area_name',
               u'avg_visit',u'avg_visit_holiday', u'avg_visit_dow', 
               u'avg_visit_month', u'latitude',u'longitude']
#===config===

#===load data===
df_train=pd.read_csv(join(dataDir,'model6_trainData.csv'))
df_train['visit_date']=pd.to_datetime(df_train.visit_date)
X=df_train.loc[:,feature_names].values
Y=np.log1p(df_train.visitors.values)

X_train,X_eval,Y_train,Y_eval=train_test_split(X,Y,test_size=.1,
                                               random_state=5)    
X_train=pd.DataFrame(X_train,columns=feature_names)
X_eval=pd.DataFrame(X_eval,columns=feature_names)
#===load data===

#===load fitted model===
fittedMdlPath='/home/arash/MEGA/MEGAsync/Machine Learning/'+\
                'Kaggle/Recruit/Fitted models/model6_nonCV_2018-05-20.pkl'
bst = joblib.load(fittedMdlPath)    
#===load fitted model===  

#===compute prediction error===
y_pred=bst.predict(X_eval)
err=Y_eval-y_pred
X_eval=X_eval.assign(err=err)
#===compute prediction error===

#===plot error vs features===
desired_features=[u'dow', u'holiday_flg', u'gldn_flg', u'month',u'day',
                  u'air_genre_name',u'air_area_name',u'avg_visit']
fig=plt.figure(figsize=(16,8))
for i,feat in enumerate(desired_features):
    fig.add_subplot(2,4,i+1)
    plt.plot(X_eval[feat],X_eval.err,'.')
    plt.xlabel(feat,fontsize=20)

fig.savefig(join(figDir,'prediction_error.jpg'))
#===plot error vs features===

#===feature importance===
ax = xgb.plot_importance(bst)
ax.figure.set_figwidth(10)
ax.figure.savefig(join(figDir,'feature_importance.jpg'))
#===feature importance===

#===plot predicted time-series for desired stores===
derired_stores=['air_74cf22153214064c','air_ba937bf13d40fb24',
                'air_26f10355d9b4d82a','air_cfcc94797d2b5d3d']

fig=plt.figure(figsize=(16,8));
h=100;

for i,store in enumerate(derired_stores):
    df=df_train.loc[df_train.air_store_id==store,:].\
        sort_values(by='visit_date')
    
    x_pred=df.loc[:,feature_names]
    df=df.assign(y_pred=bst.predict(x_pred))
    fig.add_subplot(2,2,i+1)
    
    plt.plot(df.visit_date[-h:],df.y_pred[-h:]);
    plt.plot(df.visit_date[-37:],np.log1p(df.visitors[-37:]),'--');
    plt.xticks([],rotation=90) if i<2 else plt.xticks(rotation=90)


fig.savefig(join(figDir,'ts_prediction.jpg'))    
#===plot predicted time-series for desired stores===


  