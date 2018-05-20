'''
This is similar to model 3 with a major difference:
In model 3, to compute the features for a given sample we only used
the data prior to visit_date of that sample. Threfore, we had to 
use data of different time intervals for train and validation sets.

Here, we relax this constraint: we use all data to compute the 
features (e.g., avg. no. of visits for a store) and then split the 
data into train and validation randomly. In terms of time-series
cross-validation this is not a right thing to do but this way
we preserve all data for training.
'''

from utils import loadData
import pandas as pd
import numpy as np
from os.path import join
from time import ctime
import matplotlib.pyplot as plt

import recruit_config
from featureExtraction import extractDateFeatures,\
extractPrevDaysAsFeatrures

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,scale,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import lightgbm as gbm
print('\014')

#===config===
#if we want to forecast for days x to x+38, the number of visitors
#in days x-n_prev_days to x-1 are used as featues.
n_prev_days=40
modelSelection=True
ifLoadFeatures=False
modelFitFlg=1#1:fit model,2:grid search,3:load model


dataDir=join(recruit_config.DATADIR,'processed_data')
fittedModelDir='/home/arash/MEGA/MEGAsync/Machine Learning/'+\
                'Kaggle/Recruit/Fitted models'
submissionsDir='/home/arash/datasets/Kaggle/Recruit/submissions'                
                
feature_names=[u'dow', u'holiday_flg', u'gldn_flg',u'year', u'month',u'day',
               u'air_genre_name',u'air_area_name',
               u'avg_visit',u'avg_visit_holiday', u'avg_visit_dow', 
               u'avg_visit_month', u'latitude',u'longitude']

feature_names=feature_names+['day-{}'.format(d)\
                             for d in np.arange(n_prev_days,0,-1)]+['horizon']
#===config===

#===feature extraction===
if ifLoadFeatures:    
    df_train=pd.read_csv(join(dataDir,'model4_trainData.csv'))
    df_test=pd.read_csv(join(dataDir,'model4_testData.csv'),
                        parse_dates=['visit_date'])
else:
    #===load data===
    dataDict=loadData(['air_reserve','air_store_info','air_visit_data',
                       'date_info'])
    df_R,df_S,df_V,df_date=\
        (dataDict['air_reserve'],dataDict['air_store_info'],
         dataDict['air_visit_data'],dataDict['date_info'])
    df_test=pd.read_csv(join(dataDir,'test.csv'),parse_dates=['visit_date'])    
    #===load data===
    
    #===date-related features===
    #---golden week---
    rng=pd.date_range('2016-04-29',periods=7,freq='D').\
            append(pd.date_range('2017-04-29',periods=7,freq='D'))
    df_date['gldn_flg']=0
    df_date.loc[df_date.calendar_date.isin(rng),'gldn_flg']=1
    #---golden week---
    
    #---encode day of week---
    df_date.day_of_week=df_date.calendar_date.dt.dayofweek    
    df_date.rename(columns={'calendar_date':'visit_date',
                            'day_of_week':'dow'},inplace=True)
    #---encode day of week--- 
    
    #---merge df_V and df_date---    
    df_V=df_V.merge(df_date,on='visit_date')
    df_test=df_test.merge(df_date,on='visit_date')
    #---merge df_V and df_date---
    
    #---other date-related features---   
    df_V=extractDateFeatures(df_V)
    df_test=extractDateFeatures(df_V,df_test)
    #---other date-related features---
    #===date-related features===
    
    #===store-related features===
    #---encoding categorical features in df_S---
    df_S['air_genre_name']=LabelEncoder().fit_transform(df_S.air_genre_name)
    df_S['air_area_name']=LabelEncoder().fit_transform(df_S.air_area_name)
    #---encoding categorical features in df_S---
    
    #---scale lon and lat in df_S---
    df_S['latitude']=scale(df_S.latitude);
    df_S['longitude']=scale(df_S.longitude);
    #---scale lon and lat in df_S---

    #---join df_V and df_S---    
    df_train=df_V.merge(df_S,on=['air_store_id'])
    df_test=df_test.merge(df_S,on=['air_store_id']) 
    #---join df_V and df_S---
    #===store-related features===
    
    #===save derived data===
    df_train.to_csv(join(dataDir,'model4_trainData.csv'),index=False)
    df_test.to_csv(join(dataDir,'model4_testData.csv'),index=False)
    #===save derived data===

X=df_train.loc[:,feature_names].values
Y=np.log1p(df_train.visitors.values)

X_train,X_eval,Y_train,Y_eval=train_test_split(X,Y,test_size=.1)

X_test=df_test.loc[:,feature_names].values
#===feature extraction===

date=str(pd.to_datetime(ctime()).date())

    
bst=gbm.LGBMRegressor(boosting_type='gbdt', num_leaves=50, 
                      max_depth=-1, 
                      learning_rate=.1,n_estimators=5000,
                      subsample_for_bin=1000000, objective='rmse', 
                      subsample=.9, subsample_freq=1, 
                      colsample_bytree=.9, reg_lambda=10, 
                      silent=False)    

if modelFitFlg==1:
    print(ctime()+'...training model...')
    bst.fit(X=X_train,y=Y_train,
            eval_set=[(X_eval,Y_eval)],eval_names=['eval'],
            eval_metric=['rmse'],early_stopping_rounds=100,
            feature_name=feature_names)
    joblib.dump(bst,join(fittedModelDir,
                         'model4_nonCV_{}{}'.format(date,'.pkl')))



#===make prediction for test set==
fittedMdlPath='/home/arash/MEGA/MEGAsync/Machine Learning/'+\
                'Kaggle/Recruit/Fitted models/model4_nonCV_{}.pkl'.\
                format(date)
bst = joblib.load(fittedMdlPath)
gbm.plot_importance(bst)

y_test=bst.predict(X_test)

df=pd.DataFrame({'id':df_test.air_store_id+'_'+\
                 df_test.visit_date.dt.strftime('%Y-%m-%d'),
                 'visitors':np.expm1(y_test)})
df.sort_values(by='id',inplace=True) 
df.to_csv(join(submissionsDir,'model4_{}.csv'.format(date)),index=False)
#===make prediction for test set===  







