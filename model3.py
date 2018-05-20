'''
We need to do the forecasting for 2017-04-23 to 2017-05-31 (38 days).
Here, we choose the training set to be from 2016-01-01 to 2017-03-14,
and the evaluation set be the data from 2017-03-15 to 2017-04-22.
After model selection, we re-train the best model on data 2016-01-01 to
2017-04-22.
'''

from utils import loadData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os.path import join
import gc
from time import ctime

import recruit_config
from featureExtraction import groupByTwoFeatures
from featureExtraction import extractDateFeatures

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.externals import joblib

import lightgbm as gbm
print('\014')

#===config===
modelSelection=False
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
categorical_features=feature_names[:8]
categorical_features=[]#???
#===config===

#===feature extraction===
if ifLoadFeatures:    
    df_train=pd.read_csv(join(dataDir,'model3_trainData.csv'))
    df_eval=pd.read_csv(join(dataDir,'model3_evalData.csv'))
    df_all=pd.read_csv(join(dataDir,'model3_allData.csv'))
    df_test=pd.read_csv(join(dataDir,'model3_testData.csv'),
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
    
    #===split train and eval sets===
    train_rng=pd.date_range('2016-01-01','2017-03-14')
    eval_rng=pd.date_range('2017-03-15','2017-04-22')
    df_V_train=df_V[df_V.visit_date.isin(train_rng)]
    df_V_eval=df_V[df_V.visit_date.isin(eval_rng)]
    #===split train and eval sets===
    
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
    df_V_train=df_V_train.merge(df_date,on='visit_date')
    df_V_eval=df_V_eval.merge(df_date,on='visit_date')
    
    df_V=df_V.merge(df_date,on='visit_date')
    df_test=df_test.merge(df_date,on='visit_date')
    #---merge df_V and df_date---
    
    #---other date-related features---
    df_V_train=extractDateFeatures(df_V_train)
    df_V_eval=extractDateFeatures(df_V_train,df_V_eval)
    
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
    df_train=df_V_train.merge(df_S,on=['air_store_id'])
    df_train.drop(['air_store_id','visit_date'],axis=1,inplace=True)
    
    df_eval=df_V_eval.merge(df_S,on=['air_store_id'])
    df_eval.drop(['air_store_id','visit_date'],axis=1,inplace=True)
    
    df_all= df_V.merge(df_S,on=['air_store_id'])  
    df_all.drop(['air_store_id','visit_date'],axis=1,inplace=True)
    
    df_test=df_test.merge(df_S,on=['air_store_id'])
#    df_test.drop(['air_store_id','visit_date'],axis=1,inplace=True)    
    #---join df_V and df_S---
    #===store-related features===
    
    #===save derived data===
    df_train.to_csv(join(dataDir,'model3_trainData.csv'),index=False)
    df_eval.to_csv(join(dataDir,'model3_evalData.csv'),index=False)
    df_all.to_csv(join(dataDir,'model3_allData.csv'),index=False)
    df_test.to_csv(join(dataDir,'model3_testData.csv'),index=False)
    #===save derived data===

X_train=df_train.loc[:,feature_names].values
Y_train=np.log1p(df_train.visitors.values)

X_eval=df_eval.loc[:,feature_names].values
Y_eval=np.log1p(df_eval.visitors.values)

X_all=df_all.loc[:,feature_names].values
Y_all=np.log1p(df_all.visitors.values)

X_test=df_test.loc[:,feature_names].values
#del df;gc.collect()    
#===feature extraction===  

#===model selection===
date=str(pd.to_datetime(ctime()).date())
if modelSelection:
    bst=gbm.LGBMRegressor(boosting_type='gbdt', num_leaves=5000, 
                          max_depth=10, 
                          learning_rate=1,n_estimators=5000,
                          subsample_for_bin=1000000, objective='rmse', 
                          subsample=.8, subsample_freq=1, 
                          colsample_bytree=.6, reg_lambda=10, 
                          silent=False)
    
    if modelFitFlg==1:
        print(ctime()+'...training model...')
        bst.fit(X=X_train,y=Y_train,
                eval_set=[(X_eval,Y_eval)],eval_names=['eval'],
                eval_metric=['rmse'],early_stopping_rounds=50,
                feature_name=feature_names,
                categorical_feature=categorical_features)
#        joblib.dump(bst,join(fittedModelDir,
#                             'model3_nonCV_{}{}'.format(date,'.pkl')))        
#===model selection===

if not modelSelection:
    #===train the final model on all data===
    #train the model with best hyper-parameters on all data        
    bst=gbm.LGBMRegressor(boosting_type='gbdt', num_leaves=5000, 
                  max_depth=10, 
                  learning_rate=1,n_estimators=2,
                  subsample_for_bin=1000000, objective='rmse', 
                  subsample=.8, subsample_freq=1, 
                  colsample_bytree=.6, reg_lambda=10, 
                  silent=False)
    
    
    print(ctime()+'...training final model...')
    bst.fit(X=X_all,y=Y_all,
            eval_set=[(X_all,Y_all)],eval_names=['eval'],
            eval_metric=['rmse'],early_stopping_rounds=5000,
            feature_name=feature_names,
            categorical_feature=categorical_features)
    joblib.dump(bst,join(fittedModelDir,
                         'model3_nonCV_{}{}'.format(date,'.pkl'))) 
    #===train the final model on all data===        
        
    #===make prediction for test set==
    fittedMdlPath='/home/arash/MEGA/MEGAsync/Machine Learning/'+\
                    'Kaggle/Recruit/Fitted models/model3_nonCV_{}.pkl'.format(date)
    bst = joblib.load(fittedMdlPath)
    gbm.plot_importance(bst)
    
    X_test=df_test.loc[:,feature_names]
    y_test=bst.predict(X_test)
    
    df=pd.DataFrame({'id':df_test.air_store_id+'_'+\
                     df_test.visit_date.dt.strftime('%Y-%m-%d'),
                     'visitors':np.expm1(y_test)})
    df.sort_values(by='id',inplace=True) 
    df.to_csv(join(submissionsDir,'model3_{}.csv'.format(date)),index=False)
    #===make prediction for test set=== 








