'''
ML lagorithm: lightGBM
features: 'year','month','day','air_genre_name','air_area_name',
'latitude','longitude','dow','holiday_flg','gldn_flg'
grid search over: max_depth
best score on kaggle:0.72

The evaluation is not good because the evaluation set 
is not chosen from the future time-steps.
'''

from utils import loadData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import recruit_config
from os.path import join
import gc
from time import ctime

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,make_scorer
from sklearn.externals import joblib

import lightgbm as gbm
print('\014')

modelFitFlg=1#1:fit model,2:grid search,3:load model

#===config===
ifLoadFeatures=True
dataDir=join(recruit_config.DATADIR,'processed_data')
fittedModelDir='/home/arash/MEGA/MEGAsync/Machine Learning/'+\
                'Kaggle/Recruit/Fitted models'
submissionsDir='/home/arash/datasets/Kaggle/Recruit/submissions'                
                
feature_names=['year','month','day','air_genre_name',
               'air_area_name','latitude','longitude','dow','holiday_flg',
               'gldn_flg']
categorical_features=feature_names[:5]+feature_names[-3:]
#===config===
    
#===feature extraction===
if ifLoadFeatures:
    df=pd.read_csv(join(dataDir,'model1_trainData.csv'))
    df_test=pd.read_csv(join(dataDir,'model1_testData.csv'),
                        parse_dates=['visit_date'])
else:
    #===load data===
    dataDict=loadData(['air_reserve','air_store_info','air_visit_data',
                       'date_info'])
    df_A_R,df_A_S,df_A_V,df_date=\
        (dataDict['air_reserve'],dataDict['air_store_info'],
         dataDict['air_visit_data'],dataDict['date_info'])
    df_test=pd.read_csv(join(dataDir,'test.csv'),parse_dates=['visit_date'])    
    #===load data===
    
    #===encoding categorical features in df_A_S===
    df_A_S['air_genre_name']=LabelEncoder().fit_transform(df_A_S.air_genre_name)
    df_A_S['air_area_name']=LabelEncoder().fit_transform(df_A_S.air_area_name)
    #===encoding categorical features in df_A_S===
    
    #===scale lon and lat in df_A_S===
    df_A_S['latitude']=scale(df_A_S.latitude);
    df_A_S['longitude']=scale(df_A_S.longitude);
    #===scale lon and lat in df_A_S===
    
    #===extract date features from df_A_V and df_test===
    df_A_V['year']=df_A_V.visit_date.dt.year
    df_A_V['month']=df_A_V.visit_date.dt.month
    df_A_V['day']=df_A_V.visit_date.dt.day
    
    df_test['year']=df_test.visit_date.dt.year
    df_test['month']=df_test.visit_date.dt.month
    df_test['day']=df_test.visit_date.dt.day    
    #===extract date features from df_A_V and df_test===
    
    #===extract date features from df_date===
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
    #===extract date features from df_date===
    
    #===join tables===
    df=df_A_V.merge(df_A_S,on=['air_store_id']).\
                merge(df_date,on=['visit_date'])
    df.drop(['air_store_id','visit_date'],axis=1,inplace=True)
    
    df_test=df_test.merge(df_A_S,on=['air_store_id']).\
                merge(df_date,on=['visit_date'])
    #===join tables===
    
    #===save derived data===
    df.to_csv(join(dataDir,'model1_trainData.csv'),index=False)
    df_test.to_csv(join(dataDir,'model1_testData.csv'),index=False)
    #===save derived data===

X=df.loc[:,feature_names].values
Y=np.log1p(df.visitors.values)
#del df;gc.collect()    
#===feature extraction===    

#===train model===    
X_train,X_eval,Y_train,Y_eval = train_test_split(X,Y,test_size=.1)

date=str(pd.to_datetime(ctime()).date())
bst=gbm.LGBMRegressor(boosting_type='gbdt', num_leaves=31, 
                      max_depth=-1, learning_rate=0.2, n_estimators=5000,
                      subsample_for_bin=200000, objective='rmse', 
                      subsample=.9, subsample_freq=1, 
                      colsample_bytree=.9, reg_lambda=0.1, 
                      silent=False)
if modelFitFlg==1:
    print(ctime()+'...training model...')
    bst.fit(X=X_train,y=Y_train,
            eval_set=[(X_eval,Y_eval)],eval_names=['eval'],
            eval_metric=['rmse'],early_stopping_rounds=50,
            feature_name=feature_names,
            categorical_feature=categorical_features)
    joblib.dump(bst,join(fittedModelDir,
                         'model1_nonCV_{}{}'.format(date,'.pkl')))
#===train model===

#===grid search===
if modelFitFlg==2:
    fit_params={'feature_name':feature_names,
                'categorical_feature':categorical_features} 
    gs = GridSearchCV(bst,param_grid={'num_leaves':[4,8,16]},cv=3,
                      scoring=make_scorer(mean_squared_error,
                                          greater_is_better=False),
                      verbose=5,
                      fit_params=fit_params)
  
    gs_results = gs.fit(X,Y)

    df_gs=pd.DataFrame(gs_results.cv_results_)
    joblib.dump(gs_results.best_estimator_,
                join(fittedModelDir,'model1_gs_{}{}'.format(date,'.pkl')))    
#===grid search===

#===make prediction for test set==
fittedMdlPath='/home/arash/MEGA/MEGAsync/Machine Learning/'+\
                'Kaggle/Recruit/Fitted models/model1_gs_2018-04-26.pkl'
bst = joblib.load(fittedMdlPath)
gbm.plot_importance(bst)

X_test=df_test.loc[:,feature_names]
y_test=bst.predict(X_test)

df=pd.DataFrame({'id':df_test.air_store_id+'_'+\
                 df_test.visit_date.dt.strftime('%Y-%m-%d'),
                 'visitors':np.expm1(y_test)})
df.sort_values(by='id',inplace=True) 
df.to_csv(join(submissionsDir,'model1_{}.csv'.format(date)),index=False)
#===make prediction for test set===
    















    