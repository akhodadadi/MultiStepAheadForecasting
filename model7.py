'''
This is an NN with two full connected layers.
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
               u'avg_visit_month', u'latitude',u'longitude','horizon']
#===config===

#===feature extraction===
if ifLoadFeatures:    
    df_train_pred=pd.read_csv(join(dataDir,'model7_trainData.csv'))
    df_eval=pd.read_csv(join(dataDir,'model7_evalData.csv'))
    df_test=pd.read_csv(join(dataDir,'model7_testData.csv'),
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
    train_feat_rng=pd.date_range('2016-01-01','2017-02-03')
    train_pred_rng=pd.date_range('2017-02-04','2017-03-14')
    train_rng=pd.date_range('2016-01-01','2017-03-14')
    eval_rng=pd.date_range('2017-03-15','2017-04-22')
    
    df_V_train_feat=df_V[df_V.visit_date.isin(train_feat_rng)]
    df_V_train_pred=df_V[df_V.visit_date.isin(train_pred_rng)]
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
    df_V_train_feat=df_V_train_feat.merge(df_date,on='visit_date')
    df_V_train_pred=df_V_train_pred.merge(df_date,on='visit_date')
    df_V_eval=df_V_eval.merge(df_date,on='visit_date')    
    df_V=df_V.merge(df_date,on='visit_date')
    df_test=df_test.merge(df_date,on='visit_date')
    #---merge df_V and df_date---
    
    #---other date-related features---
    df_V_train_feat=extractDateFeatures(df_V_train_feat)
    df_V_train_pred=extractDateFeatures(df_V_train_feat,df_V_train_pred)
    
    df_V_train=extractDateFeatures(df_V_train)
    df_V_eval=extractDateFeatures(df_V_train,df_V_eval)
    
    df_V=extractDateFeatures(df_V)
    df_test=extractDateFeatures(df_V,df_test)
    #---other date-related features---
    #===date-related features===
    
    #===use # of visitors in prev. days as featuers===
    df_V_train_pred=extractPrevDaysAsFeatrures(df_V_train_feat,
                                               df_V_train_pred,
                                               ifStandardize=False,
                                               n_prev_days=n_prev_days)
    df_V_eval=extractPrevDaysAsFeatrures(df_V_train,df_V_eval,
                                         ifStandardize=False,
                                         n_prev_days=n_prev_days)
    df_test=extractPrevDaysAsFeatrures(df_V,df_test,
                                       ifStandardize=False,
                                       n_prev_days=n_prev_days)
    #===use # of visitors in prev. days as featuers===
    
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
    df_train_pred=df_V_train_pred.merge(df_S,on=['air_store_id'])
    df_train_pred.drop(['air_store_id','visit_date'],axis=1,inplace=True)    
    
    df_eval=df_V_eval.merge(df_S,on=['air_store_id'])
    df_eval.drop(['air_store_id','visit_date'],axis=1,inplace=True)
    
    df_test=df_test.merge(df_S,on=['air_store_id']) 
    #---join df_V and df_S---
    #===store-related features===
    
    #===save derived data===
    df_train_pred.to_csv(join(dataDir,'model7_trainData.csv'),index=False)
    df_eval.to_csv(join(dataDir,'model7_evalData.csv'),index=False)
    df_test.to_csv(join(dataDir,'model7_testData.csv'),index=False)
    #===save derived data===
#===feature extraction===

#===prepare data for keras===
'''
No. of visitors in the previous days are fed into the LSTM
and then the output of this layer is concatenated with other features.
So the observations in the previous days should be in shape
[n_samples,n_prev_days,1]
'''

feat=['day-{}'.format(d) for d in np.arange(n_prev_days,0,-1)]
X_train_lags=np.log1p(df_train_pred.loc[:,feat].values)
X_eval_lags=np.log1p(df_eval.loc[:,feat].values)
X_test_lags=np.log1p(df_test.loc[:,feat].values)

train_features=df_train_pred.loc[:,feature_names].values
eval_features=df_eval.loc[:,feature_names].values
test_features=df_test.loc[:,feature_names].values

#---scale inputs---
scaler=MinMaxScaler().fit(np.concatenate((train_features,eval_features,
                           test_features),axis=0))
train_features=scaler.transform(train_features)
test_features=scaler.transform(test_features)
eval_features=scaler.transform(eval_features)
#---scale inputs---

#---scaling outputs---
Y_train=np.log1p(df_train_pred.visitors.values)
Y_eval=np.log1p(df_eval.visitors.values)
#---scaling outputs---
#===prepare data for keras===

#===build NN===
'''
To make the features compatible with LSTM model (model 8)
here  the lagged obervations are separate from 
the other features and so we first concatenate them.
'''

date=str(pd.to_datetime(ctime()).date())
fittedMdlPath='/home/arash/MEGA/MEGAsync/Machine Learning/'+\
                'Kaggle/Recruit/Fitted models/model7_nonCV_{}.pkl'.\
                format(date)
                
X_train = np.concatenate((X_train_lags,train_features),axis=1)
X_eval = np.concatenate((X_eval_lags,eval_features),axis=1)
X_test = np.concatenate((X_test_lags,test_features),axis=1)


from keras.layers import Input,Dense
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.models import load_model
from keras.optimizers import rmsprop

inputs = Input(shape=(X_train.shape[1],))
dense1=Dense(10,activation='tanh')(inputs)
dense2=Dense(10,activation='tanh')(dense1)
outputs = Dense(1)(dense2)
model = Model(inputs=inputs,outputs=outputs)

early=EarlyStopping(monitor='val_loss', min_delta=0, patience=1)
checkpoint = ModelCheckpoint(fittedMdlPath, monitor='val_loss', 
                             save_best_only=True, mode='min', period=1)

opt = rmsprop(lr=.001)
model.compile(optimizer=opt,
              loss='mean_squared_error',
              metrics=['mean_squared_error'])

history = model.fit(X_eval, Y_eval,validation_split=.1,
          batch_size=30,
          epochs=20,verbose=1,callbacks=[checkpoint])
plt.plot(np.sqrt(history.history['val_loss'][10:]))
#===build NN===


#===make prediction for test set==
model = load_model(fittedMdlPath)
y_test=model.predict(X_test)

df=pd.DataFrame({'id':df_test.air_store_id+'_'+\
                 df_test.visit_date.dt.strftime('%Y-%m-%d'),
                 'visitors':np.expm1(y_test.flatten())})
df.sort_values(by='id',inplace=True) 
df.to_csv(join(submissionsDir,'model7_{}.csv'.format(date)),index=False)
#===make prediction for test set=== 




    