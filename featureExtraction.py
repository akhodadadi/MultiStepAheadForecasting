import numpy as np
import pandas as pd
import recruit_config
dataDir=recruit_config.DATADIR
from sklearn.preprocessing import StandardScaler
from utils import loadData
import gc
from os.path import join


def extractTimeSeries(dstDir):
    '''
    This function extracts the time-series corresponding each 
    location and saves the results as numpy.array in binary files.
    '''
    
    #===load data===
    dataDict = loadData(['air_visit_data','hpg_reserve'])
    df_A_V,df_H_R=dataDict['air_visit_data'],dataDict['hpg_reserve']
    del dataDict;gc.collect()
    #===load data===
    
    #===create hpg visit data===
    #the resulting df will have similar fields as air_visit_data
    df_H_R['visit_date']=pd.to_datetime(df_H_R.visit_datetime.dt.date)
    df_H_R.drop(['visit_datetime','reserve_datetime'],axis=1,inplace=True)
    gb = df_H_R.groupby(['hpg_store_id','visit_date'])
    df_H_R=gb.reserve_visitors.sum().reset_index()
    df_H_R.rename(columns={'reserve_visitors':'visitors'},inplace=True)    
    #===create hpg visit data===

    #===create pivot tables with sotre_id as index and time as columns===
    df_H_R.sort_values(by=['visit_date','hpg_store_id'],inplace=True)
    pt=pd.pivot_table(df_H_R,index=['hpg_store_id'],columns=['visit_date'],
                      values=['visitors'])
    pt.columns=pt.columns.droplevel().date;pt.columns.names=[None]
    pt.reset_index(inplace=True)
    pt.fillna(0,inplace=True)
    pt.to_csv(join(dstDir,'hpg_ts.csv'),index=False)
    
    df_A_V.sort_values(by=['visit_date','air_store_id'],inplace=True)
    pt=pd.pivot_table(df_A_V,index=['air_store_id'],
                      columns=['visit_date'],values=['visitors'])
    pt.columns=pt.columns.droplevel().date;pt.columns.names=[None]
    pt.reset_index(inplace=True)
    pt.fillna(0,inplace=True)
    pt.to_csv(join(dstDir,'air_ts.csv'),index=False)
    #===create pivot tables with sotre_id as index and time as columns===
    
    
def sampleFromTimeSeries(ts,samples_per_ts=10,input_length=100,
                         pred_horizon=38,ifSample_hpg=False):
    from itertools import chain
    
    def sample_from_one_ts(x):
        x=x[1]
        store_id=x[0]
        dates=x.index[1:]
        x=x[1:].values
        start=np.where(x!=0)[0][0]#idx of first date with non-zero visit
        
        #if there are too few nnz values, only return x itself
        max_idx=x.size-pred_horizon-input_length
        if (max_idx-start)<2*samples_per_ts:
            return [(store_id,dates[max_idx],
                     x[max_idx:(max_idx+input_length)].\
                     reshape((1,input_length)))]
        
        idx=map(lambda s:int(s),
                np.linspace(start,max_idx,samples_per_ts))
        return [(store_id,dates[i],x[i:(i+input_length)],
                 x[(i+input_length):(i+input_length+pred_horizon)])\
                for i in idx]
        
                
    l=map(sample_from_one_ts,ts.iterrows())
    return list(chain.from_iterable(l))#this acts like flatmap 
        
     
def groupByTwoFeatures(df_src,df_dst,feat1,feat2,val,new_feat_name):
    '''
    given two features, this function groups df_src by those features,
    computes the mean of `val` for each group and creates a df with n rows
    and m columns where n and m are the number of levels of the first
    and the second features, respectively.
    It finally, merges the original df_dst with this df on the first
    and second features.
    '''
    
    gb=df_src.groupby(by=[feat1,feat2])[val].mean().reset_index().fillna(0)
    gb.rename(columns={val:new_feat_name},inplace=True)
    return df_dst.merge(gb,on=[feat1,feat2])

def extractDateFeatures(df_src,df_dst=None):
    
    if df_dst is None:
        #---avg visitors for each store---
        df=df_src.groupby(by='air_store_id').visitors.mean().reset_index()
        df.rename(columns={'visitors':'avg_visit'},inplace=True)
        df_dst=df_src.merge(df,on='air_store_id')
        #---avg visitors for each store---
    
        #---avg visitors grouped by holiday_flg for each store---
        df_dst=groupByTwoFeatures(df_dst,df_dst,feat1='air_store_id',
                                  feat2='holiday_flg',val='visitors',
                                  new_feat_name='avg_visit_holiday')
        #---avg visitors grouped by holiday_flg for each store---
        
        #---avg visitors grouped by dow for each store---
        df_dst=groupByTwoFeatures(df_dst,df_dst,feat1='air_store_id',
                                      feat2='dow',val='visitors',
                                      new_feat_name='avg_visit_dow')    
        #---avg visitors grouped by dow for each store---
        
        #---extract day, month and year---
        df_dst['year']=df_dst.visit_date.dt.year
        df_dst['month']=df_dst.visit_date.dt.month
        df_dst['day']=df_dst.visit_date.dt.day
        #---extract day, month and year---
        
        #---avg visitors grouped by month for each store---
        df_dst=groupByTwoFeatures(df_dst,df_dst,feat1='air_store_id',
                                      feat2='month',val='visitors',
                                      new_feat_name='avg_visit_month')    
        #---avg visitors grouped by month for each store---
        
    else:
        #---extract day, month and year---
        df_dst['year']=df_dst.visit_date.dt.year
        df_dst['month']=df_dst.visit_date.dt.month
        df_dst['day']=df_dst.visit_date.dt.day
        #---extract day, month and year---
        
        #---avg visitors for each store---
        df_dst=df_dst.merge(df_src[['air_store_id',
                                    'avg_visit']].drop_duplicates(),
                             on=['air_store_id'],how='left')

        #---avg visitors grouped by holiday_flg for each store---
        df_dst=df_dst.merge(df_src[['air_store_id','holiday_flg',
                                    'avg_visit_holiday']].drop_duplicates(),
                             on=['air_store_id','holiday_flg'],how='left')
        df_dst.fillna(df_dst.avg_visit_holiday.mean(),inplace=True)
        
        #---avg visitors grouped by dow for each store---
        df_dst=df_dst.merge(df_src[['air_store_id','dow',
                                    'avg_visit_dow']].drop_duplicates(),
                             on=['air_store_id','dow'],how='left')
        df_dst.fillna(df_dst.avg_visit_dow.mean(),inplace=True)
        
        #---avg visitors grouped by month for each store---
        df_dst=df_dst.merge(df_src[['air_store_id','month',
                                    'avg_visit_month']].drop_duplicates(),
                             on=['air_store_id','month'],how='left')
        df_dst.fillna(df_dst.avg_visit_month.mean(),inplace=True)

        
    return df_dst


def extractPrevDaysAsFeatrures(df_src,df_dst,n_prev_days=40,
                               ifStandardize=False):
    
    last_day=df_src.visit_date.max()
    rng = pd.date_range(last_day-pd.DateOffset(days=n_prev_days-1),
                        last_day,freq='D')
    
    df_src=df_src.loc[df_src.visit_date.isin(rng),
                      ['air_store_id','visit_date','visitors']]
    df_src=pd.pivot_table(df_src,index='air_store_id',columns='visit_date',
                          values='visitors')
    
    
    df_src=df_src.T.fillna(df_src.mean(axis=1)).T
    
    #---standardize each time-series---
    if ifStandardize:
        scaler = StandardScaler()
        X=scaler.fit_transform(df_src.values.T).T
        df_src=pd.DataFrame(X,columns=df_src.columns,index=df_src.index).\
                    reset_index()
    else:
        df_src=df_src.reset_index()        
    #---standardize each time-series---
    
        
    df_src.columns = ['air_store_id']+['day-{}'.format(d)\
                                        for d in np.arange(n_prev_days,0,-1)]
    df_dst=df_dst.merge(df_src,on='air_store_id',how='left').fillna(0)
    df_dst['horizon']=(df_dst.visit_date-last_day).dt.days
    
    return df_dst
    
    
    
    
    
    
    
    