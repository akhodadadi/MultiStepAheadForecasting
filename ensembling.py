import pandas as pd
import numpy as np
from os.path import join
import recruit_config
from time import ctime

date='2018-05-19'
desired_files=['model2_{}.csv'.format(date),'model3_{}.csv'.format(date),
               'model4_{}.csv'.format(date),'model5_{}.csv'.format(date),
               'model6_{}.csv'.format(date),'model8_{}.csv'.format(date),
               'model9_{}.csv'.format(date)]
weights=[2,1,3,2,3,6,1]


dataDir=join(recruit_config.DATADIR,'submissions')
for i,f in enumerate(desired_files):
    if i==0:
        df_en = pd.read_csv(join(dataDir,f))
        df_en['visitors']=weights[i]*df_en['visitors']
    else:
        df = pd.read_csv(join(dataDir,f))
        df_en['visitors']=df_en['visitors']+weights[i]*df['visitors']

df_en['visitors']=df_en['visitors']/np.sum(weights)

date=str(pd.to_datetime(ctime()).date())
df_en.to_csv(join(dataDir,'ensemble_{}.csv'.format(date)),index=False)
