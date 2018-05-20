import numpy as np
import pandas as pd
import recruit_config
from os.path import join
dataDir=recruit_config.DATADIR
from matplotlib import pylab

def loadData(files=None):
    dataTypes=recruit_config.DATATYPES
    parseDate=recruit_config.PARSEDATE
    
    if files is None:
        files=dataTypes.keys()
           
    dataDict={}
    for f in files:
        fn=join(dataDir,f+'.csv')
        dataDict.update({f:pd.read_csv(fn,dtype=dataTypes[f],
                                       parse_dates=parseDate[f])})
    
    return dataDict

def plotSpatialData(x,y,C,reduce_C_function):
    from mpl_toolkits.basemap import Basemap
    min_lat,max_lat,min_lon,max_lon=(y.min(),y.max(),x.min(),x.max())
    m = Basemap(projection='mill',
                llcrnrlon=min_lon,llcrnrlat=min_lat,
                urcrnrlon=max_lon,urcrnrlat=max_lat)

    parallels = np.arange(np.floor(min_lat),np.ceil(max_lat),2.)
    # labels = [left,right,top,bottom]
    m.drawparallels(parallels,labels=[True,True,True,True])
    meridians = np.arange(np.floor(min_lon),np.ceil(max_lon),2.)
    m.drawmeridians(meridians,labels=[True,True,True,True])
    m.drawcoastlines()

    lon,lat=m(x,y)
    hexbin_data = m.hexbin(x=lon,y=lat,C=C,
                           reduce_C_function=reduce_C_function,
                           gridsize=25,cmap=pylab.cm.Greens)
    m.colorbar(pad=.5)
    
    #===find lon and lat where the max occurs===
    max_idx=np.argmax(hexbin_data.get_array())
    max_coor = hexbin_data.get_offsets()[max_idx,:]
    lon_idx,lat_idx=(np.argmin(np.abs(max_coor[0]-lon)),\
                    np.argmin(np.abs(max_coor[1]-lat)))
    return x[lon_idx],y[lat_idx]
    #===find lon and lat where the max occurs===
    
def prepareTestData():
    '''
    This function prepares the test data. The file sample_sumission.csv
    is read and the air_store_id and visit_date are extracted
    from the id column of this file. The results are saved as `test.csv`.
    '''
    
    df=pd.read_csv(join(dataDir,'sample_submission.csv'),usecols=['id'])
    df = df.id.str.rsplit('_',n=1,expand=True)
    df.columns=['air_store_id','visit_date']
    df.to_csv(join(dataDir,'test.csv'),index=False)
    
    
def computeDist(lats1,lons1,lats2,lons2):
    '''
    This function uses haversine formula to compute the distance
    between a set of points on earth given their coordinates (lats,lons).
    See here:
        https://en.wikipedia.org/wiki/Haversine_formula
    '''
    
    phi1=np.deg2rad(lats1);lam1=np.deg2rad(lons1)
    phi2=np.deg2rad(lats2);lam2=np.deg2rad(lons2)
    phi1_mat,phi2_mat=np.meshgrid(phi1,phi2)
    lam1_mat,lam2_mat=np.meshgrid(lam1,lam2)
    delta_phi=phi1_mat-phi2_mat
    delta_lam=lam1_mat-lam2_mat
    phi1_times_phi2=np.dot(phi2.reshape((phi2.size,1)),
                           phi1.reshape((1,phi1.size)))
    
    a = np.sin(delta_phi/2)**2 + phi1_times_phi2 * np.sin(delta_lam/2)**2
    a[a>1]=1.
    dist = 2*6356.752e3*np.arcsin(np.sqrt(a))
    return dist
    

    
    
    
    
    
    
    
    
    
    
    
