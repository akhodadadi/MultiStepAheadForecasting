DATADIR='/home/arash/datasets/Kaggle/Recruit'

DATATYPES={'air_reserve':{'reserve_visitors':'int32'},
           'air_store_info':None,
           'air_visit_data':{'visitors':'int32'},
           'hpg_reserve':{'visitors':'int32'},
           'hpg_store_info':None,
           'date_info':{'holiday_flg':'int8'},
           'store_id_relation':None}

PARSEDATE={'air_reserve':['visit_datetime','reserve_datetime'],
           'air_store_info':False,
           'air_visit_data':['visit_date'],
           'hpg_reserve':['visit_datetime','reserve_datetime'],
           'hpg_store_info':False,
           'date_info':['calendar_date'],
           'store_id_relation':False}

