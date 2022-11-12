import requests # to get image from the web
import shutil # to save it locally
import os
import pandas as pd
from multiprocessing import Pool
from datetime import datetime
import time
from itertools import repeat

DATA_PATH = '/Users/travistang/Documents/TorchScene/'

numberImages = 0 

if __name__ == '__main__':
    gooddata_img_df = pd.read_csv('/Users/travistang/Documents/TorchScene/data/csv/oct2022-gooddata.csv')
    df_prediction_result_oct2022_gooddata = pd.read_csv('/Users/travistang/Documents/TorchScene/result/intermediate/tpot_221109 122pm/df_prediction_result_oct2022_gooddata.csv', index_col = [0])
    gooddata_img_df['set'] = 'oct2022_gooddata'
    gooddata_img_df = gooddata_img_df.merge(df_prediction_result_oct2022_gooddata, left_on = 'outlet_id', right_index=True)


    baddata_img_df = pd.read_csv('/Users/travistang/Documents/TorchScene/data/csv/oct2022-baddata.csv')
    df_prediction_result_oct2022_baddata = pd.read_csv('/Users/travistang/Documents/TorchScene/result/intermediate/tpot_221109 122pm/df_prediction_result_oct2022_baddata.csv', index_col = [0])
    baddata_img_df['set'] = 'oct2022_baddata'
    baddata_img_df = baddata_img_df.rename({'saudagar_id':'outlet_id', 'outlet_photo_url':'restaurant_photo_url'}, axis = 1)
    baddata_img_df = baddata_img_df.merge(df_prediction_result_oct2022_baddata, left_on = 'outlet_id', right_index=True)

    oct2022_data = pd.concat([gooddata_img_df[['outlet_id','restaurant_photo_url','set','inappropriate_prob_tpot_221109']], 
                        baddata_img_df[['outlet_id','restaurant_photo_url','set','inappropriate_prob_tpot_221109']]])
    oct2022_data = oct2022_data.reset_index(drop=True)
    oct2022_data.to_csv('/Users/travistang/Documents/TorchScene/result/intermediate/tpot_221109 122pm/df_prediction_result_oct2022.csv')