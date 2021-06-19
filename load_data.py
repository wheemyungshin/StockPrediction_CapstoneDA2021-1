max_test_size = 11
simulation_size = 4
sample_step = 1
split_iter = 37
split_stride = 1

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
sns.set()
tf.compat.v1.random.set_random_seed(1234)

import sys
import warnings

import torch
import torch.nn as nn
import torch.nn.modules.conv as conv


if not sys.warnoptions:
    warnings.simplefilter('ignore')
  
!pip install -U finance-datareader
import FinanceDataReader as fdr
# sample execution (requires torchvision)0
from PIL import Image
from torchvision import transforms

def load_data():
  start_date = '2016-01-02'
  end_date = '2020-12-31'
  max_train_size = 1200

  df_kospi = fdr.StockListing('KOSPI')
  df_kospi.head()

  df2 = fdr.DataReader('KS11', start_date, end_date)

  print(df2)

  df_ratio2 = df2.iloc[:, 0:1].astype('float32').fillna(0)
  df_log2 = pd.DataFrame(df_ratio2)


  df_dict = {
      0 : fdr.DataReader('IXIC', start_date, end_date),#나스닥
      1 : fdr.DataReader('KQ11', start_date, end_date),#코스닥
      2 : fdr.DataReader('USD/KRW', start_date, end_date),#달러/원
      3 : fdr.DataReader('KS50', start_date, end_date),#코스피50
      4 : fdr.DataReader('KS100', start_date, end_date),#코스피100
      5 : fdr.DataReader('KS200', start_date, end_date),#코스피200
      6 : fdr.DataReader('NG', start_date, end_date),#천연가스 선물
      7 : fdr.DataReader('ZG', start_date, end_date),#금 선물
      8 : fdr.DataReader('VCB', start_date, end_date),#베트남무역은행
      #9 : fdr.DataReader('KR1YT=RR', start_date, end_date),#한국채권1년수익률
      9 : fdr.DataReader('US1MT=X', start_date, end_date),#미국채권1개월수익률
  }
  '''
  df_dict = {
      0 : fdr.DataReader('IXIC', start_date, end_date),#나스닥
      1 : fdr.DataReader('USD/EUR', start_date, end_date),#달러/유로 1 : fdr.DataReader('KQ11', start_date, end_date),#코스닥
      2 : fdr.DataReader('USD/KRW', start_date, end_date),#달러/원
      3 : fdr.DataReader('KS50', start_date, end_date),#코스피50
      4 : fdr.DataReader('KS100', start_date, end_date),#코스피100
      5 : fdr.DataReader('KS200', start_date, end_date),#코스피200
      6 : fdr.DataReader('TSE', start_date, end_date),#도쿄 증권거래소 6 : fdr.DataReader('NG', start_date, end_date),#천연가스 선물
      7 : fdr.DataReader('ZG', start_date, end_date),#금 선물
      8 : fdr.DataReader('VCB', start_date, end_date),#베트남무역은행
      9 : fdr.DataReader('KR1YT=RR', start_date, end_date),#한국국채1년수익률
      10 : fdr.DataReader('US1MT=X', start_date, end_date),#미국국채1개월수익률
      11 : fdr.DataReader('KR10YT=RR', start_date, end_date),#한국국채10년수익률
      12 : fdr.DataReader('US10YT=X', start_date, end_date),#미국국채10개월수익률
  }
  '''

  for i in range(len(df_dict)):
    extra_df = df_dict[i]
    df_ratio_extra = extra_df.iloc[:, 0:1].astype('float32').fillna(0) #((extra_df.iloc[:, 0:1].astype('float32') - extra_df.iloc[:, 0:1].shift().astype('float32')) / extra_df.iloc[:, 0:1].shift().astype('float32')).fillna(0)
    df_log_extra = pd.DataFrame(df_ratio_extra)

    df_log2 = pd.concat([df_log2, df_log_extra],axis=1)

   df_trains = np.array([])
  df_tests = np.array([])
  df_vals = np.array([])
  df_val_targets = np.array([])
  df_ratios = np.array([])
  df_volumes = np.array([])

  scaler = MinMaxScaler()

  stock_names = []
  stock_dates = []

  print(np.flip(df_kospi.to_numpy(), axis=0).shape)
  read_lines = np.flip(df_kospi.to_numpy(), axis=0)[:100]
  read_lines = np.append(read_lines, np.flip(df_kospi.to_numpy(), axis=0)[3200:3300], axis=0)
  read_lines = np.append(read_lines, np.flip(df_kospi.to_numpy(), axis=0)[2200:2700], axis=0)
  read_lines = np.append(read_lines, np.flip(df_kospi.to_numpy(), axis=0)[3400:3550], axis=0)
  read_lines = np.append(read_lines, np.flip(df_kospi.to_numpy(), axis=0)[1600:1900], axis=0)
  read_lines = np.append(read_lines, np.flip(df_kospi.to_numpy(), axis=0)[5012:5025], axis=0)
  read_lines = np.append(read_lines, np.flip(df_kospi.to_numpy(), axis=0)[2900:3180], axis=0)
  min_train_size = 560
  for line in np.flip(read_lines, axis=0):
    try:
      df = fdr.DataReader(line[0], start_date, end_date)
      num_vals = df.iloc[:, 0:1].astype(bool).sum(axis=0)
      if len(num_vals) == 1:
        num_vals = int(num_vals)
      else:
        num_vals = 0
      #print(num_vals)
      #print(max_train_size + max_test_size)
      if num_vals > max_train_size + max_test_size:
        df_ratio = df.iloc[:, 3].astype('float32')
        df_log1 = pd.DataFrame(df_ratio)
        df_ratios = np.append(df_ratios, df_ratio.to_numpy())

        for j in range(0,split_iter):
            split_point_start = j * max_test_size
            split_point_end = (split_iter - j + 1) * max_test_size
            df_train1 = df_log1.iloc[-max_train_size+split_point_start:-split_point_end]
            df_test1 = df_log1.iloc[-split_point_end:-split_point_end+max_test_size]
            df_train2 = df_log2.iloc[:]#df_log2.iloc[-max_test_size-max_train_size+split_point_start:-split_point_end]
            df_test2 = df_log2.iloc[:]#df_log2.iloc[-split_point_end:-split_point_end+max_test_size]

            df_train =pd.concat([df_train1, df_train2],axis=1).dropna(axis=0)[-min_train_size:]
            df_test = pd.concat([df_test1, df_test2],axis=1).dropna(axis=0)

            if df_test.shape[0] == 0:
                print("NAN detected!:", line[2])    
                continue  

            for date in df_train.index:
              if date not in stock_dates:
                stock_dates.append(date)
            #print(df_train)
            indexes = list(df_train.index)#[::sample_step]

            #df_train = df_train.rolling(sample_step).mean()[::sample_step][1:]  
            df_train_ = np.array([])
            previous_train = np.zeros(df_train.shape[1])
            for num, i in enumerate(df_train.to_numpy()):#[::sample_step]):
                if num == 0:
                    df_train_ = np.expand_dims(previous_train, axis=0) 
                else:
                    if (previous_train == 0).any():
                      print(previous_train)
                    new_item = (i - previous_train) / previous_train
                    df_train_ = np.append(df_train_, np.expand_dims(new_item, axis=0), axis=0)
                previous_train = i


            df_train = df_train_
            df_test = df_test.to_numpy()
            df_test = np.array([(df_test[-1] - df_test[0])/ df_test[0] >= 0.02, (df_test[-1] - df_test[0])/ df_test[0] < -0.02])
            df_test = np.append(df_test, np.expand_dims(np.logical_not(df_test[0]) * np.logical_not(df_test[1]), axis=0), axis=0)


            # if min_train_size > df_train.shape[0]:
            #     min_train_size = df_train.shape[0]
            # else:
            #     df_train = df_train[-min_train_size:]


            test_size = df_test.shape[0]
            df_train_np = np.expand_dims(df_train, axis=0)
            df_test_np = np.expand_dims(df_test, axis=0)

            #print("df_train_np: ",df_train_np)

            if df_train_np[np.isnan(df_train_np)].size > 0:
                print("NAN detected!!:", line[2])    
                continue  
            if j >= (split_iter-2):# * split_stride:
                if df_vals.size == 0:
                    df_vals = df_train_np
                    df_val_targets = df_test_np
                else:
                    df_vals = np.append(df_vals, df_train_np, axis=0)
                    df_val_targets = np.append(df_val_targets, df_test_np, axis=0)

            else:
                if df_trains.size == 0:
                    df_trains = df_train_np
                    df_tests = df_test_np
                else:
                    df_trains = np.append(df_trains, df_train_np, axis=0)
                    df_tests = np.append(df_tests, df_test_np, axis=0)
            if j == split_iter - 1:
              print("Added: ", line[2])
              stock_names.append(line[2])


    except ValueError as e:
      print(e)

  df_trains = np.transpose(df_trains, (0,2,1))
  df_tests = np.transpose(df_tests, (0,2,1))
  df_vals = np.transpose(df_vals, (0,2,1))
  df_val_targets = np.transpose(df_val_targets, (0,2,1))
  print(df_trains.shape, df_tests.shape)
  print(df_vals.shape, df_val_targets.shape)

  return df_trains, df_tests, df_vals, df_val_targets
