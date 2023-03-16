import pandas as pd
import numpy as np
from datetime import datetime

def fe_lag(dataset, items):
    dataset = dataset.copy()
    return_df = pd.DataFrame()
    for item in items:
        data = dataset[dataset['item_code'] == item]
        data = data.copy()
        data['qs_week1'] = data['quantity_sold'].shift(1).fillna(0)
        data['qs_week2'] = data['quantity_sold'].shift(2).fillna(0)
        data['qs_week3'] = data['quantity_sold'].shift(3).fillna(0)
        data['qs_week4'] = data['quantity_sold'].shift(4).fillna(0)
        data['qs_week5'] = data['quantity_sold'].shift(5).fillna(0)
        data['qs_week6'] = data['quantity_sold'].shift(6).fillna(0)
        data['qs_week7'] = data['quantity_sold'].shift(7).fillna(0)
        data['qs_week8'] = data['quantity_sold'].shift(8).fillna(0)
        data['qs_week9'] = data['quantity_sold'].shift(9).fillna(0)
        data['qs_week10'] = data['quantity_sold'].shift(10).fillna(0)
        data['qs_week11'] = data['quantity_sold'].shift(11).fillna(0)
        data['qs_week12'] = data['quantity_sold'].shift(12).fillna(0)
        
        
        return_df = return_df.append(data, ignore_index=True)
    return return_df

def fe_rolling_sum(dataset, items):
    dataset = dataset.copy()
    return_df = pd.DataFrame()
    for item in items:
        data = dataset[dataset['item_code'] == item]
        data = data.copy()
        data['qs_week2_rsum'] = data['quantity_sold'].rolling(2).sum().fillna(0)
        data['qs_week3_rsum'] = data['quantity_sold'].rolling(3).sum().fillna(0)
        data['qs_week4_rsum'] = data['quantity_sold'].rolling(4).sum().fillna(0)
        data['qs_week5_rsum'] = data['quantity_sold'].rolling(5).sum().fillna(0)
        data['qs_week6_rsum'] = data['quantity_sold'].rolling(6).sum().fillna(0)
        data['qs_week7_rsum'] = data['quantity_sold'].rolling(7).sum().fillna(0)
        data['qs_week8_rsum'] = data['quantity_sold'].rolling(8).sum().fillna(0)
        data['qs_week9_rsum'] = data['quantity_sold'].rolling(9).sum().fillna(0)
        data['qs_week10_rsum'] = data['quantity_sold'].rolling(10).sum().fillna(0)
        data['qs_week11_rsum'] = data['quantity_sold'].rolling(11).sum().fillna(0)
        data['qs_week12_rsum'] = data['quantity_sold'].rolling(12).sum().fillna(0)
        return_df = return_df.append(data, ignore_index=True)
    return return_df   
    
def fe_rolling_mean(dataset, items):
    dataset = dataset.copy()
    return_df = pd.DataFrame()
    for item in items:
        data = dataset[dataset['item_code'] == item]
        data = data.copy()
        
        data['qs_week2_rmean'] = data['quantity_sold'].rolling(window=2).mean().fillna(0)
        data['qs_week3_rmean'] = data['quantity_sold'].rolling(window=3).mean().fillna(0)
        data['qs_week4_rmean'] = data['quantity_sold'].rolling(window=4).mean().fillna(0)
        data['qs_week5_rmean'] = data['quantity_sold'].rolling(window=5).mean().fillna(0)
        data['qs_week6_rmean'] = data['quantity_sold'].rolling(window=6).mean().fillna(0)
        data['qs_week7_rmean'] = data['quantity_sold'].rolling(window=7).mean().fillna(0)
        data['qs_week8_rmean'] = data['quantity_sold'].rolling(window=8).mean().fillna(0)
        data['qs_week9_rmean'] = data['quantity_sold'].rolling(window=9).mean().fillna(0)
        data['qs_week10_rmean'] = data['quantity_sold'].rolling(window=10).mean().fillna(0)
        data['qs_week11_rmean'] = data['quantity_sold'].rolling(window=11).mean().fillna(0)
        data['qs_week12_rmean'] = data['quantity_sold'].rolling(window=12).mean().fillna(0)
        return_df = return_df.append(data, ignore_index=True)
    return return_df

def fe_expanding_mean(dataset, items):
    dataset = dataset.copy()
    return_df = pd.DataFrame()
    for item in items:
        data = dataset[dataset['item_code'] == item]
        data = data.copy()
        
        data['qs_week2_exmean'] = data['quantity_sold'].expanding(2).mean().fillna(0)
        data['qs_week3_exmean'] = data['quantity_sold'].expanding(3).mean().fillna(0)
        data['qs_week4_exmean'] = data['quantity_sold'].expanding(4).mean().fillna(0)
        data['qs_week5_exmean'] = data['quantity_sold'].expanding(5).mean().fillna(0)
        data['qs_week6_exmean'] = data['quantity_sold'].expanding(6).mean().fillna(0)
        data['qs_week7_exmean'] = data['quantity_sold'].expanding(7).mean().fillna(0)
        data['qs_week8_exmean'] = data['quantity_sold'].expanding(8).mean().fillna(0)
        data['qs_week9_exmean'] = data['quantity_sold'].expanding(9).mean().fillna(0)
        data['qs_week10_exmean'] = data['quantity_sold'].expanding(10).mean().fillna(0)
        data['qs_week11_exmean'] = data['quantity_sold'].expanding(11).mean().fillna(0)
        data['qs_week12_exmean'] = data['quantity_sold'].expanding(12).mean().fillna(0)
        return_df = return_df.append(data, ignore_index=True)
    return return_df

def fe_rolling_std(dataset, items):
    dataset = dataset.copy()
    return_df = pd.DataFrame()
    for item in items:
        data = dataset[dataset['item_code'] == item]
        data = data.copy()
        
        data['qs_week2_rstd'] = data['quantity_sold'].rolling(window=2).std().fillna(0)
        data['qs_week3_rstd'] = data['quantity_sold'].rolling(window=3).std().fillna(0)
        data['qs_week4_rstd'] = data['quantity_sold'].rolling(window=4).std().fillna(0)
        data['qs_week5_rstd'] = data['quantity_sold'].rolling(window=5).std().fillna(0)
        data['qs_week6_rstd'] = data['quantity_sold'].rolling(window=6).std().fillna(0)
        data['qs_week7_rstd'] = data['quantity_sold'].rolling(window=7).std().fillna(0)
        data['qs_week8_rstd'] = data['quantity_sold'].rolling(window=8).std().fillna(0)
        data['qs_week9_rstd'] = data['quantity_sold'].rolling(window=9).std().fillna(0)
        data['qs_week10_rstd'] = data['quantity_sold'].rolling(window=10).std().fillna(0)
        data['qs_week11_rstd'] = data['quantity_sold'].rolling(window=11).std().fillna(0)
        data['qs_week12_rstd'] = data['quantity_sold'].rolling(window=12).std().fillna(0)
        return_df = return_df.append(data, ignore_index=True)
    return return_df

def seasons(dataset):
    dataset[['is_pandemic', 'ber_month', 'rainy_season', 'march_may_period']] = 0
    dataset.loc[(dataset.transaction_date.dt.year == 2019) & (dataset.transaction_date.dt.month >= 4) | (dataset.transaction_date.dt.year == 2020) & (dataset.transaction_date.dt.month <= 12), 'is_pandemic'] = 1
    dataset.loc[(dataset.transaction_date.dt.month >= 9), 'ber_month'] = 1
    dataset.loc[(dataset.transaction_date.dt.month >= 6) & (dataset.transaction_date.dt.month <= 11), 'rainy_season'] = 1
    dataset.loc[(dataset.transaction_date.dt.month >= 4) & (dataset.transaction_date.dt.month <= 6), 'summer_season'] = 1

def pandemic_time(dataset):
    dataset = dataset.copy()
    non_pandemic = dataset[dataset['is_pandemic'] == 0]
    pandemic = dataset[dataset['is_pandemic'] == 1]
    pandemic = pandemic.copy()
    pandemic[['qs_pandemic']] = 0
    pandemic['qs_pandemic'] = pandemic['quantity_sold'].fillna(pandemic['quantity_sold'])
    pandemic_item = pandemic.item_code.unique()
    
    return_df = pd.DataFrame()
    rolling_sum = pd.DataFrame()
    for item in pandemic_item:
        data = pandemic[pandemic['item_code'] == item]
        data = data.copy()

        data['qs_pandemic_rsum2'] = data['quantity_sold'].rolling(2).sum().fillna(0)
        data['qs_pandemic_rsum3'] = data['quantity_sold'].rolling(3).sum().fillna(0)
        data['qs_pandemic_rsum4'] = data['quantity_sold'].rolling(4).sum().fillna(0)
        data['qs_pandemic_rsum5'] = data['quantity_sold'].rolling(5).sum().fillna(0)
        data['qs_pandemic_rsum6'] = data['quantity_sold'].rolling(6).sum().fillna(0)
        data['qs_pandemic_rsum7'] = data['quantity_sold'].rolling(7).sum().fillna(0)
        data['qs_pandemic_rsum8'] = data['quantity_sold'].rolling(8).sum().fillna(0)
        data['qs_pandemic_rsum9'] = data['quantity_sold'].rolling(9).sum().fillna(0)
        data['qs_pandemic_rsum10'] = data['quantity_sold'].rolling(10).sum().fillna(0)
        data['qs_pandemic_rsum11'] = data['quantity_sold'].rolling(11).sum().fillna(0)
        data['qs_pandemic_rsum12'] = data['quantity_sold'].rolling(12).sum().fillna(0)
        
        data['qs_pandemic_rmean2'] = data['quantity_sold'].rolling(2).mean().fillna(0)
        data['qs_pandemic_rmean3'] = data['quantity_sold'].rolling(3).mean().fillna(0)
        data['qs_pandemic_rmean4'] = data['quantity_sold'].rolling(4).mean().fillna(0)
        data['qs_pandemic_rmean5'] = data['quantity_sold'].rolling(5).mean().fillna(0)
        data['qs_pandemic_rmean6'] = data['quantity_sold'].rolling(6).mean().fillna(0)
        data['qs_pandemic_rmean7'] = data['quantity_sold'].rolling(7).mean().fillna(0)
        data['qs_pandemic_rmean8'] = data['quantity_sold'].rolling(8).mean().fillna(0)
        data['qs_pandemic_rmean9'] = data['quantity_sold'].rolling(9).mean().fillna(0)
        data['qs_pandemic_rmean10'] = data['quantity_sold'].rolling(10).mean().fillna(0)
        data['qs_pandemic_rmean11'] = data['quantity_sold'].rolling(11).mean().fillna(0)
        data['qs_pandemic_rmean12'] = data['quantity_sold'].rolling(12).mean().fillna(0)
        
        data['qs_pandemic_week1'] = data['quantity_sold'].shift(1).fillna(0)
        data['qs_pandemic_week2'] = data['quantity_sold'].shift(2).fillna(0)
        data['qs_pandemic_week3'] = data['quantity_sold'].shift(3).fillna(0)
        data['qs_pandemic_week4'] = data['quantity_sold'].shift(4).fillna(0)
        data['qs_pandemic_week5'] = data['quantity_sold'].shift(5).fillna(0)
        data['qs_pandemic_week6'] = data['quantity_sold'].shift(6).fillna(0)
        data['qs_pandemic_week7'] = data['quantity_sold'].shift(7).fillna(0)
        data['qs_pandemic_week8'] = data['quantity_sold'].shift(8).fillna(0)
        data['qs_pandemic_week9'] = data['quantity_sold'].shift(9).fillna(0)
        data['qs_pandemic_week10'] = data['quantity_sold'].shift(10).fillna(0)
        data['qs_pandemic_week11'] = data['quantity_sold'].shift(11).fillna(0)
        data['qs_pandemic_week12'] = data['quantity_sold'].shift(12).fillna(0)
        
        rolling_sum = rolling_sum.append(data, ignore_index=True)
    return_df = non_pandemic.append(rolling_sum, ignore_index=True).fillna(0)

    if rolling_sum.empty:
        non_pandemic_ = pd.DataFrame()
        non_pandemic_[['qs_pandemic']] = 0
        non_pandemic_['qs_pandemic_rsum2'] = (0)
        non_pandemic_['qs_pandemic_rsum3'] = (0)
        non_pandemic_['qs_pandemic_rsum4'] = (0)
        non_pandemic_['qs_pandemic_rsum5'] = (0)
        non_pandemic_['qs_pandemic_rsum6'] = (0)
        non_pandemic_['qs_pandemic_rsum7'] = (0)
        non_pandemic_['qs_pandemic_rsum8'] = (0)
        non_pandemic_['qs_pandemic_rsum9'] = (0)
        non_pandemic_['qs_pandemic_rsum10'] = (0)
        non_pandemic_['qs_pandemic_rsum11'] = (0)
        non_pandemic_['qs_pandemic_rsum12'] = (0)
        
        non_pandemic_['qs_pandemic_rmean2'] = (0)
        non_pandemic_['qs_pandemic_rmean3'] = (0)
        non_pandemic_['qs_pandemic_rmean4'] = (0)
        non_pandemic_['qs_pandemic_rmean5'] = (0)
        non_pandemic_['qs_pandemic_rmean6'] = (0)
        non_pandemic_['qs_pandemic_rmean7'] = (0)
        non_pandemic_['qs_pandemic_rmean8'] = (0)
        non_pandemic_['qs_pandemic_rmean9'] = (0)
        non_pandemic_['qs_pandemic_rmean10'] = (0)
        non_pandemic_['qs_pandemic_rmean11'] = (0)
        non_pandemic_['qs_pandemic_rmean12'] = (0)
        
        non_pandemic_['qs_pandemic_week1'] = (0)
        non_pandemic_['qs_pandemic_week2'] = (0)
        non_pandemic_['qs_pandemic_week3'] = (0)
        non_pandemic_['qs_pandemic_week4'] = (0)
        non_pandemic_['qs_pandemic_week5'] = (0)
        non_pandemic_['qs_pandemic_week6'] = (0)
        non_pandemic_['qs_pandemic_week7'] = (0)
        non_pandemic_['qs_pandemic_week8'] = (0)
        non_pandemic_['qs_pandemic_week9'] = (0)
        non_pandemic_['qs_pandemic_week10'] = (0)
        non_pandemic_['qs_pandemic_week11'] = (0)
        non_pandemic_['qs_pandemic_week12'] = (0)
    return_df = non_pandemic.append(non_pandemic_, ignore_index=True).fillna(0)
    return return_df

def non_rainy_season(dataset):
    dataset = dataset.copy()
    non_rainy_season = dataset[dataset['rainy_season'] == 0]
    rainy_season = dataset[dataset['rainy_season'] == 1]
    non_rainy_season = non_rainy_season.copy()
    non_rainy_season[['qs_non_rainy_season']] = 0
    non_rainy_season['qs_non_rainy_season'] = non_rainy_season['quantity_sold'].fillna(non_rainy_season['quantity_sold'])
    non_rainy_season_item = non_rainy_season.item_code.unique()
    
    return_df = pd.DataFrame()
    rolling_sum = pd.DataFrame()
    for item in non_rainy_season_item:
        data = non_rainy_season[non_rainy_season['item_code'] == item]
        data = data.copy()

        data['qs_non_rainy_season_rsum2'] = data['quantity_sold'].rolling(2).sum().fillna(0)
        data['qs_non_rainy_season_rsum3'] = data['quantity_sold'].rolling(3).sum().fillna(0)
        data['qs_non_rainy_season_rsum4'] = data['quantity_sold'].rolling(4).sum().fillna(0)
        data['qs_non_rainy_season_rsum5'] = data['quantity_sold'].rolling(5).sum().fillna(0)
        data['qs_non_rainy_season_rsum6'] = data['quantity_sold'].rolling(6).sum().fillna(0)
        data['qs_non_rainy_season_rsum7'] = data['quantity_sold'].rolling(7).sum().fillna(0)
        data['qs_non_rainy_season_rsum8'] = data['quantity_sold'].rolling(8).sum().fillna(0)
        data['qs_non_rainy_season_rsum9'] = data['quantity_sold'].rolling(9).sum().fillna(0)
        data['qs_non_rainy_season_rsum10'] = data['quantity_sold'].rolling(10).sum().fillna(0)
        data['qs_non_rainy_season_rsum11'] = data['quantity_sold'].rolling(11).sum().fillna(0)
        data['qs_non_rainy_season_rsum12'] = data['quantity_sold'].rolling(12).sum().fillna(0)
        
        data['qs_non_rainy_season_rmean2'] = data['quantity_sold'].rolling(2).mean().fillna(0)
        data['qs_non_rainy_season_rmean3'] = data['quantity_sold'].rolling(3).mean().fillna(0)
        data['qs_non_rainy_season_rmean4'] = data['quantity_sold'].rolling(4).mean().fillna(0)
        data['qs_non_rainy_season_rmean5'] = data['quantity_sold'].rolling(5).mean().fillna(0)
        data['qs_non_rainy_season_rmean6'] = data['quantity_sold'].rolling(6).mean().fillna(0)
        data['qs_non_rainy_season_rmean7'] = data['quantity_sold'].rolling(7).mean().fillna(0)
        data['qs_non_rainy_season_rmean8'] = data['quantity_sold'].rolling(8).mean().fillna(0)
        data['qs_non_rainy_season_rmean9'] = data['quantity_sold'].rolling(9).mean().fillna(0)
        data['qs_non_rainy_season_rmean10'] = data['quantity_sold'].rolling(10).mean().fillna(0)
        data['qs_non_rainy_season_rmean11'] = data['quantity_sold'].rolling(11).mean().fillna(0)
        data['qs_non_rainy_season_rmean12'] = data['quantity_sold'].rolling(12).mean().fillna(0)
        
        data['qs_non_rainy_season_week1'] = data['quantity_sold'].shift(1).fillna(0)
        data['qs_non_rainy_season_week2'] = data['quantity_sold'].shift(2).fillna(0)
        data['qs_non_rainy_season_week3'] = data['quantity_sold'].shift(3).fillna(0)
        data['qs_non_rainy_season_week4'] = data['quantity_sold'].shift(4).fillna(0)
        data['qs_non_rainy_season_week5'] = data['quantity_sold'].shift(5).fillna(0)
        data['qs_non_rainy_season_week6'] = data['quantity_sold'].shift(6).fillna(0)
        data['qs_non_rainy_season_week7'] = data['quantity_sold'].shift(7).fillna(0)
        data['qs_non_rainy_season_week8'] = data['quantity_sold'].shift(8).fillna(0)
        data['qs_non_rainy_season_week9'] = data['quantity_sold'].shift(9).fillna(0)
        data['qs_non_rainy_season_week10'] = data['quantity_sold'].shift(10).fillna(0)
        data['qs_non_rainy_season_week11'] = data['quantity_sold'].shift(11).fillna(0)
        data['qs_non_rainy_season_week12'] = data['quantity_sold'].shift(12).fillna(0)
        
        rolling_sum = rolling_sum.append(data, ignore_index=True)
    return_df = rainy_season.append(rolling_sum, ignore_index=True).fillna(0)
    return return_df

def rainy_season(dataset):
    dataset = dataset.copy()
    non_rainy_season = dataset[dataset['rainy_season'] == 0]
    rainy_season = dataset[dataset['rainy_season'] == 1]
    rainy_season = rainy_season.copy()
    rainy_season[['qs_rainy_season']] = 0
    rainy_season['qs_rainy_season'] = rainy_season['quantity_sold'].fillna(rainy_season['quantity_sold'])
    rainy_season_item = non_rainy_season.item_code.unique()
    
    return_df = pd.DataFrame()
    rolling_sum = pd.DataFrame()
    for item in rainy_season_item:
        data = rainy_season[rainy_season['item_code'] == item]
        data = data.copy()

        data['qs_rainy_season_rsum2'] = data['quantity_sold'].rolling(2).sum().fillna(0)
        data['qs_rainy_season_rsum3'] = data['quantity_sold'].rolling(3).sum().fillna(0)
        data['qs_rainy_season_rsum4'] = data['quantity_sold'].rolling(4).sum().fillna(0)
        data['qs_rainy_season_rsum5'] = data['quantity_sold'].rolling(5).sum().fillna(0)
        data['qs_rainy_season_rsum6'] = data['quantity_sold'].rolling(6).sum().fillna(0)
        data['qs_rainy_season_rsum7'] = data['quantity_sold'].rolling(7).sum().fillna(0)
        data['qs_rainy_season_rsum8'] = data['quantity_sold'].rolling(8).sum().fillna(0)
        data['qs_rainy_season_rsum9'] = data['quantity_sold'].rolling(9).sum().fillna(0)
        data['qs_rainy_season_rsum10'] = data['quantity_sold'].rolling(10).sum().fillna(0)
        data['qs_rainy_season_rsum11'] = data['quantity_sold'].rolling(11).sum().fillna(0)
        data['qs_rainy_season_rsum12'] = data['quantity_sold'].rolling(12).sum().fillna(0)
        
        data['qs_rainy_season_rmean2'] = data['quantity_sold'].rolling(2).mean().fillna(0)
        data['qs_rainy_season_rmean3'] = data['quantity_sold'].rolling(3).mean().fillna(0)
        data['qs_rainy_season_rmean4'] = data['quantity_sold'].rolling(4).mean().fillna(0)
        data['qs_rainy_season_rmean5'] = data['quantity_sold'].rolling(5).mean().fillna(0)
        data['qs_rainy_season_rmean6'] = data['quantity_sold'].rolling(6).mean().fillna(0)
        data['qs_rainy_season_rmean7'] = data['quantity_sold'].rolling(7).mean().fillna(0)
        data['qs_rainy_season_rmean8'] = data['quantity_sold'].rolling(8).mean().fillna(0)
        data['qs_rainy_season_rmean9'] = data['quantity_sold'].rolling(9).mean().fillna(0)
        data['qs_rainy_season_rmean10'] = data['quantity_sold'].rolling(10).mean().fillna(0)
        data['qs_rainy_season_rmean11'] = data['quantity_sold'].rolling(11).mean().fillna(0)
        data['qs_rainy_season_rmean12'] = data['quantity_sold'].rolling(12).mean().fillna(0)
        
        data['qs_rainy_season_week1'] = data['quantity_sold'].shift(1).fillna(0)
        data['qs_rainy_season_week2'] = data['quantity_sold'].shift(2).fillna(0)
        data['qs_rainy_season_week3'] = data['quantity_sold'].shift(3).fillna(0)
        data['qs_rainy_season_week4'] = data['quantity_sold'].shift(4).fillna(0)
        data['qs_rainy_season_week5'] = data['quantity_sold'].shift(5).fillna(0)
        data['qs_rainy_season_week6'] = data['quantity_sold'].shift(6).fillna(0)
        data['qs_rainy_season_week7'] = data['quantity_sold'].shift(7).fillna(0)
        data['qs_rainy_season_week8'] = data['quantity_sold'].shift(8).fillna(0)
        data['qs_rainy_season_week9'] = data['quantity_sold'].shift(9).fillna(0)
        data['qs_rainy_season_week10'] = data['quantity_sold'].shift(10).fillna(0)
        data['qs_rainy_season_week11'] = data['quantity_sold'].shift(11).fillna(0)
        data['qs_rainy_season_week12'] = data['quantity_sold'].shift(12).fillna(0)
        
        rolling_sum = rolling_sum.append(data, ignore_index=True)
    return_df = non_rainy_season.append(rolling_sum, ignore_index=True).fillna(0)
    return return_df


def bermonths_time(dataset):
    dataset = dataset.copy()
    non_bermonths = dataset[dataset['ber_month'] == 0]
    bermonths = dataset[dataset['ber_month'] == 1]
    bermonths = bermonths.copy()
    bermonths[['qs_ber_months']] = 0
    bermonths['qs_ber_months'] = bermonths['quantity_sold'].fillna(bermonths['quantity_sold'])
    bermonths_item = bermonths.item_code.unique()
    
    return_df = pd.DataFrame()
    rolling_sum = pd.DataFrame()
    for item in bermonths_item:
        data = bermonths[bermonths['item_code'] == item]
        data = data.copy()

        data['qs_bermonths_rsum2'] = data['quantity_sold'].rolling(2).sum().fillna(0)
        data['qs_bermonths_rsum3'] = data['quantity_sold'].rolling(3).sum().fillna(0)
        data['qs_bermonths_rsum4'] = data['quantity_sold'].rolling(4).sum().fillna(0)
        data['qs_bermonths_rsum5'] = data['quantity_sold'].rolling(5).sum().fillna(0)
        data['qs_bermonths_rsum6'] = data['quantity_sold'].rolling(6).sum().fillna(0)
        data['qs_bermonths_rsum7'] = data['quantity_sold'].rolling(7).sum().fillna(0)
        data['qs_bermonths_rsum8'] = data['quantity_sold'].rolling(8).sum().fillna(0)
        data['qs_bermonths_rsum9'] = data['quantity_sold'].rolling(9).sum().fillna(0)
        data['qs_bermonths_rsum10'] = data['quantity_sold'].rolling(10).sum().fillna(0)
        data['qs_bermonths_rsum11'] = data['quantity_sold'].rolling(11).sum().fillna(0)
        data['qs_bermonths_rsum12'] = data['quantity_sold'].rolling(12).sum().fillna(0)
        
        data['qs_bermonths_rmean2'] = data['quantity_sold'].rolling(2).mean().fillna(0)
        data['qs_bermonths_rmean3'] = data['quantity_sold'].rolling(3).mean().fillna(0)
        data['qs_bermonths_rmean4'] = data['quantity_sold'].rolling(4).mean().fillna(0)
        data['qs_bermonths_rmean5'] = data['quantity_sold'].rolling(5).mean().fillna(0)
        data['qs_bermonths_rmean6'] = data['quantity_sold'].rolling(6).mean().fillna(0)
        data['qs_bermonths_rmean7'] = data['quantity_sold'].rolling(7).mean().fillna(0)
        data['qs_bermonths_rmean8'] = data['quantity_sold'].rolling(8).mean().fillna(0)
        data['qs_bermonths_rmean9'] = data['quantity_sold'].rolling(9).mean().fillna(0)
        data['qs_bermonths_rmean10'] = data['quantity_sold'].rolling(10).mean().fillna(0)
        data['qs_bermonths_rmean11'] = data['quantity_sold'].rolling(11).mean().fillna(0)
        data['qs_bermonths_rmean12'] = data['quantity_sold'].rolling(12).mean().fillna(0)
        
        data['qs_bermonths_week1'] = data['quantity_sold'].shift(1).fillna(0)
        data['qs_bermonths_week2'] = data['quantity_sold'].shift(2).fillna(0)
        data['qs_bermonths_week3'] = data['quantity_sold'].shift(3).fillna(0)
        data['qs_bermonths_week4'] = data['quantity_sold'].shift(4).fillna(0)
        data['qs_bermonths_week5'] = data['quantity_sold'].shift(5).fillna(0)
        data['qs_bermonths_week6'] = data['quantity_sold'].shift(6).fillna(0)
        data['qs_bermonths_week7'] = data['quantity_sold'].shift(7).fillna(0)
        data['qs_bermonths_week8'] = data['quantity_sold'].shift(8).fillna(0)
        data['qs_bermonths_week9'] = data['quantity_sold'].shift(9).fillna(0)
        data['qs_bermonths_week10'] = data['quantity_sold'].shift(10).fillna(0)
        data['qs_bermonths_week11'] = data['quantity_sold'].shift(11).fillna(0)
        data['qs_bermonths_week12'] = data['quantity_sold'].shift(12).fillna(0)
        
        rolling_sum = rolling_sum.append(data, ignore_index=True)
    return_df = non_bermonths.append(rolling_sum, ignore_index=True).fillna(0)
    return return_df

def summer_season(dataset):
    dataset = dataset.copy()
    non_april_may_period = dataset[dataset['summer_season'] == 0]
    april_may_period = dataset[dataset['summer_season'] == 1]
    april_may_period = april_may_period.copy()
    april_may_period[['qs_summer_season']] = 0
    april_may_period['qs_summer_season'] = april_may_period['quantity_sold'].fillna(april_may_period['quantity_sold'])
    april_may_period_item = april_may_period.item_code.unique()
    
    return_df = pd.DataFrame()
    rolling_sum = pd.DataFrame()
    for item in april_may_period_item:
        data = april_may_period[april_may_period['item_code'] == item]
        data = data.copy()

        data['qs_summer_season_rsum2'] = data['quantity_sold'].rolling(2).sum().fillna(0)
        data['qs_summer_season_rsum3'] = data['quantity_sold'].rolling(3).sum().fillna(0)
        data['qs_summer_season_rsum4'] = data['quantity_sold'].rolling(4).sum().fillna(0)
        data['qs_summer_season_rsum5'] = data['quantity_sold'].rolling(5).sum().fillna(0)
        
        data['qs_summer_season_rmean2'] = data['quantity_sold'].rolling(2).mean().fillna(0)
        data['qs_summer_season_rmean3'] = data['quantity_sold'].rolling(3).mean().fillna(0)
        data['qs_summer_season_rmean4'] = data['quantity_sold'].rolling(4).mean().fillna(0)
        data['qs_summer_season_rmean5'] = data['quantity_sold'].rolling(5).mean().fillna(0)
        
        rolling_sum = rolling_sum.append(data, ignore_index=True)
    return_df = non_april_may_period.append(rolling_sum, ignore_index=True).fillna(0)
    return return_df

def create_features(dataset):
    df = dataset.copy()
    df.set_index('transaction_date', inplace=True)
    #df = df[['item_code', 'quantity_sold', 'total_item_price']]
    df['year'] = pd.DatetimeIndex(df.index).year
    df['quarter'] = pd.DatetimeIndex(df.index).quarter
    df['month'] = pd.DatetimeIndex(df.index).month
    df['day'] = pd.DatetimeIndex(df.index).day
    df['dayofweek'] = (pd.DatetimeIndex(df.index).dayofweek + 1)
    df['week'] = (((pd.to_numeric(pd.DatetimeIndex(df.index).day, errors='coerce') / 7) + 1).astype(int))
    return df.reset_index()