# import re module
# re module provides support
# for regular expressions
import re, os
from .secrets import ALLOWED_EXTENSIONS
from datetime import timedelta, date, datetime
import pandas as pd
import numpy as np
from . import db
from .models import *
from werkzeug.utils import secure_filename
from .feature_engineering import *
from .secrets import *

# FOR FORECASTING
import pandas as pd
import numpy as np
from datetime import date, datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models
from pandas.tseries.offsets import DateOffset, QuarterEnd
import types
from sqlalchemy.exc import OperationalError, ProgrammingError
import math
import pickle



# Define a function for
# for validating an Email
def check_email(email):
    # Make a regular expression
    # for validating an Email
    regex = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'

    # pass the regular expression
    # and the string into the fullmatch() method
    if(re.fullmatch(regex, email)):
        return True
    else:
        return False

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def generate_secret_key():
    pass
    PERMANENT_SESSION_LIFETIME = timedelta(minutes=30)
    SESSION_REFRESH_EACH_REQUEST = True

def treat_outliers(dataset, items):
    dataset = dataset.copy()
    return_df = pd.DataFrame()
    for item in items:
        data = list(dataset[dataset['item_code'] == item]['quantity_sold'])
        Q1 = np.percentile(data, 25) 
        Q2 = np.percentile(data, 50) 
        Q3 = np.percentile(data, 75) 
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        
        item_df = dataset[dataset['item_code'] == item][:]
        outliers = (item_df['item_code'] == item) & ((item_df['quantity_sold'] < lower_limit) | (item_df['quantity_sold'] > upper_limit))
        item_df.loc[outliers, 'quantity_sold'] = Q2
        return_df = return_df.append(item_df, ignore_index=True)
    return return_df

def check_duplicate_data(file_check, branch_id):
    date_today = (date.today())

    forecasts_data = db.session.query(Forecast_data).filter(Forecast_data.branch_id == branch_id).order_by(Forecast_data.transaction_date.asc()).all()
    # file_path_check = file_check
    # file_path_check = os.path.join(os.path.abspath(os.path.dirname(__file__)),TEMPS,secure_filename(datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p") + '_' + file_check.filename))
    # file_check.save(file_path_check) # Then save the raw file

    dataset = pd.read_csv(file_check, index_col=False)

    dataset = dataset[dataset['qtysold'] > 0]
    dataset_new = dataset.groupby(['trandate', 'itemcode', 'description']).sum().reset_index()
    dataset_new.rename(columns = {
        'trandate':'transaction_date',
        'itemcode':'item_code',
        'description': 'item_description',
        'qtysold':'quantity_sold',
        'qtyreturn':'quantity_return',
        'unitprice':'item_price',
        'lineextprice':'total_item_price',
        'itemcost':'item_cost',
        'lineextcost':'total_item_cost'
    }, inplace=True)
    dataset_new['transaction_date'] = pd.DatetimeIndex(dataset_new.transaction_date)
    dataset_new = dataset_new.groupby(['item_code', 'item_description', pd.Grouper(key="transaction_date", freq="W")]).sum().reset_index()
    data_2018_2021 = dataset_new.loc[(pd.DatetimeIndex(dataset_new['transaction_date']).year >= 2018) & (pd.DatetimeIndex(dataset_new['transaction_date']).year <= (date_today.year - 1)) ]
    
    duplicate_flag = False

    for forecast in forecasts_data:
        # print(forecast.transaction_date)
        # print(forecast.item_code)
        # print('=================')
        # print(data_2018_2021.iloc[0, 0]) #item code
        # print(str(data_2018_2021.iloc[0, 2])[:10]) #transaction date

        if (str(forecast.transaction_date) == str(data_2018_2021.iloc[0, 2])[:10] and int(forecast.item_code) == int(data_2018_2021.iloc[0, 0])):
            duplicate_flag = True
            os.remove(file_check)
            break
    
    return duplicate_flag

def data_preprocessing(filename, is_retraining_data=False):
    date_today = (date.today())
    # PREPROCESSING OF DATA
    dataset = pd.read_csv(filename, index_col=False)
    
    dataset = dataset[dataset['qtysold'] > 0]
    dataset_new = dataset.groupby(['trandate', 'itemcode', 'description']).sum().reset_index()
    dataset_new.rename(columns = {
        'trandate':'transaction_date',
        'itemcode':'item_code',
        'description': 'item_description',
        'qtysold':'quantity_sold',
        'qtyreturn':'quantity_return',
        'unitprice':'item_price',
        'lineextprice':'total_item_price',
        'itemcost':'item_cost',
        'lineextcost':'total_item_cost'
    }, inplace=True)
    dataset_new['transaction_date'] = pd.DatetimeIndex(dataset_new.transaction_date)
    dataset_new = dataset_new.groupby(['item_code', 'item_description', pd.Grouper(key="transaction_date", freq="W")]).sum().reset_index()

    if (is_retraining_data):
        date_today = (date.today())
        data_2018_2021 = dataset_new.loc[(pd.DatetimeIndex(dataset_new['transaction_date']).year >= date_today.year)]
    else:
        data_2018_2021 = dataset_new.loc[(pd.DatetimeIndex(dataset_new['transaction_date']).year >= 2018) & (pd.DatetimeIndex(dataset_new['transaction_date']).year <= (date_today.year - 1)) ]
    
    items = list(data_2018_2021['item_code'].unique())
    new_data = treat_outliers(data_2018_2021, items)

    # ALL PRODUCTS
    # products = new_data.groupby(['item_code', 'item_description']).sum().reset_index()
    # all_products = np.asarray(products[['item_code', 'item_description']])

    #TOP 500 PRODUCTS
    # pivot_data = new_data.pivot_table(index="item_code", values="quantity_sold", aggfunc="sum").reset_index().sort_values(by='quantity_sold', ascending=False)
    # topitems_df = pivot_data.nlargest(500, columns=['quantity_sold'])

    # topitems_df = (topitems_df.item_code.unique())
    # dframe = new_data[new_data['item_code'].isin(topitems_df)]
    # products = dframe.groupby(['item_code', 'item_description']).sum().reset_index()
    # all_products = np.asarray(products[['item_code', 'item_description']])

    
    # AT LEAST 52 RECORDS PER PRODUCT
    if (is_retraining_data):
        date_today = (date.today())
        df_2020_2021 = new_data.loc[(pd.DatetimeIndex(new_data['transaction_date']).year >= date_today.year)]
    else:
        df_2020_2021 = new_data.loc[(pd.DatetimeIndex(new_data['transaction_date']).year >= 2020)] # For Products Selections
    weekly_transaction_count = df_2020_2021[['item_code']].value_counts().reset_index(name="count")


    if (is_retraining_data):
        product_df = np.array(weekly_transaction_count[weekly_transaction_count['count'] >= 45]['item_code'])
    else:
        product_df = np.array(weekly_transaction_count[weekly_transaction_count['count'] >= 52]['item_code'])
    

    dframe = df_2020_2021[df_2020_2021['item_code'].isin(product_df)]
    products = dframe.groupby(['item_code', 'item_description']).sum().reset_index()

    if products.empty:
        return pd.DataFrame(), []

    #TOP 500 PRODUCTS

    '''
        Condition if at least 52 records is not satisfied
    '''
    # ntop_products = 500
    if not len(product_df) >= 500:
        ntop_products = len(product_df)
    else:
        ntop_products = len(product_df)
    '''
        Condition if at least 52 records is not satisfied
    '''
    pivot_data = products.pivot_table(index="item_code", values="quantity_sold", aggfunc="sum").reset_index().sort_values(by='quantity_sold', ascending=False)
    topitems_df = pivot_data.nlargest(ntop_products, columns=['quantity_sold'])
    topitems_df = (topitems_df.item_code.unique())
    dframe2 = products[products['item_code'].isin(topitems_df)]
    products2 = dframe2.groupby(['item_code', 'item_description']).sum().reset_index()
    top500_products = np.asarray(products2[['item_code', 'item_description']])
    return dframe, top500_products

def validation_data_preprocessing(filename):
    dataset = pd.read_csv(filename, index_col=False)

    dataset.dropna(inplace=True)
    dataset = dataset[dataset['qtysold'] > 0]
    dataset_new = dataset.groupby(['trandate', 'itemcode', 'description']).sum().reset_index()
    dataset_new.rename(columns = {
        'trandate':'transaction_date',
        'itemcode':'item_code',
        'description': 'item_description',
        'qtysold':'quantity_sold',
        'qtyreturn':'quantity_return',
        'unitprice':'item_price',
        'lineextprice':'total_item_price',
        'itemcost':'item_cost',
        'lineextcost':'total_item_cost'
    }, inplace=True)
    dataset_new['transaction_date'] = pd.DatetimeIndex(dataset_new.transaction_date)
    dataset_new = dataset_new.groupby(['item_code', 'item_description', pd.Grouper(key="transaction_date", freq="W")]).sum().reset_index()
    items = list(dataset_new['item_code'].unique())
    new_data = treat_outliers(dataset_new, items)
    return new_data

def timeseries_generator(dataset, lookback, future):
    trainX = []
    trainY = []

    n_future = future  # Number of days we want to look into the future based on the past days.
    n_past = lookback  # Number of past days we want to use to predict the future.
    for i in range(n_past, len(dataset) - n_future +1):
        trainX.append(dataset[i - n_past:i, 1:dataset.shape[1]])
        trainY.append(dataset[i + n_future - 1:i + n_future, 0])
        #print(dataset[i - n_past:i, 1:dataset.shape[1]])
        #print(dataset[i + n_future - 1:i + n_future, 0])
    trainX, trainY = np.array(trainX), np.array(trainY)
    
    return trainX, trainY

def feature_engineering(file_name, branch_id):
    dataset = pd.read_csv(file_name, parse_dates=['transaction_date'], index_col=False)
    
    items = list(dataset['item_code'].unique()) # Used for checking existing products

    df1 = fe_lag(dataset, items)
    df2 = fe_rolling_sum(df1, items)
    df3 = fe_rolling_mean(df2, items)
    seasons(df3)
    df4 = create_features(df3)
    df5 = fe_expanding_mean(df4, items)
    df6 = fe_rolling_std(df5, items)
    df7 = pandemic_time(df6)
    df8 = non_rainy_season(df7)
    df9 = bermonths_time(df8)
    df10 = summer_season(df9)
    df10['branch_id'] = branch_id
    df10_sorted = df10.sort_values(by='transaction_date')

    return df10_sorted







def prepare_dataset(file_path):
    dataset = pd.read_csv(file_path, parse_dates=['transaction_date'])
    dataset.set_index('transaction_date', inplace=True)
    items_df = dataset.groupby(['item_code', 'item_description']).sum().reset_index()
    items = list(items_df.item_code)
    item_description = list(items_df.item_description)
    first_column = dataset.pop('quantity_sold')
    dataset.insert(0, 'quantity_sold', first_column)

    second_column = dataset.pop('is_pandemic')
    dataset.insert(1, 'is_pandemic', second_column)

    third_column = dataset.pop('ber_month')
    dataset.insert(1, 'ber_month', third_column)

    fourth_column = dataset.pop('branch_id')
    dataset.insert(1, 'branch_id', fourth_column)

    fifth_column = dataset.pop('rainy_season')
    dataset.insert(1, 'rainy_season', fifth_column)

    sixth_column = dataset.pop('march_may_period')
    dataset.insert(1, 'march_may_period', fifth_column)

    # dataset.drop(columns=['item_description'], inplace=True)
    dataset.drop(columns=['item_description', 'year', 'quarter', 'month', 'day', 'dayofweek', 'week', 'is_pandemic', 'ber_month', 'rainy_season', 'march_may_period', 'summer_season'], inplace=True)
    return dataset

def get_totalqty_itemprice(list_quantity, branch_id, product_code):
    total_quantity = 0
    for i in range(len(list_quantity)):
        total_quantity = total_quantity + round(list_quantity[i])

    # GET PRODUCT PRICE
    item = db.session.query(Forecast_data).filter(Forecast_data.branch_id == int(branch_id), Forecast_data.item_code == int(product_code)).order_by(Forecast_data.transaction_date.desc()).first()
    
    return total_quantity, item.item_price, item.item_cost

def get_product_name(branch_id, product_code):
    # GET PRODUCT PRICE
    item = db.session.query(Forecast_data).filter(Forecast_data.branch_id == int(branch_id), Forecast_data.item_code == int(product_code)).first()
    return item.item_description

def read_test_dataset(all_models='current_used_model', model_id=None):
    final_data_obj = db.session.query(FinalData).filter(FinalData.is_active == 1).first()
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),MERGED_DATA_FOLDER,secure_filename(final_data_obj.csv_filename))
    # df_training = dataset.loc[dataset.index.year <= 2019]

    '''
        Determined Used Model
    '''
    if all_models == 'current_used_model':
        algorithm_ = db.session.query(Forecasting_model).filter(Forecasting_model.is_used == 1).first()
        if algorithm_.is_retrained == True:
            _csv_filename = algorithm_.csv_filename
            file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),MERGED_DATA_FOLDER,secure_filename(_csv_filename))
        
        dataset = prepare_dataset(file_path)
        df_test = dataset.loc[dataset.index.year >= int(algorithm_.start_testing_data)]

    elif all_models == 'all_models':
        algorithm_ = db.session.query(Forecasting_model).filter(Forecasting_model.id == model_id).first()
        if algorithm_.is_retrained == True:
            _csv_filename = algorithm_.csv_filename
            file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),MERGED_DATA_FOLDER,secure_filename(_csv_filename))
        dataset = prepare_dataset(file_path)
        df_test = dataset.loc[dataset.index.year >= int(algorithm_.start_testing_data)]
        # df_test = dataset.loc[dataset.index.year >= int(algorithm_.start_testing_data)]
    return df_test

def read_validation_dataset(branch_id):
    validation_obj = db.session.query(Validation_data).filter(Validation_data.branch_id == branch_id, Validation_data.is_used == 1).first()

    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), VALIDATION_DATA, secure_filename(validation_obj.file_name))
    dataset = pd.read_csv(file_path, parse_dates=['transaction_date'])

    return dataset

def read_model(all_models='current_used_model'):
    if all_models == 'current_used_model':
        algorithm_ = db.session.query(Forecasting_model).filter(Forecasting_model.is_used == 1).first()
        file_path = MODELS + "\\" + algorithm_.algorithm_name + "\\" + algorithm_.training_data
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),file_path,secure_filename(algorithm_.model_name))
        
        if algorithm_.algorithm_name == 'Support Vector Regression':
            selected_model = pickle.load(open(model_path, 'rb'))
        else:
            selected_model = models.load_model(model_path)

        return selected_model, algorithm_.algorithm_name

    elif all_models == 'all_models':
        # Retrieve all models
        forecasting_models = Forecasting_model.query.all()
        c = 0
        for forecasting_model in forecasting_models:
            algorithm_ = db.session.query(Forecasting_model).filter(Forecasting_model.id == forecasting_model.id).first()
            file_path = MODELS + "\\" + algorithm_.algorithm_name + "\\" + algorithm_.training_data
            model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),file_path,secure_filename(algorithm_.model_name))
            
            if algorithm_.algorithm_name == 'Support Vector Regression':
                selected_model = pickle.load(open(model_path, 'rb'))
            else:
                selected_model = models.load_model(model_path)
            all_models_forecast(forecasting_model.id, selected_model, algorithm_.algorithm_name)
            c = c + 1
        db.session.commit()
    

# def rearrange_columns(dataset):
#     first_column = dataset.pop('quantity_sold')
#     dataset.insert(0, 'quantity_sold', first_column)

#     second_column = dataset.pop('is_pandemic')
#     dataset.insert(1, 'is_pandemic', second_column)

#     third_column = dataset.pop('ber_month')
#     dataset.insert(1, 'ber_month', third_column)

#     fourth_column = dataset.pop('branch_id')
#     dataset.insert(1, 'branch_id', fourth_column)

#     fifth_column = dataset.pop('rainy_season')
#     dataset.insert(1, 'rainy_season', fifth_column)

#     sixth_column = dataset.pop('march_may_period')
#     dataset.insert(1, 'march_may_period', fifth_column)
#     return dataset

def prediction_results(main_data, frequency, product_code, branch_id, selected_model, algorithm_name):
    
    # PRODUCT DETAILS
    item_id = int(product_code)
    branch_id = int(branch_id)
    item_df = main_data[(main_data['item_code'] == item_id) & (main_data['branch_id'] == branch_id)]

    if not item_df.empty and item_df.shape[0] >= 45:
        scaler_df = StandardScaler()
        scaled_df = scaler_df.fit_transform(item_df)
        X, y = timeseries_generator(scaled_df, 1, 1)
        
        # PREDICTIONS
        if algorithm_name == 'Artificial Neural Network':
            prediction = selected_model.predict(X[-X.shape[0]:])
            prediction_copies = np.repeat(prediction, item_df.shape[1], axis=-1)
            reshape_predictions = (prediction_copies.reshape(prediction_copies.shape[0], prediction_copies.shape[2]))
            predicted_values = scaler_df.inverse_transform(reshape_predictions)[:,0]

            # START TO DATAFRAME
            start_forecast_date = (item_df.index.max())
            # print('Start Date: ' + str(start_forecast_date))
            future_df = pd.DataFrame()

            # print(start_forecast_date)

            future_df['transaction_date'] = [start_forecast_date + DateOffset(weeks=x) for x in range(1,len(predicted_values) + 1)]
            future_df.sort_values(by='transaction_date', ascending=True)
            future_df['quantity_sold'] = list(predicted_values)
            future_df['item_code'] = item_id

            # df = item_df.reset_index()
            df_1 = future_df.groupby(['item_code', pd.Grouper(key="transaction_date", freq=frequency)]).sum().reset_index()

            # START DATE | CURRENT DATE
            date_today = str(date.today()) #YYYY-MM-DD
            forecasted_df = df_1[(df_1['transaction_date'] >= date_today)]

        elif algorithm_name == 'Recurrent Neural Network':
            prediction = selected_model.predict(X[-X.shape[0]:])
            prediction_copies = np.repeat(prediction, item_df.shape[1], axis=-1)
            predicted_values = scaler_df.inverse_transform(prediction_copies)[:,0]

            # START TO DATAFRAME
            start_forecast_date = (item_df.index.max())
            future_df = pd.DataFrame()

            future_df['transaction_date'] = [start_forecast_date + DateOffset(weeks=x) for x in range(1,len(predicted_values) + 1)]
            future_df.sort_values(by='transaction_date', ascending=True)
            future_df['quantity_sold'] = list(predicted_values)
            future_df['item_code'] = item_id

            # df = item_df.reset_index()
            df_1 = future_df.groupby(['item_code', pd.Grouper(key="transaction_date", freq=frequency)]).sum().reset_index()

            # START DATE | CURRENT DATE
            date_today = str(date.today()) #YYYY-MM-DD
            forecasted_df = df_1[(df_1['transaction_date'] >= date_today)]

        elif algorithm_name == 'Support Vector Regression':
            features_data = scaled_df[:, 1:]
            target_data   = scaled_df[:, 0:1]

            prediction = selected_model.predict(features_data)
            svr_predictions = pd.concat([pd.DataFrame(prediction),pd.DataFrame(scaled_df[:, 1:][:])], axis=1)
            reverse_predictions = scaler_df.inverse_transform(svr_predictions)

            predictions_df = item_df[reverse_predictions[:, 0].shape[0]*-1:]
            predictions_df = predictions_df.copy()
            predictions_df['predicted']=reverse_predictions[:, 0]

            # START TO DATAFRAME
            start_forecast_date = (item_df.index.max())
            future_df = pd.DataFrame()

            future_df['transaction_date'] = [start_forecast_date + DateOffset(weeks=x) for x in range(1,predictions_df.shape[0] + 1)]
            future_df.sort_values(by='transaction_date', ascending=True)
            future_df['quantity_sold'] = list(predictions_df['predicted'])
            future_df['item_code'] = item_id

            future_df.drop(future_df.tail(1).index,inplace=True)

            # df = item_df.reset_index()
            df_1 = future_df.groupby(['item_code', pd.Grouper(key="transaction_date", freq=frequency)]).sum().reset_index()

            # START DATE | CURRENT DATE
            date_today = str(date.today()) #YYYY-MM-DD
            forecasted_df = df_1[(df_1['transaction_date'] >= date_today)]
        return forecasted_df

def data_frequency(request_frequency):
    # Frequency
    frequency = ''
    if(str(request_frequency) == 'weekly'):
        frequency = 'W'
    elif (str(request_frequency) == 'monthly'):
        frequency = 'M'
    elif (str(request_frequency) == 'quarterly'):
        frequency = 'Q'
    elif (str(request_frequency) == 'yearly'):
        frequency = 'Y'
    return frequency

def forecast_all_products():
    # Check for forecasted_products
    current_model = db.session.query(Forecasting_model).filter(Forecasting_model.is_used == True).first()
    all_forecasted_products = db.session.query(Forecasted_products).filter(Forecasted_products.model_id == current_model.id).all()

    forecasted_products_records = db.session.query(Forecasted_products).all()
    # get_product_name(2, 16)
    if(forecasted_products_records):
        all_forecasted_products_dict = {}
        c = 0
        for product_record in all_forecasted_products:
            df = pd.DataFrame.from_dict(product_record.forecast_data, orient='index')
            df.reset_index(inplace=True)
            df.rename(columns = {'index':'transaction_date', 0 : 'quantity_sold'}, inplace = True)
            df['transaction_date'] = pd.to_datetime(df['transaction_date'])
            
            # START DATE | CURRENT DATE
            date_today = str(date.today()) #YYYY-MM-DD
            df = df[(df['transaction_date'] >= date_today)]

            if current_model.algorithm_name == 'Support Vector Regression':
                df.drop(df.tail(1).index,inplace=True)
            
            dates = list(df['transaction_date'])
            quantity = list(df['quantity_sold'])
            
            # ITEM DICTIONARY
            product_prediction = {}
            for j in range(len(dates)):
                product_prediction.update({str(dates[j]): (str(quantity[j]))})

            qty, price, cost = get_totalqty_itemprice([], product_record.branch_id, product_record.product_code)

            # print(cost * len(dates))
            
            product_name = get_product_name(product_record.branch_id, product_record.product_code)
            item_dict = {}
            item_dict.update({
                'product_code': product_record.product_code,
                'branch_id': product_record.branch_id,
                'product_data': product_prediction,
                'product_name': product_name,
                'product_price': price,
                'product_cost': cost

            })

            all_forecasted_products_dict.update({c: item_dict})
            c = c + 1
            if c == 20:
                break
        return json.dumps(all_forecasted_products_dict)
    else:
        branches_ = Branches.query.all()
        if branches_:
            # Save Prediction Per Products
            # Per MODEL
            read_model('all_models')

            current_model = db.session.query(Forecasting_model).filter(Forecasting_model.is_used == 1).first()
            all_forecasted_products = db.session.query(Forecasted_products).filter(Forecasted_products.model_id == current_model.id).all()
            
            all_forecasted_products_dict = {}
            c = 0
            for product_record in all_forecasted_products:
                qty, price, cost = get_totalqty_itemprice([], product_record.branch_id, product_record.product_code)
                product_name = get_product_name(product_record.branch_id, product_record.product_code)
                item_dict = {}
                item_dict.update({
                    'product_code': product_record.product_code,
                    'branch_id': product_record.branch_id,
                    'product_data': product_record.forecast_data,
                    'product_name': product_name,
                    'product_price': price

                })

                all_forecasted_products_dict.update({c: item_dict})
                c = c + 1
                if c == 20:
                    break
            return json.dumps(all_forecasted_products_dict)
            # return all_forecasted_products_dict
    
def all_models_forecast(model_id, selected_model, algorithm_name):
    #Read Dataset
    dataset = read_test_dataset(all_models='all_models', model_id=model_id)

    # # Read Model
    # selected_model, algorithm_name = read_model()

    # START DATE | CURRENT DATE
    date_today = str(date.today()) #YYYY-MM-DD

    # Frequency
    frequency = 'W'

    # Get all models
    forecasting_model = Forecasting_model.query.all()

    #All Products
    all_products = db.session.query(Products).all()

    # Initialize Counter
    c = 0
    for product in all_products:
        forecast_ = prediction_results(dataset, frequency, product.product_code, product.branch_id, selected_model, algorithm_name)
        if(type(forecast_) != types.NoneType):
            dates = list(forecast_['transaction_date'])
            quantity = list(forecast_['quantity_sold'])
            # ITEM DICTIONARY
            product_prediction = {}
            for j in range(len(dates)):
                product_prediction.update({str(dates[j]): str(round(quantity[j]))})

            if (len(dates) == 0):
                continue
            else:
                # _product_prediction = str(product_prediction).replace('"', "'")
                # _product_prediction = str(product_prediction)
                insert_product_forecast = Forecasted_products(product.product_code, product.branch_id, product_prediction, model_id)
                db.session.add(insert_product_forecast)
        c = c + 1
        # if c == 7:
        #     break