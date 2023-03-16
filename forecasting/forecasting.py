from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify, make_response, session
from flask_login import login_required, current_user
from .classes import UploadCSV
from werkzeug.utils import secure_filename
import os
from .secrets import UPLOAD_FOLDER, PROFILE_PICTURES, LAST_FILE_UPLOADED, MERGED_DATA_FOLDER, MODELS, VALIDATION_DATA
from datetime import datetime
# from .models import User, UserSchema, Permissions, PermissionsSchema, Uploads, UploadsSchema, Forecast_data, ForecastDataSchema, Branches, BranchesSchema, Products, ProductsSchema, Forecasted_products, Forecasted_productsSchema, Forecasting_model, ForecastingModelSchema
from .models import *
import json
from . import db, conn
from .functions import *
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import func
from sqlalchemy import delete
from werkzeug.security import generate_password_hash
import csv
from sqlalchemy import create_engine, text # for truncating table
from .views import permissions
import numpy as np
import pickle

# FOR FORECASTING
import pandas as pd
import numpy as np
from datetime import date, datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# from tensorflow.keras import metrics, callbacks, initializers
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, LeakyReLU
# from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models

# from sklearn.model_selection import train_test_split
# from keras.preprocessing.sequence import TimeseriesGenerator
# from keras.optimizers import Adam
# import tensorflow as tf
# from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from pandas.tseries.offsets import DateOffset, QuarterEnd

import types
from sqlalchemy.exc import OperationalError, ProgrammingError
import math

forecast = Blueprint('forecast', __name__)


@forecast.route('/forecasting')
@login_required
def forecasting():
    
    session['current_menu'] = request.path # for active menu

    products = Products.query.order_by(Products.branch_id.asc()).all()
    products_schema = ProductsSchema(many=True)

    branches = Branches.query.all()
    branches_schema = BranchesSchema(many=True)

    forecasting_model = Forecasting_model.query.filter_by(is_active = 1).all()
    forecasting_schema = ForecastingModelSchema(many=True)

    current_model = db.session.query(Forecasting_model).filter(Forecasting_model.is_used == 1).first()

    all_forecasts = forecast_all_products()
    return render_template("modules/forecasting/forecasting.html", auth_user=current_user, permissions=permissions(), products=products_schema.dump(products), branches=branches_schema.dump(branches), all_forecasts=all_forecasts, forecast_models = forecasting_schema.dump(forecasting_model), current_model=current_model.algorithm_name, model_accuracy=current_model.model_accuracy, model_retrained=current_model.is_retrained)

@forecast.route('/forecast_data', methods=['POST'])
@login_required
def forecast_data():
    # Read Dataset
    df_test = read_test_dataset()

    # Read Model
    selected_model, algorithm_name = read_model()
    
    # Features
    # features = list(df_test.columns)

    # 
    # forecast specific item
    #
    item_id = int(request.form['product_code'])
    branch_id = int(request.form['branch_id'])

    item_df = df_test[(df_test['item_code'] == item_id) & (df_test['branch_id'] == branch_id)]
    scaler_df = StandardScaler()
    scaled_df = scaler_df.fit_transform(item_df)

    X, y = timeseries_generator(scaled_df, 1, 1)
    
    if algorithm_name == 'Artificial Neural Network':
        prediction = selected_model.predict(X[-X.shape[0]:])
        prediction_copies = np.repeat(prediction, item_df.shape[1], axis=-1)
        reshape_predictions = (prediction_copies.reshape(prediction_copies.shape[0], prediction_copies.shape[2]))
        predicted_values = scaler_df.inverse_transform(reshape_predictions)[:,0]

        start_forecast_date = (item_df.index.max())
        future_df = pd.DataFrame()
        future_df['transaction_date'] = [start_forecast_date + DateOffset(weeks=x) for x in range(0,len(predicted_values))]
        future_df.sort_values(by='transaction_date', ascending=True)
        future_df['quantity_sold'] = list(predicted_values)

        # filter dates for dislaying
        start_date = str(request.form['start_date'])
        end_date = str(request.form['end_date'])
        forecast_start_date = (start_date[-4:]) +'-'+ (start_date[0:2]) +'-'+ (start_date[3:5])
        forecast_end_date = (end_date[-4:]) +'-'+ (end_date[0:2]) +'-'+ (end_date[3:5])
        filter_df = future_df[(future_df['transaction_date'] >= forecast_start_date) & (future_df['transaction_date'] <= forecast_end_date)]
        # filter dates for dislaying

        dates = list(filter_df['transaction_date'])
        quantity = list(filter_df['quantity_sold'])
        
        pred_dates = {}
        for i in range(len(dates)):
            pred_dates.update({str(i): dates[i]})
        
        pred_quantity = {}
        for i in range(len(dates)):
            pred_quantity.update({str(i): round(quantity[i])})

        total_quantity, item_price, item_cost = get_totalqty_itemprice(quantity, request.form['branch_id'], request.form['product_code'])
        
        pred_dict = {}
        pred_dict.update({'prediction_date': pred_dates})
        pred_dict.update({'quantity': pred_quantity})
        pred_dict.update({'product_name': request.form['product_name']})
        pred_dict.update({'forecasted_weeks': filter_df.shape[0]})
        pred_dict.update({'estimated_sales': round(total_quantity * item_price)})
        pred_dict.update({'estimated_revenue': round(total_quantity * item_price) - round(total_quantity * item_cost)})

        return pred_dict

    elif algorithm_name == 'Recurrent Neural Network':
        prediction = selected_model.predict(X[-X.shape[0]:])
        prediction_copies = np.repeat(prediction, item_df.shape[1], axis=-1)
        predicted_values = scaler_df.inverse_transform(prediction_copies)[:,0]

        start_forecast_date = (item_df.index.max())
        future_df = pd.DataFrame()
        future_df['transaction_date'] = [start_forecast_date + DateOffset(weeks=x) for x in range(0,len(predicted_values))]
        future_df.sort_values(by='transaction_date', ascending=True)
        future_df['quantity_sold'] = list(predicted_values)

        # filter dates for dislaying
        start_date = str(request.form['start_date'])
        end_date = str(request.form['end_date'])
        forecast_start_date = (start_date[-4:]) +'-'+ (start_date[0:2]) +'-'+ (start_date[3:5])
        forecast_end_date = (end_date[-4:]) +'-'+ (end_date[0:2]) +'-'+ (end_date[3:5])
        filter_df = future_df[(future_df['transaction_date'] >= forecast_start_date) & (future_df['transaction_date'] <= forecast_end_date)]
        # filter dates for dislaying

        dates = list(filter_df['transaction_date'])
        quantity = list(filter_df['quantity_sold'])
        
        pred_dates = {}
        for i in range(len(dates)):
            pred_dates.update({str(i): dates[i]})
        
        pred_quantity = {}
        for i in range(len(dates)):
            pred_quantity.update({str(i): round(quantity[i])})

        total_quantity, item_price, item_cost = get_totalqty_itemprice(quantity, request.form['branch_id'], request.form['product_code'])
        
        pred_dict = {}
        pred_dict.update({'prediction_date': pred_dates})
        pred_dict.update({'quantity': pred_quantity})
        pred_dict.update({'product_name': request.form['product_name']})
        pred_dict.update({'forecasted_weeks': filter_df.shape[0]})
        pred_dict.update({'estimated_sales': round(total_quantity * item_price)})
        pred_dict.update({'estimated_revenue': round(total_quantity * item_price) - round(total_quantity * item_cost)})

        return pred_dict
    elif algorithm_name == 'Support Vector Regression':
        features_data = scaled_df[:, 1:]
        target_data   = scaled_df[:, 0:1]

        prediction = selected_model.predict(features_data)
        svr_predictions = pd.concat([pd.DataFrame(prediction),pd.DataFrame(scaled_df[:, 1:][:])], axis=1)
        reverse_predictions = scaler_df.inverse_transform(svr_predictions)

        predictions_df = item_df[reverse_predictions[:, 0].shape[0]*-1:]
        predictions_df = predictions_df.copy()
        predictions_df['predicted']=reverse_predictions[:, 0]

        start_forecast_date = (item_df.index.max())
        future_df = pd.DataFrame()
        future_df['transaction_date'] = [start_forecast_date + DateOffset(weeks=x) for x in range(0,predictions_df.shape[0])]
        future_df.sort_values(by='transaction_date', ascending=True)
        future_df['quantity_sold'] = list(predictions_df['predicted'])

        # # filter dates for dislaying
        start_date = str(request.form['start_date'])
        end_date = str(request.form['end_date'])
        forecast_start_date = (start_date[-4:]) +'-'+ (start_date[0:2]) +'-'+ (start_date[3:5])
        forecast_end_date = (end_date[-4:]) +'-'+ (end_date[0:2]) +'-'+ (end_date[3:5])
        filter_df = future_df[(future_df['transaction_date'] >= forecast_start_date) & (future_df['transaction_date'] <= forecast_end_date)]

        dates = list(filter_df['transaction_date'])
        quantity = list(filter_df['quantity_sold'])

        pred_dates = {}
        for i in range(len(dates)):
            pred_dates.update({str(i): dates[i]})
        
        pred_quantity = {}
        for i in range(len(dates)):
            pred_quantity.update({str(i): round(quantity[i])})

        total_quantity, item_price, item_cost = get_totalqty_itemprice(quantity, request.form['branch_id'], request.form['product_code'])
        
        pred_dict = {}
        pred_dict.update({'prediction_date': pred_dates})
        pred_dict.update({'quantity': pred_quantity})
        pred_dict.update({'product_name': request.form['product_name']})
        pred_dict.update({'forecasted_weeks': filter_df.shape[0]})
        pred_dict.update({'estimated_sales': round(total_quantity * item_price)})
        pred_dict.update({'estimated_revenue': round(total_quantity * item_price) - round(total_quantity * item_cost)})

        return pred_dict

@forecast.route('/change_frequency', methods=['POST'])
@login_required
def change_frequency():
    # Read Dataset
    df_test = read_test_dataset()

    # Read Model
    selected_model, algorithm_name = read_model()

    # Frequency
    frequency = data_frequency(request.form['frequency'])
    
    # FORECASTING
    forecasted_df = prediction_results(df_test, frequency, request.form['product_code'], request.form['branch_id'], selected_model, algorithm_name)

    # SAVE TO DICTIONARY
    dates = list(forecasted_df['transaction_date'])
    quantity = list(forecasted_df['quantity_sold'])
    
    pred_dates = {}
    for i in range(len(dates)):
        pred_dates.update({str(i): dates[i]})
    
    pred_quantity = {}
    for i in range(len(dates)):
        pred_quantity.update({str(i): round(quantity[i])})

    total_quantity, item_price, item_cost = get_totalqty_itemprice(quantity, request.form['branch_id'], request.form['product_code'])
    
    pred_dict = {}
    pred_dict.update({'prediction_date': pred_dates})
    pred_dict.update({'quantity': pred_quantity})
    pred_dict.update({'product_name': request.form['product_name']})
    pred_dict.update({'forecasted_weeks': forecasted_df.shape[0]})
    pred_dict.update({'estimated_sales': round(total_quantity * item_price)})
    pred_dict.update({'estimated_revenue': round(total_quantity * item_price) - round(total_quantity * item_cost)})

    return pred_dict

@forecast.route('/compare_products', methods=['POST'])
@login_required
def compare_products():
    # Read Dataset
    df_test = read_test_dataset()

    # Read Model
    selected_model, algorithm_name = read_model()

    # Frequency
    frequency = data_frequency(request.form['frequency'])

    # print(type(request.form['product_code_comp']))
    # FORECASTING PRODUCT 1
    forecasted_df = prediction_results(df_test, frequency, request.form['product_code'], request.form['branch_id'], selected_model, algorithm_name)
    
    # FORECASTING PRODUCT 2
    forecasted_df2 = prediction_results(df_test, frequency, request.form['product_code_comp'], request.form['branch_id_comp'], selected_model, algorithm_name)

    # SAVE TO DICTIONARY
    if (forecasted_df.transaction_date.max() > forecasted_df2.transaction_date.max()):
        dates = list(forecasted_df['transaction_date'])
    else:
        dates = list(forecasted_df2['transaction_date'])
    
    quantity_product1 = list(forecasted_df['quantity_sold'])
    quantity_product2 = list(forecasted_df2['quantity_sold'])
    
    pred_dates = {}
    for i in range(len(dates)):
        pred_dates.update({str(i): (dates[i])})
    
    pred_product1 = {}
    for i in range(len(quantity_product1)):
        pred_product1.update({str(i): round(quantity_product1[i])})
    
    pred_product2 = {}
    for i in range(len(quantity_product2)):
        pred_product2.update({str(i): round(quantity_product2[i])})
    
    total_quantity, item_price, item_cost = get_totalqty_itemprice(quantity_product1, request.form['branch_id'], request.form['product_code'])
    total_quantity_comp, item_price_comp, item_cost_comp = get_totalqty_itemprice(quantity_product2, request.form['branch_id_comp'], request.form['product_code_comp'])

    forecasted_dict = {}
    forecasted_dict.update({'forecasted_dates': pred_dates})
    forecasted_dict.update({'quantity_1': pred_product1})
    forecasted_dict.update({'quantity_2': pred_product2})
    forecasted_dict.update({'product_names': {'product_name1': request.form['product_name'], 'product_name2': request.form['product_name_comp']}})
    forecasted_dict.update({'forecasted_weeks': {'number_weeks1': forecasted_df.shape[0], 'number_weeks2': forecasted_df2.shape[0]}})
    forecasted_dict.update({'estimated_sales': {'e_sales1': round(total_quantity * item_price), 'e_sales2': round(total_quantity_comp * item_price_comp)}})
    forecasted_dict.update({'estimated_revenue': {'e_revenue1': round(total_quantity * item_price) - round(total_quantity * item_cost), 'e_revenue2': round(total_quantity_comp * item_price_comp) - round(total_quantity_comp * item_cost_comp)}})
    
    # print(forecasted_df.transaction_date.max() > forecasted_df2.transaction_date.max())
    # print(forecasted_df.transaction_date.max(), forecasted_df2.transaction_date.max())

    return forecasted_dict

@forecast.route('/change_frequency_ap', methods=['POST'])
@login_required
def change_frequency_ap():
    product_df = pd.DataFrame()

    # Check for forecasted_products
    current_model = db.session.query(Forecasting_model).filter(Forecasting_model.is_used == 1).first()
    all_forecasted_products = db.session.query(Forecasted_products).filter(Forecasted_products.model_id == current_model.id, Forecasted_products.product_code == int(request.form['product_code']), Forecasted_products.branch_id == int(request.form['branch_id'])).first()
    # if(all_forecasted_products):
    #     all_forecasted_products_dict = {}
    #     c = 0
    #     for product_record in all_forecasted_products:
    #         qty, price, cost = get_totalqty_itemprice([], product_record.branch_id, product_record.product_code)
    #         product_name = get_product_name(product_record.branch_id, product_record.product_code)
    #         item_dict = {}
    #         item_dict.update({
    #             'product_code': product_record.product_code,
    #             'branch_id': product_record.branch_id,
    #             'product_data': product_record.forecast_data,
    #             'product_name': product_name,
    #             'product_price': price

    #         })

    #         all_forecasted_products_dict.update({c: item_dict})
    #         c = c + 1
    #         # if c == 5:
    #         #     break
    #     # return json.dumps(all_forecasted_products_dict)
    #     return_dict.update({'all_forecast': all_forecasted_products_dict})





    # print(all_forecasted_products.forecast_data)
    # date_list = request.form['product_data_dates'].split(",")
    # qty_list = request.form['product_data_quantity'].split(",")
    date_list = all_forecasted_products.forecast_data.keys()
    qty_list = all_forecasted_products.forecast_data.values()

    product_df['transaction_date'] = date_list
    product_df['quantity_sold'] = qty_list
    product_df['item_code'] = request.form['product_code']
    product_df['transaction_date'] = pd.to_datetime(product_df['transaction_date'])
    product_df['quantity_sold'] = product_df['quantity_sold'].astype(int)
    df_1 = product_df.groupby(['item_code', pd.Grouper(key="transaction_date", freq=request.form['frequency'])]).sum().reset_index()

    # Retrieve Product Name
    product_name = get_product_name(request.form['branch_id'], request.form['product_code'])
    # Retrieve Product Price
    qty, price, cost = get_totalqty_itemprice([], request.form['branch_id'], request.form['product_code'])
    
    # Return a dictionary
    dates = list(df_1['transaction_date'])
    qty = list(df_1['quantity_sold'])

    return_dict = {}
    return_dict.update({'transaction_date': dates})
    return_dict.update({'quantity_sold': qty})
    return_dict.update({'product_code': request.form['product_code']})
    return_dict.update({'branch_id': request.form['product_code']})
    return_dict.update({'product_name': product_name})
    return_dict.update({'price': price})
    return_dict.update({'cost': cost})

    return return_dict

@forecast.route('/change_algorithm', methods=['POST'])
def change_algorithm():
    # change flag is_use to 0
    update_current_forecast_model = db.session.query(Forecasting_model).filter(Forecasting_model.is_used == True).first()
    update_current_forecast_model.is_used = False
    

    update_forecast_model = Forecasting_model.query.filter_by(id=request.form['algorithm_id']).first()
    update_forecast_model.is_used = True

    db.session.commit()

    forecasting_model = Forecasting_model.query.all()
    forecasting_schema = ForecastingModelSchema(many=True)
    current_model = db.session.query(Forecasting_model).filter(Forecasting_model.is_used == True).first()
    forecasting_model_dict = {}
    c = 0
    for forecast_model in forecasting_model:
        forecasting_model_dict.update({c: {'id': forecast_model.id, 'algorithm_name': forecast_model.algorithm_name, 'model_name': forecast_model.model_name, 'is_retrained': forecast_model.is_retrained, 'training_data':forecast_model.training_data, 'start_testing_data': forecast_model.start_testing_data, 'model_accuracy': forecast_model.model_accuracy, 'is_used': forecast_model.is_used}})
        c = c + 1

    return_dict = {}
    return_dict.update({'forecast_model': forecasting_model_dict})
    return_dict.update({'current_model': current_model.algorithm_name})
    return_dict.update({'model_accuracy': current_model.model_accuracy})
    
    # Check for forecasted_products
    current_model = db.session.query(Forecasting_model).filter(Forecasting_model.is_used == 1).first()
    all_forecasted_products = db.session.query(Forecasted_products).filter(Forecasted_products.model_id == current_model.id).all()

    if(all_forecasted_products):
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
            # if c == 5:
            #     break
        # return json.dumps(all_forecasted_products_dict)
        return_dict.update({'all_forecast': all_forecasted_products_dict})
    return return_dict

@forecast.route('/model_validation', methods=['POST'])
def model_validation():
    '''
        ================================
                ACTUAL DATA 2022
        ================================
    '''
    if not (int(request.form['branch_id']) == 1):
        validation_dataset = read_validation_dataset(int(request.form['branch_id']))
    else:
        validation_dataset = read_validation_dataset(int(request.form['branch_id']))

    val_df = validation_dataset[(validation_dataset['item_code'] == int(request.form['product_code']))][['transaction_date', 'quantity_sold']]
    dates_actual = list(val_df['transaction_date'])
    quantity_actual = list(val_df['quantity_sold'])

    actual_dates_dict = {}
    for i in range(len(dates_actual)):
        actual_dates_dict.update({str(i): dates_actual[i]})
    
    actual_quantity_dict = {}
    for i in range(len(dates_actual)):
        actual_quantity_dict.update({str(i): round(quantity_actual[i])})
    

    '''
        ================================
                FORECAST 2022
        ================================
    '''

    df_test = read_test_dataset()
    # selected_model, algorithm_name = read_model()
    item_id = int(request.form['product_code'])
    branch_id = int(request.form['branch_id'])

    item_df = df_test[(df_test['item_code'] == item_id) & (df_test['branch_id'] == branch_id)]
    scaler_df = StandardScaler()
    scaled_df = scaler_df.fit_transform(item_df)

    X, y = timeseries_generator(scaled_df, 1, 1)

    forecasting_models = Forecasting_model.query.filter_by(is_active = 1).all()
    validation_data = {}
    for forecasting_model in forecasting_models:
        # algorithm_ = db.session.query(Forecasting_model).filter(Forecasting_model.id == forecasting_model.id).first()
        file_path = MODELS + "\\" + forecasting_model.algorithm_name + "\\" + forecasting_model.training_data
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),file_path,secure_filename(forecasting_model.model_name))
        
        if forecasting_model.algorithm_name == 'Support Vector Regression':
            selected_model = pickle.load(open(model_path, 'rb'))

            df_features = scaled_df[:, 1:]
            df_target   = scaled_df[:, 0:1]

            prediction = selected_model.predict(df_features)
            svr_predictions = pd.concat([pd.DataFrame(prediction),pd.DataFrame(scaled_df[:, 1:][:])], axis=1)
            reverse_svr_predictions = scaler_df.inverse_transform(svr_predictions)

            data_ = item_df[reverse_svr_predictions[:, 0].shape[0]*-1:]
            data_ = data_.copy()
            data_['predicted']=reverse_svr_predictions[:, 0]

            start_forecast_date = (item_df.index.max())
            future_df = pd.DataFrame()
            future_df['transaction_date'] = [start_forecast_date + DateOffset(weeks=x) for x in range(0,(data_.shape[0]))]
            future_df.sort_values(by='transaction_date', ascending=True)
            future_df['quantity_sold'] = list(data_['predicted'])

            date_today = (date.today()) #YYYY-MM-DD
            filter_df = future_df[(future_df['transaction_date'].dt.year == date_today.year)]

            dates = list(filter_df['transaction_date'])
            quantity = list(filter_df['quantity_sold'])
            
            pred_dates = {}
            for i in range(len(dates)):
                pred_dates.update({str(i): dates[i]})
            
            pred_quantity = {}
            for i in range(len(dates)):
                pred_quantity.update({str(i): round(quantity[i])})

            model_dict = {}
            model_dict.update({'actual': {'dates': actual_dates_dict, 'quantity': actual_quantity_dict}})
            model_dict.update({'forecast': {'dates': pred_dates, 'quantity': pred_quantity}})
            model_dict.update({'algorithm_name': forecasting_model.algorithm_name})
            model_dict.update({'training_data': forecasting_model.training_data})
            model_dict.update({'accuracy': forecasting_model.model_accuracy})

        else:
            selected_model = models.load_model(model_path)

            if forecasting_model.algorithm_name == 'Recurrent Neural Network':
                prediction = selected_model.predict(X[-X.shape[0]:])
                prediction_copies = np.repeat(prediction, item_df.shape[1], axis=-1)
                predicted_values = scaler_df.inverse_transform(prediction_copies)[:,0]
            elif forecasting_model.algorithm_name == 'Artificial Neural Network':
                prediction = selected_model.predict(X[-X.shape[0]:])
                prediction_copies = np.repeat(prediction, item_df.shape[1], axis=-1)
                reshape_predictions = (prediction_copies.reshape(prediction_copies.shape[0], prediction_copies.shape[2]))
                predicted_values = scaler_df.inverse_transform(reshape_predictions)[:,0]

            start_forecast_date = (item_df.index.max())
            future_df = pd.DataFrame()
            future_df['transaction_date'] = [start_forecast_date + DateOffset(weeks=x) for x in range(0,len(predicted_values))]
            future_df.sort_values(by='transaction_date', ascending=True)
            future_df['quantity_sold'] = list(predicted_values)

            date_today = (date.today()) #YYYY-MM-DD
            filter_df = future_df[(future_df['transaction_date'].dt.year == date_today.year)]

            dates = list(filter_df['transaction_date'])
            quantity = list(filter_df['quantity_sold'])
            
            pred_dates = {}
            for i in range(len(dates)):
                pred_dates.update({str(i): dates[i]})
            
            pred_quantity = {}
            for i in range(len(dates)):
                pred_quantity.update({str(i): round(quantity[i])})

            model_dict = {}
            model_dict.update({'actual': {'dates': actual_dates_dict, 'quantity': actual_quantity_dict}})
            model_dict.update({'forecast': {'dates': pred_dates, 'quantity': pred_quantity}})
            model_dict.update({'algorithm_name': forecasting_model.algorithm_name})
            model_dict.update({'training_data': forecasting_model.training_data})
            model_dict.update({'accuracy': forecasting_model.model_accuracy})

        validation_data.update({forecasting_model.id: model_dict})
    return validation_data