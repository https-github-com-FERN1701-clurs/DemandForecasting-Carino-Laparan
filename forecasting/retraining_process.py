from .models import *
from .secrets import *
from . import db

from werkzeug.utils import secure_filename
import os
import shutil
from tensorflow.keras import models
import pickle
from sqlalchemy.sql import func
# import pytz


'''
    LIBRARY FOR ANN
'''
import pandas as pd
import numpy as np
from datetime import date, datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from tensorflow.keras import metrics, callbacks, initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import models

from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.optimizers import Adam
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from pandas.tseries.offsets import DateOffset

def retraining_process(model_id):
    current_algorithm = Forecasting_model.query.filter_by(id = model_id).first()
    model_path = MODELS + "\\" + current_algorithm.algorithm_name + "\\" + current_algorithm.training_data
    model_filename = os.path.join(os.path.abspath(os.path.dirname(__file__)),model_path,secure_filename(current_algorithm.model_name))

    generate_filename = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p") +'_' + current_algorithm.model_name
    target_model_filename = os.path.join(os.path.abspath(os.path.dirname(__file__)),model_path,secure_filename(generate_filename))

    shutil.copyfile(model_filename, target_model_filename)

    new_final_data = Settings.query.filter_by(id=1).first()
    new_final_data_filename = new_final_data.temp_final_data
    new_final_data_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),MERGED_DATA_FOLDER,secure_filename(new_final_data_filename))
    dataset = pd.read_csv(new_final_data_path, parse_dates=['transaction_date'])
    
    # 
    # 
    # PREPARE DATASET
    # 
    #
    # Set transaction date as Index
    dataset.set_index('transaction_date', inplace=True)

    ### Pop quantity_sold column and insert to first column
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

    dataset.drop(columns=['item_description', 'year', 'quarter', 'month', 'day', 'dayofweek', 'week', 'is_pandemic', 'ber_month', 'rainy_season', 'march_may_period', 'summer_season'], inplace=True)
    
    features = list(dataset.columns)
    df_training = dataset[features]
    scaler_df_training = StandardScaler()
    scaled_df_training = scaler_df_training.fit_transform(df_training)
    
    train_features = scaled_df_training[:, 1:]
    train_target   = scaled_df_training[:, 0:1]

    # train_features, test_features, train_target, test_target = train_test_split(main_train_features, main_train_target, test_size=0.20, shuffle=False)
    # test_features = test_features.reshape(test_features.shape[0], 1, test_features.shape[1])
    
    # 
    #
    # LOAD MODEL
    # 
    #

    if current_algorithm.algorithm_name == 'Support Vector Regression':
        selected_model = pickle.load(open(target_model_filename, 'rb'))
        selected_model.fit(train_features, np.ravel(train_target, order='C'))

        svr_predictions_train = regressor.predict(train_features)
        svr_predictions_features_train = pd.concat([pd.DataFrame(svr_predictions_train),pd.DataFrame(scaled_df_training[:, 1:][:])], axis=1)
        reverse_trans_pred_svr_train = scaler_df_test.inverse_transform(svr_predictions_features_train)

        df_train = df_training[reverse_trans_pred_svr_train[:, 0].shape[0]*-1:]
        df_train = df_train.copy()
        df_train['predicted']=reverse_trans_pred_svr_train[:, 0]

        qty_sold_train = list(df_train.quantity_sold)
        pred_train = list(df_train.predicted)
        print(f'MAE: {mean_absolute_error(qty_sold_train, pred_train)}')
        print(f'MSE: {mean_squared_error(qty_sold_train, pred_train)}')
        print(f'RMSE: {mean_squared_error(qty_sold_train, pred_train, squared=False)}')
        print(f'MAPE: {mean_absolute_percentage_error(qty_sold_train, pred_train)}')
        print(f'R2 SCORE: {r2_score(qty_sold_train, pred_train)}')
        model_accu = "{:.0%}".format(r2_score(qty_sold_train, pred_train))
    else:
        train_features = train_features.reshape(train_features.shape[0], 1, train_features.shape[1])
        selected_model = models.load_model(target_model_filename)

        early_stopping = EarlyStopping(monitor='loss', patience=3, mode='min')
        selected_model.compile(loss=tf.losses.MeanSquaredError(), optimizer='adam', metrics=['mse', 'mae', 'mape'])
        history = selected_model.fit(train_features, train_target, batch_size=32, epochs=30, shuffle=False, callbacks=[early_stopping])
        
        predictions=selected_model.predict(train_features)
        predictions = predictions.reshape(-1, 1)

        train_predictions_features = pd.concat([pd.DataFrame(predictions),pd.DataFrame(scaled_df_training[:, 1:][-predictions.shape[0]:])], axis=1)
        reverse_train_pred = scaler_df_training.inverse_transform(train_predictions_features)

        df_train = df_training[reverse_train_pred[:, 0].shape[0]*-1:]
        df_train = df_train.copy()
        df_train['predicted']=reverse_train_pred[:, 0]
        qty_sold_train = list(df_train.quantity_sold)
        pred_train = list(df_train.predicted)
        print(f'MAE: {mean_absolute_error(qty_sold_train, pred_train)}')
        print(f'MSE: {mean_squared_error(qty_sold_train, pred_train)}')
        print(f'RMSE: {mean_squared_error(qty_sold_train, pred_train, squared=False)}')
        print(f'MAPE: {mean_absolute_percentage_error(qty_sold_train, pred_train)}')
        print(f'R2 SCORE: {r2_score(qty_sold_train, pred_train)}')
        model_accu = "{:.0%}".format(r2_score(qty_sold_train, pred_train))
       
    os.remove(target_model_filename)
    generate_filename = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p") +'_' + current_algorithm.model_name
    new_target_model_filename = os.path.join(os.path.abspath(os.path.dirname(__file__)),model_path,secure_filename(generate_filename))
    selected_model.save(new_target_model_filename)

    # Set to inactive the selected model
    retrain_model = Forecasting_model.query.filter_by(id=model_id).first()
    retrain_model.is_active = False

    # Set is_used to false if selected model is is used
    use_this = False
    if current_algorithm.is_used == True:
        current_algorithm.is_used = False
        use_this = True

    # TImezone
    # manila = pytz.timezone("Asia/Manila") 
    
    # Add new retrain model
    today = date.today()
    new_model = Forecasting_model(current_algorithm.algorithm_name, generate_filename, True, today.year,current_algorithm.training_data, current_algorithm.start_testing_data, model_accu, use_this, True, new_final_data_filename, func.now())
    db.session.add(new_model)
    db.session.flush()

    # Commit changes
    db.session.commit()

    return int(new_model.id)
        
def rolling_cv(dataset, model, features, early_stopping):
    start_date = dataset.index.min()
    end_date = dataset.index.max()
    _training_date = (start_date + DateOffset(weeks=10))
    s_validation_date = _training_date + DateOffset(weeks=4)
    _e_validation_date = s_validation_date + DateOffset(weeks=5)
    week_incrementor = 10
    
    mae = 0
    mse = 0
    rmse = 0
    mape = 0
    r2score = 0
    counter = 1
    while _e_validation_date < end_date:
        df = dataset.copy()
        val_df_training = df[(df.index >= start_date) & (df.index <= _training_date)]
        val_df_test = df[(df.index >= s_validation_date) & (df.index <= _e_validation_date)]

        val_df_training = val_df_training[features]
        val_df_test = val_df_test[features]

        scaler_val_df_training = StandardScaler()
        scaler_val_df_test = StandardScaler()

        scaled_val_df_training = scaler_val_df_training.fit_transform(val_df_training)
        scaled_val_df_test = scaler_val_df_test.fit_transform(val_df_test)

        val_train_features = scaled_val_df_training[:, 1:]
        val_train_target   = scaled_val_df_training[:, 0:1]
        val_train_features = val_train_features.reshape(val_train_features.shape[0], 1, val_train_features.shape[1])

        val_test_features = scaled_val_df_test[:, 1:]
        val_test_target   = scaled_val_df_test[:, 0:1]
        val_test_features = val_test_features.reshape(val_test_features.shape[0], 1, val_test_features.shape[1])
        
        history = model.fit(val_train_features, val_train_target, batch_size=32, epochs=30, shuffle=False, callbacks=[early_stopping])
        model.evaluate(val_train_features, val_train_target, verbose=0)
        predictions = model.predict(val_test_features)
        predictions = predictions.reshape(-1, 1)

        reserse_transformation = pd.concat([pd.DataFrame(predictions),pd.DataFrame(scaled_val_df_test[:, 1:][-predictions.shape[0]:])], axis=1)
        reserse_transformation_pred = scaler_val_df_test.inverse_transform(reserse_transformation)

        val_df_test = val_df_test[reserse_transformation_pred[:, 0].shape[0]*-1:]
        val_df_test = val_df_test.copy()
        val_df_test['predicted']=reserse_transformation_pred[:, 0]

        qty_sold = list(val_df_test.quantity_sold)
        pred = list(val_df_test.predicted)
        
        mae =  mae + mean_absolute_error(qty_sold, pred)
        mse = mse + mean_squared_error(qty_sold, pred)
        rmse = rmse + mean_squared_error(qty_sold, pred, squared=False)
        mape = mape + mean_absolute_percentage_error(qty_sold, pred)
        r2score = r2score + r2_score(qty_sold, pred)
        counter = counter + 1
        print(f'MAE: {mean_absolute_error(qty_sold, pred)}')
        print(f'MSE: {mean_squared_error(qty_sold, pred)}')
        print(f'RMSE: {mean_squared_error(qty_sold, pred, squared=False)}')
        print(f'MAPE: {mean_absolute_percentage_error(qty_sold, pred)}')
        print(f'R2 SCORE: {r2_score(qty_sold, pred)}')
        print('\n')
        
        week_incrementor = week_incrementor + 4
        _training_date = (start_date + DateOffset(weeks=week_incrementor))
        s_validation_date = _training_date + DateOffset(weeks=1)
        _e_validation_date = s_validation_date + DateOffset(weeks=5)
        break

    mae = mae / counter
    mse = mse / counter
    rmse = rmse / counter
    mape = mape / counter
    r2score = r2score / counter
    
    return mae, mse, rmse, mape, r2score, counter
    