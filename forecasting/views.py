from flask import Blueprint, render_template, request, flash, redirect, url_for, jsonify, make_response, session
from flask_login import login_required, current_user
from .classes import UploadCSV, UploadValidationCSV, UploadRetrainingCSV
from werkzeug.utils import secure_filename
import os
from .secrets import BRANCH_DATA_FOLDER, PROFILE_PICTURES, LAST_FILE_UPLOADED, VALIDATION_DATA, RETRAINING_DATA, MERGED_DATA_FOLDER, MODELS
from datetime import datetime, date
from .models import User, UserSchema, Permissions, PermissionsSchema, Uploads, UploadsSchema, Forecast_data, ForecastDataSchema, Branches, BranchesSchema, Products, ProductsSchema, SettingsSchema, Settings, Validation_dataSchema, Validation_data, Retraining_dataSchema, Retraining_data, FinalData, ForecastingModelSchema, Forecasting_model
import json
from . import db, conn
from .retraining_process import *
from .functions import *
from sqlalchemy.exc import IntegrityError
from sqlalchemy.sql import func
from sqlalchemy import delete
from werkzeug.security import generate_password_hash
import csv
import pandas as pd
from sqlalchemy import create_engine, text # for truncating table
import shutil
from tensorflow.keras import models
import pickle

# Importing Image module from PIL package
from PIL import Image
import PIL

from flask_wtf import FlaskForm


views = Blueprint('views', __name__)

@views.route('/dashboard')
@login_required
def dashboard():
    session['current_menu'] = request.path
    return render_template("modules/dashboard.html", auth_user=current_user, permissions=permissions())

# @views.route('/forecasting')
# @login_required
# def forecasting():
#     session['current_menu'] = request.path # for active menu
#     return render_template("modules/forecasting/forecasting.html", auth_user=current_user,permissions=permissions())

#######################################################
#
# USERS MODULE ROUTE
#
#######################################################

@views.route('/users', methods=['GET', 'POST'], endpoint="users")
@login_required
def users():
    session['current_menu'] = request.path # for active menu
    if request.method == 'GET':
        search = request.args.getlist('search')
        search = (','.join(search))

        fstatus = request.args.getlist('status')
        fstatus = (','.join(fstatus))

        fposition = request.args.getlist('position')
        fposition = (','.join(fposition))

        # Set the pagination configuration
        page = request.args.get('page', 1, type=int)

        if search:
            users = db.session.query(User, Permissions).join(Permissions)\
                .filter((User.user_type != -1))\
                .filter((User.first_name.like('%' + search + '%'))      |
                        (User.middle_name.like('%' + search + '%'))     |
                        (User.last_name.like('%' + search + '%'))       |
                        (User.address.like('%' + search + '%'))         |
                        (User.contact_number.like('%' + search + '%'))  |
                        (User.company_name.like('%' + search + '%'))    |
                        (User.email.like('%' + search + '%')))\
                .paginate(page=page, per_page=10)
        elif fstatus:
            users = db.session.query(User, Permissions).join(Permissions)\
                .filter((User.user_type != -1))\
                .filter(User.is_active == (True if (fstatus == '1') else False))\
                .paginate(page=page, per_page=10)
        elif fposition:
            users = db.session.query(User, Permissions).join(Permissions)\
                .filter((User.user_type != -1))\
                .filter(User.user_type == fposition)\
                .paginate(page=page, per_page=10)
        else:
            users = db.session.query(User, Permissions).join(Permissions).filter((User.user_type != -1), (User.is_active != 0)).order_by(User.first_name.asc()).paginate(page=page, per_page=10)
        return render_template("modules/users/users.html",
                           auth_user=current_user,
                           users = users,
                           prev = '',
                           new_user = 'false',
                           update_user = 'false',
                           search = search,
                           fstatus = fstatus,
                           fposition = fposition,
                           permissions=permissions(),
                           current_model='',
                           forecast_models=[])

@views.route('/create-users', methods=['POST'])
@login_required
def create_users():
    session['current_menu'] = '/users'
    if request.method == 'POST':
        try:
            first_name = request.form['first_name']
            middle_name = request.form['middle_name']
            last_name = request.form['last_name']
            email = request.form['email']
            address = request.form['address']
            contact_number = request.form['contact_number']
            user_type = request.form['user_type']
            password = generate_password_hash(request.form['password'], method="sha256")

            new_user = User(first_name, middle_name, last_name, address, contact_number, 'WeTech', email, password, None, 'default.png' if user_type == 0 else 'personnel.png', user_type, 1)
            db.session.add(new_user)
            db.session.flush()
            user_id = new_user.id
            if request.form['user_type'] == '0':
                permission = Permissions(user_id, True, True, True, True, True, True, True, True, True, True)
                db.session.add(permission)
            elif request.form['user_type'] == '1':
                permission = Permissions(user_id, True, True, None, None, None, None, None, None, None, None)
                db.session.add(permission)

            db.session.commit()
            
            flash('New User Successfully Created!', category='success')
            return redirect(url_for('.users'))
        except IntegrityError:
            db.session.rollback()
            page = request.args.get('page', 1, type=int)
            flash('Email Already Taken', category='error')
            users = db.session.query(User, Permissions).join(Permissions).filter(User.user_type != -1).paginate(page=page, per_page=10)
            return render_template("modules/users/users.html", auth_user=current_user, users = users, prev = request.form.to_dict(), new_user = 'true', update_user = 'false', search = '', fstatus = '', fposition = '', permissions=permissions(), current_model='', forecast_models=[])

@views.route('/update-users', methods=['POST'])
@login_required
def update_users():
    session['current_menu'] = '/users'
    if request.method == 'POST':
        try:
            update_user = User.query.filter_by(id=request.form['user_id']).first()
            update_user.first_name = request.form['first_name']
            update_user.middle_name = request.form['middle_name']
            update_user.last_name = request.form['last_name']
            update_user.email = request.form['email']
            update_user.address = request.form['address']
            update_user.contact_number = request.form['contact_number']
            update_user.user_type = request.form['user_type']
            update_user.is_active = True if request.form['is_active'] == '1' else False

            if request.form['password'] != None:
                password = generate_password_hash(request.form['password'], method="sha256")
                update_user.password = password

            db.session.commit()
            
            flash('User Successfully Updated!', category='success')
            return redirect(url_for('.users'))
        except IntegrityError:
            db.session.rollback()
            page = request.args.get('page', 1, type=int)
            flash('Email Already Taken', category='error')
            users = db.session.query(User, Permissions).join(Permissions).filter(User.user_type != -1).paginate(page=page, per_page=10)
            return render_template("modules/users/users.html", auth_user=current_user, users = users, prev = request.form.to_dict(), new_user = 'false', update_user = 'true', search = '', fstatus = '', fposition = '', permissions=permissions(), current_model='', forecast_models=[])

@views.route('/delete-users', methods=['POST'])
@login_required
def delete_users():
    session['current_menu'] = '/users'
    delete_user = User.query.filter_by(id=request.form['user_id']).first()
    delete_permission = Permissions.query.filter_by(user_id=request.form['user_id']).first()
    db.session.delete(delete_permission)
    db.session.delete(delete_user)
    db.session.commit()
    
    flash('User Successfully Deleted!', category='success')
    return redirect(url_for('.users'))

@views.route('/permission-user', methods=['POST'])
@login_required
def user_permission():
    session['current_menu'] = '/users'
    permission = Permissions.query.filter_by(permission_id=request.form['permission_id']).first()
    permission.forecasting = True if request.form['forecasting'] == 'true' else False
    permission.user_c = True if request.form['user_c'] == 'true' else False
    permission.user_r = True if request.form['user_r'] == 'true' else False
    permission.user_u = True if request.form['user_u'] == 'true' else False
    permission.user_d = True if request.form['user_d'] == 'true' else False
    permission.user_p = True if request.form['user_p'] == 'true' else False
    permission.setting_c = True if request.form['setting_c'] == 'true' else False
    permission.setting_u = True if request.form['setting_u'] == 'true' else False
    permission.setting_d = True if request.form['setting_d'] == 'true' else False
    permission.uploading = True if request.form['uploading'] == 'true' else False
    db.session.commit()
    
    flash('Permissions Successfully Updated!', category='success')
    return redirect(url_for('.users'))

@views.route('/update-password', methods=['POST'])
@login_required
def update_password():
    if request.method == 'POST':
        data = request.json
        update_user = User.query.filter_by(id=data['user_id']).first()
        update_user.password = generate_password_hash(data['password'], method="sha256")
        db.session.commit()
        
        return 'true'

@views.route('/update-profile', methods=['POST'])
@login_required
def update_profile():
    if request.method == 'POST':
        update_profile = User.query.filter_by(id=request.form['user_id']).first()
        update_profile.first_name = request.form['first_name']
        update_profile.middle_name = request.form['middle_name'] if request.form['middle_name'] else None
        update_profile.last_name = request.form['last_name']
        update_profile.address = request.form['address']
        update_profile.contact_number = request.form['contact_number']
        update_profile.email = request.form['email']
        file = request.files['avatar'] if request.files['avatar'] else request.files['avatar2']
        if(file):
            file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),PROFILE_PICTURES,secure_filename(file.filename))
            file.save(file_path)
            update_profile.avatar = file.filename

        flash('Profile Successfully updated!', category='p_success')
        db.session.commit()
        
        return redirect(url_for('.dashboard')) if session['current_menu'] == '/dashboard'  else (redirect(url_for('.users')) if session['current_menu'] == '/users' else (redirect(url_for('.upload')) if session['current_menu'] == '/upload' else (redirect(url_for('.settings')) if session['current_menu'] == '/settings' else (redirect(url_for('forecast.forecasting'))))))

#######################################################
#
# UPLOAD MODULE ROUTE
#
#######################################################

@views.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    session['current_menu'] = request.path
    form = UploadCSV()
    form_val = UploadValidationCSV()
    form_ret = UploadRetrainingCSV()
    # file = []

    # check for existing uploaded file
    last_filename = Uploads.query.all()
    extrated_data = {}
    branches = {}
    
    # Set the pagination configuration
    page = request.args.get('page', 1, type=int)

    if request.method == 'GET':
        # Branches
        existing_branch_forecast = db.session.query(Forecast_data.branch_id).group_by(Forecast_data.branch_id).all()
        branch_list = []
        for branch in existing_branch_forecast:
            branch_list.append(branch.branch_id)
        # branches = db.session.query(Branches).filter(Branches.branch_id.notin_(branch_list)).all()
        branches = db.session.query(Branches).all()
        branches_count = db.session.query(Branches).count()

        # Active tab
        active_tab = db.session.query(Settings).filter(Settings.id == 1).order_by(Settings.id.asc()).first()
        # Active tab

        if last_filename:
            extrated_data = Forecast_data.query.paginate(page=page, per_page=100)
            
        return render_template("modules/upload/upload.html", auth_user=current_user, form=form, form_val=form_val, form_ret=form_ret, csv_data = ((extrated_data)), branches=branches, branches_count=branches_count, permissions=permissions(), current_model='', forecast_models=[], active_tab = active_tab.active_upload_tab)
    elif request.method == 'POST':
        # checking if data already exists
        
        if "is_used" in request.form:
            # csv_file_data = []
            file = form_val.file.data

            file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),VALIDATION_DATA,secure_filename(datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p") + '_' + file.filename))
            file_name = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p") + '_validation_data_branch_' + request.form['branch_id'] + '.csv'
            file_path_cleaned = os.path.join(os.path.abspath(os.path.dirname(__file__)),VALIDATION_DATA,secure_filename(file_name))
            file.save(file_path) # Then save the raw file

            val_data = validation_data_preprocessing(file_path)
            os.remove(file_path)
            val_data.to_csv(file_path_cleaned, index=False)

            exsisting_upload = Validation_data.query.all()
            if exsisting_upload:
                update_validation_data = Validation_data.query.filter_by(branch_id=int(request.form['branch_id'])).first()
                update_validation_data.file_name = file_name
                db.session.commit()
            else:
                new_validation_data = Validation_data(int(request.form['branch_id']), file_name, True)
                db.session.add(new_validation_data)
                db.session.commit()
            

            return 'success'
        elif "retrain_model" in request.form:            
            file = form_ret.file.data

            # GET FILE NAME AND EXTENSION
            uploaded_file_name, uploaded_file_extension = (os.path.splitext(file.filename))
            branch_details = db.session.query(Branches).filter(Branches.branch_description == uploaded_file_name).first()

            # If naming convention is not followed
            if not (branch_details):
                return_data = {}
                return_data.update({'err_msg': 'Please follow the provied naming convention'})
                return return_data


            file_name = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p") + '_retraining_data_' + str(branch_details.branch_description).lower() + '.csv'
            file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),RETRAINING_DATA,secure_filename(file_name))
            file.save(file_path) # Then save the raw file
            
            update_retraining_data = Retraining_data.query.filter_by(branch_id = (branch_details.branch_id)).first()
            if update_retraining_data:
                update_retraining_data = Retraining_data.query.filter_by(branch_id = (branch_details.branch_id)).first()
                update_retraining_data.file_name = file_name
                db.session.commit()
                
            else:
                new_training_data = Retraining_data((branch_details.branch_id), file_name, 1)
                db.session.add(new_training_data)
                db.session.commit()
                

            # Retraining_data
            # update_validation_data = Retraining_data.query.filter_by(branch_id= (int(request.form['index']) + 1 ) ).first()
            # update_validation_data.file_name = file_name
            # db.session.commit()


            return 'Upload Successfull! File Name: (' + str(file_name) + ')'
        else:
            # csv_file_data = []
            file = form.file.data

            file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),BRANCH_DATA_FOLDER,secure_filename((datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p") + '_' + file.filename)))
            file.save(file_path) # Then save the raw file

            flag = check_duplicate_data(file_path, int(request.form['branch_id']))
            if flag == True:
                return 'duplicate_data'
                
            #
            # 
            # IF 52 WEEKS ?
            # 
            #
            
            new_data, products = data_preprocessing(file_path)
            os.remove(file_path)
            LAST_FILE_UPLOADED = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p") + '_cleaned_' + file.filename
            clean_file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),BRANCH_DATA_FOLDER,secure_filename(datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p") + '_cleaned_' + file.filename))
            new_data.to_csv(clean_file_path, index=False)
            new_file = Uploads(LAST_FILE_UPLOADED, request.form['branch_id'])
            db.session.add(new_file)

            for i in range(len(products)):
                insert_products = Products(products[i][0], products[i][1], request.form['branch_id'])
                db.session.add(insert_products)
            # db.session.commit()

            with open(clean_file_path) as file:
                csv_file = csv.reader(file)
                header = next(csv_file)
                for row in csv_file:
                    sales = Forecast_data(row[2], row[0], row[1], row[3], row[4], row[5], row[6], row[7], row[8], request.form['branch_id'])
                    db.session.add(sales)

            #commit all data and changes
            db.session.commit()
            
            return 'false'

def permissions():
    perm = Permissions.query.filter_by(user_id=current_user.id).first()
    permission_schema = PermissionsSchema()
    return permission_schema.dump(perm)

@views.route('/settings', methods=['GET', 'POST'])
@login_required
def settings():
    session['current_menu'] = request.path

    if request.method == 'GET':
        #Branches
        branches = Branches.query.all()
        last_branch = db.session.query(Branches).order_by(Branches.branch_id.desc()).first()
        last_branch_id = (last_branch.branch_id + 1) if last_branch else 1

        # Branch Data
        branch_data = {}
        c = 0
        for branch in branches:
            # Branch Description
            br = Branches.query.filter_by(branch_id=branch.branch_id).first()
            # Branch data count
            data_count  = (db.session.query(Forecast_data).filter(Forecast_data.branch_id == branch.branch_id).count())

            # Branch data count
            product_count = (db.session.query(Products).filter(Products.branch_id == branch.branch_id).count())
            if product_count != 0:
                branch_data[c] = {'branch_id': br.branch_id, 'branch_description': br.branch_description, 'data_count': data_count, 'product_count': product_count}
                c = c + 1

        forecasting_model = Forecasting_model.query.filter_by(is_active = True).all()
        forecasting_schema = ForecastingModelSchema(many=True)

    return render_template("modules/settings/settings.html", auth_user=current_user, branches=branches, branch_data=branch_data, prev = '', last_branch_id=last_branch_id, new_branch='false', update_branch='false', permissions=permissions(), current_model='', forecast_models=forecasting_schema.dump(forecasting_model))

@views.route('/create-branch', methods=['POST'])
@login_required
def create_branch():
    session['current_menu'] = '/settings'
    if request.method == 'POST':
        try:
            branch_id = request.form['branch_id']
            branch_description = request.form['branch_description']
            branch_address = request.form['branch_address']

            new_branch = Branches(branch_id, branch_description, branch_address)
            db.session.add(new_branch)
            db.session.flush()
            db.session.commit()
            
            flash('New Branch Successfully Created!', category='success')
            return redirect(url_for('.settings'))
        except IntegrityError:
            db.session.rollback()
            branches = Branches.query.all()
            flash('Branch ID Already Exist', category='error')
            # Branch Data
            branch_data = {}
            c = 0
            for branch in branches:
                # Branch Description
                br = Branches.query.filter_by(branch_id=branch.branch_id).first()
                # Branch data count
                data_count  = (db.session.query(Forecast_data).filter(Forecast_data.branch_id == branch.branch_id).count())
                # Branch data count
                product_count = (db.session.query(Products).filter(Products.branch_id == branch.branch_id).count())
                if product_count != 0:
                    branch_data[c] = {'branch_description': br.branch_description, 'data_count': data_count, 'product_count': product_count}
                    c = c + 1
            return render_template("modules/settings/settings.html", auth_user=current_user, branches=branches, branch_data=branch_data, prev = request.form.to_dict(), last_branch_id='', new_branch='true', update_branch='false', permissions=permissions(), current_model='', forecast_models=[])

@views.route('/delete-branch', methods=['POST'])
@login_required
def delete_branch():
    session['current_menu'] = '/settings'
    delete_branch = Branches.query.filter_by(id=request.form['branch_id']).first()
    db.session.delete(delete_branch)
    db.session.commit()
    
    flash('Branch Successfully Deleted!', category='success')
    return redirect(url_for('.settings'))

@views.route('/delete-branch-data', methods=['POST'])
@login_required
def delete_branch_data():
    session['current_menu'] = '/settings'

    branches = db.session.query(Uploads).filter(Uploads.branch_id == int(request.form['data_branch_id'])).first()
    
    file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),BRANCH_DATA_FOLDER,secure_filename((branches.csv_filename)))
    os.remove(file_path)

    delete_data = delete(Forecast_data).where(Forecast_data.branch_id == request.form['data_branch_id'])
    delete_product = delete(Products).where(Products.branch_id == request.form['data_branch_id'])
    delete_upload = delete(Uploads).where(Uploads.branch_id == request.form['data_branch_id'])
    
    db.session.execute(delete_data)
    db.session.execute(delete_product)
    db.session.execute(delete_upload)
    db.session.commit()
    flash('Branch Data Successfully Deleted!', category='success')
    return redirect(url_for('.settings'))


@views.route('/update-branch', methods=['POST'])
@login_required
def update_branch():
    session['current_menu'] = '/settings'
    if request.method == 'POST':
        try:
            update_branch = Branches.query.filter_by(id=request.form['u_id']).first()
            update_branch.branch_id = request.form['branch_id']
            update_branch.branch_description = request.form['branch_description']
            update_branch.branch_address = request.form['branch_address']

            db.session.commit()
            flash('Branch Successfully Updated!', category='success')
            return redirect(url_for('.settings'))
        except IntegrityError:
            db.session.rollback()
            branches = Branches.query.all()
            flash('Branch ID Already Exist', category='error')
            # Branch Data
            branch_data = {}
            c = 0
            for branch in branches:
                # Branch Description
                br = Branches.query.filter_by(branch_id=branch.branch_id).first()
                # Branch data count
                data_count  = (db.session.query(Forecast_data).filter(Forecast_data.branch_id == branch.branch_id).count())
                # Branch data count
                product_count = (db.session.query(Products).filter(Products.branch_id == branch.branch_id).count())
                if product_count != 0:
                    branch_data[c] = {'branch_description': br.branch_description, 'data_count': data_count, 'product_count': product_count}
                    c = c + 1
            return render_template("modules/settings/settings.html", auth_user=current_user, branches=branches, branch_data=branch_data, prev = request.form.to_dict(), last_branch_id='', new_branch='false', update_branch='true', permissions=permissions(), current_model='', forecast_models=[])

@views.route('/delete-model', methods=['POST'])
@login_required
def delete_model():
    if request.method == 'POST':
        delete_model = Forecasting_model.query.filter_by(id = int(request.form['model_id'])).first()
        delete_model.is_active = False
        db.session.commit()
        

        forecasting_models = Forecasting_model.query.filter_by(is_active = True).all()
        return_data = {}
        c = 0
        for forecasting_model in forecasting_models:
            forecast_data_ = {}
            forecast_data_.update({
                'id': forecasting_model.id,
                'algorithm_name': forecasting_model.algorithm_name,
                'model_name': forecasting_model.model_name,
                'is_retrained': forecasting_model.is_retrained,
                'training_data': forecasting_model.training_data,
                'start_testing_data': forecasting_model.start_testing_data,
                'model_accuracy': forecasting_model.model_accuracy,
                'is_used': forecasting_model.is_used,
                'is_active': forecasting_model.is_active,
                'date_created': forecasting_model.date_created
            })
            return_data.update({str(c):forecast_data_ })
            c = c + 1

        return return_data

@views.route('use-model', methods=['POST'])
@login_required
def use_model():
    if request.method == 'POST':
        update_model = Forecasting_model.query.filter_by(is_used = 1).first()
        update_model.is_used = False

        # change used model
        delete_model = Forecasting_model.query.filter_by(id = int(request.form['model_id'])).first()
        delete_model.is_used = True
        db.session.commit()
        

        forecasting_models = Forecasting_model.query.filter_by(is_active = True).all()
        return_data = {}
        c = 0
        for forecasting_model in forecasting_models:
            forecast_data_ = {}
            forecast_data_.update({
                'id': forecasting_model.id,
                'algorithm_name': forecasting_model.algorithm_name,
                'model_name': forecasting_model.model_name,
                'is_retrained': forecasting_model.is_retrained,
                'training_data': forecasting_model.training_data,
                'start_testing_data': forecasting_model.start_testing_data,
                'model_accuracy': forecasting_model.model_accuracy,
                'is_used': forecasting_model.is_used,
                'is_active': forecasting_model.is_active,
                'date_created': forecasting_model.date_created
            })
            return_data.update({str(c):forecast_data_ })
            c = c + 1

        return return_data

@views.route('retrain-selected-model', methods=['POST'])
@login_required
def retrain_selected_model():
    if request.method == 'POST':
        new_model_id = retraining_process(int(request.form['model_id']))
        
        # # Create all products forecast
        algorithm_ = db.session.query(Forecasting_model).filter(Forecasting_model.id == new_model_id).first()
        file_path = MODELS + "\\" + algorithm_.algorithm_name + "\\" + algorithm_.training_data
        model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),file_path,secure_filename(algorithm_.model_name))
        
        if algorithm_.algorithm_name == 'Support Vector Regression':
            selected_model = pickle.load(open(model_path, 'rb'))
        else:
            selected_model = models.load_model(model_path)
        all_models_forecast(new_model_id, selected_model, algorithm_.algorithm_name)
        db.session.commit()
        

        forecasting_models = Forecasting_model.query.filter_by(is_active = True).all()
        return_data = {}
        c = 0
        for forecasting_model in forecasting_models:
            forecast_data_ = {}
            forecast_data_.update({
                'id': forecasting_model.id,
                'algorithm_name': forecasting_model.algorithm_name,
                'model_name': forecasting_model.model_name,
                'is_retrained': forecasting_model.is_retrained,
                'training_data': forecasting_model.training_data,
                'start_testing_data': forecasting_model.start_testing_data,
                'model_accuracy': forecasting_model.model_accuracy,
                'is_used': forecasting_model.is_used,
                'is_active': forecasting_model.is_active,
                'date_created': forecasting_model.date_created
            })
            return_data.update({str(c):forecast_data_ })
            c = c + 1


        return return_data


@views.route('/update-upload-tab', methods=['POST'])
@login_required
def update_upload_tab():
    if request.method == 'POST':
        try:
            update_active_tab = Settings.query.filter_by(id=1).first()
            update_active_tab.active_upload_tab = int(request.form['tab_index'])
            db.session.commit()
            
        except IntegrityError:
            return str(request.form['tab_index'])
        
        active_tab = db.session.query(Settings).filter(Settings.id == 1).order_by(Settings.id.asc()).first()
        return str(active_tab.active_upload_tab)

@views.route('/retraining-preprocessing', methods=['POST'])
@login_required
def retraining_preprocessing():
    if request.method == 'POST':
        if "read_dataset" in request.form:
            return 'Preprocessing Datasets, Please wait...'
        elif "start_preprocessing" in request.form:
            branches = db.session.query(Branches).all()
            rows = int(request.form['no_branches'])
            cols = 2
            
            dataset_list = [[0 for x in range(cols)] for x in range(rows)]
            flag_list = [[0 for x in range(cols)] for x in range(rows)]

            c = 0
            for branch in branches:
                ret_record = Retraining_data.query.filter_by(branch_id = branch.branch_id ).first()
                file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),RETRAINING_DATA,secure_filename(ret_record.file_name))
                branch_data, list_product =  data_preprocessing(file_path, is_retraining_data=True)
                branch_data.drop(columns=['quantity_return', 'item_price', 'item_cost', 'total_item_cost'], inplace=True)
                os.remove(file_path)
                
                if (branch_data.empty):
                    flag_list[c][0] = True
                    flag_list[c][1] = branch.branch_id


                dataset_list[c][0] = branch_data
                dataset_list[c][1] = branch.branch_id
                c = c + 1

            # 
            # UNQUALIFIED DATASET
            #
            
            for f in range(len(flag_list)):
                branch_details = Branches.query.filter_by(branch_id = int(flag_list[f][1])).first()
                if flag_list[f][0] == True:
                    return_data = {}
                    return_data.update({'branch_name': str(branch_details.branch_description)})
                    return return_data
            
            for i in range(len(dataset_list)):
                branch_details = Branches.query.filter_by(branch_id = int(dataset_list[i][1])).first()

                file_name = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p") + '_cleaned_retraining_data_' + str(branch_details.branch_description).lower() + '.csv'
                file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),RETRAINING_DATA,secure_filename(file_name))

                dataset_list[i][0].to_csv(file_path, index=False)

                ret_record = Retraining_data.query.filter_by(branch_id = int(dataset_list[i][1])).first()
                ret_record.file_name = file_name
                db.session.commit()
                

            return 'Analyzing Data, Please wait...'
        elif "start_feature_engineering" in request.form:
            retraining_record = Retraining_data.query.all()
            for record in retraining_record:
                file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),RETRAINING_DATA,secure_filename(record.file_name))
                fe_data = feature_engineering(file_path, record.branch_id)
                
                file_name_ = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p") + '_FE_retraining_data_' + str(record.branch_id).lower() + '.csv'
                fe_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),RETRAINING_DATA,secure_filename(file_name_))
                fe_data.to_csv(fe_path, index=False)

                ret_record = Retraining_data.query.filter_by(branch_id = record.branch_id).first()
                ret_record.file_name = file_name_
                db.session.commit()
                
            
            retraining_record = Retraining_data.query.all()

            filename_list = []
            for record in retraining_record:
                file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)),RETRAINING_DATA,secure_filename(record.file_name))
                filename_list.append(pd.read_csv(file_path, index_col=False))

            final_data = pd.DataFrame()
            for i in range(len(filename_list)):
                final_data = final_data.append(filename_list[i], ignore_index=True)

            # APPEND CURRENT FINAL_DATA
            get_current_final_data = FinalData.query.filter_by(id=1).first()
            current_final_data = os.path.join(os.path.abspath(os.path.dirname(__file__)),MERGED_DATA_FOLDER,secure_filename(get_current_final_data.csv_filename))
            current_df = pd.read_csv(current_final_data, index_col=False)
            final_data = final_data.append(current_df, ignore_index=True)
            # APPEND CURRENT FINAL_DATA

            merge_data_filename = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p") + '_final_data' + '.csv'
            merge_data = os.path.join(os.path.abspath(os.path.dirname(__file__)),MERGED_DATA_FOLDER,secure_filename(merge_data_filename))
            
            final_data.to_csv(merge_data, index=False)

            setting_temp = Settings.query.filter_by(id = 1).first()
            setting_temp.temp_final_data = merge_data_filename
            db.session.commit()
            
            
            return 'Datasets successfully proccessed!'

@views.route('/custom-connection-cycle', methods=['POST'])
@login_required
def custom_connection_cycle():
    setting_custom = Settings.query.filter_by(id=1).first()
    print(setting_custom.active_upload_tab)
    return str(setting_custom.active_upload_tab)