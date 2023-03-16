from ast import Continue
from collections import UserList
from dataclasses import field
from email.policy import default
from time import timezone
from sqlalchemy.sql import func
from sqlalchemy import column, func
from . import db, marsh, app
from .secrets import SECRET_KEY
from flask_login import UserMixin

from marshmallow import Schema, fields
import json
from itsdangerous.serializer import Serializer
from itsdangerous.url_safe import URLSafeSerializer
from itsdangerous import Signer
from itsdangerous import TimestampSigner
from itsdangerous import BadSignature

import random
import string
from werkzeug.security import generate_password_hash, check_password_hash


class UserSchema(marsh.Schema):
    class Meta:
        fields = ('id', 'first_name', 'middle_name', 'last_name', 'address', 'contact_number', 'company_name', 'email', 'avatar', 'user_type', 'is_active')

class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    first_name = db.Column(db.String(255), nullable=False)
    middle_name = db.Column(db.String(255))
    last_name = db.Column(db.String(255), nullable=False)
    address = db.Column(db.String(255), nullable=False)
    contact_number = db.Column(db.String(15), nullable=False)
    company_name = db.Column(db.String(255), nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(255), nullable=False)
    fp_token = db.Column(db.String(255))
    avatar = db.Column(db.String(255), nullable=False, default='default.png')
    user_type = db.Column(db.SmallInteger, nullable=False, default=1) # -1 Superadmin(Built-in), 0 - Admin, 1 - Personnel
    is_active = db.Column(db.Boolean, nullable=False, default=1)
    date_created = db.Column(db.DateTime(timezone=True), default=func.now())
    permission = db.relationship('Permissions', backref='user', uselist=False)

    def __init__(self, first_name, middle_name, last_name, address, contact_number, company_name, email, password, fp_token, avatar, user_type, is_active):
        self.first_name = first_name
        self.middle_name = middle_name
        self.last_name = last_name
        self.address = address
        self.contact_number = contact_number
        self.company_name = company_name
        self.email = email
        self.password = password
        self.fp_token = fp_token
        self.avatar = avatar
        self.user_type = user_type
        self.is_active = is_active

    def get_reset_token(self):
        # Generate random string 11 characters
        # then hash the random string
        random_string = ''.join(random.choices(string.ascii_letters, k=11))
        generated_hash = generate_password_hash(random_string, method="sha256")

        user = User.query.get(self.id)
        user.fp_token = str({"key":generated_hash, "email":self.email})
        db.session.commit()
        url_serializer = URLSafeSerializer(generated_hash)
        return url_serializer.dumps({'user_id':self.id}, salt=generated_hash)

    # @staticmethod
    def verify_reset_token(token):
        fp_tokens = User.query.filter((User.fp_token != None)).all()
        user_id = ''
        matched = False
        for fp_token in fp_tokens:
            try:
                user_token = fp_token.fp_token
                # change single quote to double quote so that the json loads works
                user_token = user_token.replace("\'", "\"")
                user_token = json.loads(user_token)

                # Get user based on the email
                user_request = User.query.filter(User.email == user_token['email']).first()

                # Deserialize the token
                url_serializer = URLSafeSerializer(user_token['key'])
                user_id = url_serializer.loads(token, salt=user_token['key'])

                user_request = User.query.filter(User.email == user_token['email']).first()
                if user_id['user_id'] == user_request.id:
                    matched = True
                    break
            except(BadSignature):
                Continue
        return User.query.get(user_id['user_id']) if matched else None

class PermissionsSchema(marsh.Schema):
    class Meta:
        fields = ('permission_id', 'user_id', 'forecasting', 'user_c', 'user_r', 'user_u', 'user_d', 'user_p', 'uploading', 'setting_c', 'setting_u', 'setting_d')

class Permissions(db.Model):
    permission_id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    forecasting = db.Column(db.Boolean, default=0)
    user_c = db.Column(db.Boolean, default=0)
    user_r = db.Column(db.Boolean, default=0)
    user_u = db.Column(db.Boolean, default=0)
    user_d = db.Column(db.Boolean, default=0)
    user_p = db.Column(db.Boolean, default=0)
    setting_c = db.Column(db.Boolean, default=0)
    setting_u = db.Column(db.Boolean, default=0)
    setting_d = db.Column(db.Boolean, default=0)
    uploading = db.Column(db.Boolean, default=0)
    date_created = db.Column(db.DateTime(timezone=True), default=func.now())

    def __init__(self, user_id, forecasting, user_c, user_r, user_u, user_d, user_p, uploading, setting_c, setting_u, setting_d):
        self.user_id = user_id
        self.forecasting = forecasting
        self.user_c = user_c
        self.user_r = user_r
        self.user_u = user_u
        self.user_d = user_d
        self.user_p = user_p
        self.setting_c = setting_c
        self.setting_u = setting_u
        self.setting_d = setting_d
        self.uploading = uploading

class UploadsSchema(marsh.Schema):
    class Meta:
        fields = ('id', 'csv_filename', 'branch_id')

class Uploads(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    csv_filename = db.Column(db.String(255))
    branch_id = db.Column(db.Integer)

    def __init__(self, csv_filename, branch_id):
        self.csv_filename = csv_filename
        self.branch_id = branch_id

class Forecast_data(db.Model):
    transaction_id = db.Column(db.Integer, primary_key=True)
    transaction_date = db.Column(db.Date)
    item_code = db.Column(db.Integer)
    item_description = db.Column(db.String(255))
    quantity_sold = db.Column(db.Integer)
    quantity_return = db.Column(db.Integer)
    item_price = db.Column(db.Integer)
    total_item_price = db.Column(db.Integer)
    item_cost = db.Column(db.Integer)
    total_item_cost = db.Column(db.Integer)
    branch_id = db.Column(db.Integer)

    def __init__(self, transaction_date, item_code, item_description, quantity_sold, quantity_return, item_price, total_item_price, item_cost, total_item_cost, branch_id):
        self.transaction_date = transaction_date
        self.item_code = item_code
        self.item_description = item_description
        self.quantity_sold = quantity_sold
        self.quantity_return = quantity_return
        self.item_price = item_price
        self.total_item_price = total_item_price
        self.item_cost = item_cost
        self.total_item_cost = total_item_cost
        self.branch_id = branch_id

class ForecastDataSchema(marsh.Schema):
    class Meta:
        fields = ('transaction_id', 'transaction_date', 'item_code', 'item_description', 'quantity_sold', 'quantity_return', 'item_price', 'total_item_price', 'item_cost', 'total_item_cost', 'branch_id')

class Branches(db.Model):
    id = db.Column(db.Integer,primary_key=True)
    branch_id = db.Column(db.Integer, unique=True)
    branch_description = db.Column(db.String(255))
    branch_address = db.Column(db.String(255))
    date_created = db.Column(db.DateTime(timezone=True), default=func.now())

    def __init__(self, branch_id, branch_description, branch_address):
        self.branch_id = branch_id
        self.branch_description = branch_description
        self.branch_address = branch_address

class BranchesSchema(marsh.Schema):
    class Meta:
        fields = ('id', 'branch_id', 'branch_description', 'branch_address')

class Products(db.Model):
    product_id = db.Column(db.Integer, primary_key=True)
    product_code = db.Column(db.String(255))
    product_description = db.Column(db.String(255))
    # product_selling_price = db.Column(db.Integer)
    # product_cost = db.Column(db.Integer)
    branch_id = db.Column(db.Integer)

    def __init__(self, product_code, product_description, branch_id):
        self.product_code = product_code
        self.product_description = product_description
        # self.product_selling_price = product_selling_price
        # self.product_cost = product_cost
        self.branch_id = branch_id

class ProductsSchema(marsh.Schema):
    class Meta:
        fields = ('product_id', 'product_code', 'product_description', 'branch_id')

class FinalDataSchema(marsh.Schema):
    class Meta:
        fields = ('id', 'csv_filename', 'is_active')

class FinalData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    csv_filename = db.Column(db.String(255))
    is_active = db.Column(db.Boolean, nullable=False, default=1)
    date_created = db.Column(db.DateTime(timezone=True), default=func.now())

    def __init__(self, csv_filename, is_active):
        self.csv_filename = csv_filename
        self.is_active = is_active

class Forecasted_productsSchema(marsh.Schema):
    class Meta:
        fields = ('id', 'product_code', 'branch_id', 'forecast_data', 'model_id')

class Forecasted_products(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    product_code = db.Column(db.Integer)
    branch_id = db.Column(db.Integer)
    forecast_data = db.Column(db.JSON)
    model_id = db.Column(db.Integer, nullable=False)
    date_created = db.Column(db.DateTime(timezone=True), default=func.now())


    def __init__(self, product_code, branch_id, forecast_data, model_id):
        self.product_code = product_code
        self.branch_id = branch_id
        self.forecast_data = forecast_data
        self.model_id = model_id

class ForecastingModelSchema(marsh.Schema):
    class Meta:
        fields = ('id', 'algorithm_name', 'model_name', 'is_retrained', 'retrained_year', 'training_data', 'start_testing_data', 'model_accuracy','is_used', 'is_active', 'csv_filename', 'date_created')

class Forecasting_model(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    algorithm_name = db.Column(db.String(255))
    model_name = db.Column(db.String(255))
    is_retrained = db.Column(db.Integer, default=0)
    retrained_year = db.Column(db.String(255))
    training_data = db.Column(db.String(255))
    start_testing_data = db.Column(db.String(255), default=0)
    model_accuracy = db.Column(db.String(10))
    is_used = db.Column(db.Integer, default=0)
    is_active = db.Column(db.Integer, default=1)
    csv_filename = db.Column(db.String(255), default=0)
    date_created = db.Column(db.DateTime(timezone=True), default=func.now())


    def __init__(self, algorithm_name, model_name, is_retrained, retrained_year, training_data, start_testing_data, model_accuracy, is_used, is_active, csv_filename, date_created):
        self.algorithm_name = algorithm_name
        self.model_name = model_name
        self.is_retrained = is_retrained
        self.retrained_year = retrained_year
        self.training_data = training_data
        self.start_testing_data = start_testing_data
        self.model_accuracy = model_accuracy
        self.is_used = is_used
        self.is_active = is_active
        self.csv_filename = csv_filename
        self.date_created = date_created

class Validation_dataSchema(marsh.Schema):
    class Meta:
        fields = ('id', 'branch_id', 'file_name', 'is_used')

class Validation_data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    branch_id = db.Column(db.Integer)
    file_name = db.Column(db.String(255))
    is_used = db.Column(db.Integer, default=0)
    date_created = db.Column(db.DateTime(timezone=True), default=func.now())


    def __init__(self, branch_id, file_name, is_used):
        self.branch_id = branch_id
        self.file_name = file_name
        self.is_used = is_used

class Retraining_dataSchema(marsh.Schema):
    class Meta:
        fields = ('id', 'branch_id', 'file_name', 'is_used')

class Retraining_data(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    branch_id = db.Column(db.Integer)
    file_name = db.Column(db.String(255))
    is_used = db.Column(db.Integer, default=0)
    date_created = db.Column(db.DateTime(timezone=True), default=func.now())


    def __init__(self, branch_id, file_name, is_used):
        self.branch_id = branch_id
        self.file_name = file_name
        self.is_used = is_used

class SettingsSchema(marsh.Schema):
    class Meta:
        fields = ('id', 'active_upload_tab', 'temp_final_data', 'other1')

class Settings(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    active_upload_tab = db.Column(db.Integer, default=0)
    temp_final_data = db.Column(db.String(255), default=0)
    other1 = db.Column(db.Boolean, default=0)
    # date_created = db.Column(db.DateTime(timezone=True), default=func.now())


    def __init__(self, active_upload_tab, temp_final_data, other1):
        self.active_upload_tab = active_upload_tab
        self.temp_final_data = temp_final_data
        self.other1 = other1