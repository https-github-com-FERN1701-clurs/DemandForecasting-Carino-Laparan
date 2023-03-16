from flask import Blueprint, render_template, request, flash, redirect, url_for, session
from flask_login import login_user, login_required, logout_user, current_user
from pymysql import NULL
from sqlalchemy import null
from .models import User, Permissions, PermissionsSchema
from .functions import *
from .classes import GetLoginDetails
from . import db, mail
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import timedelta
from flask_mail import Message

auth = Blueprint('auth', __name__)

@auth.route('/', methods=['GET', 'POST'])
def login():
    form = GetLoginDetails()
    auth_user=current_user
    if auth_user.is_authenticated:
        return redirect(url_for('forecast.forecasting'))
    else:
        if form.validate_on_submit():
            if request.method == 'POST':
                #     email = request.form.get('email')
                #     password = request.form.get('password')
                #     print(generate_password_hash('carsetadmin', method="sha256"))
                # if check_email(form.email.data):
                user = User.query.filter_by(email=form.email.data).first()
                if user:
                    if check_password_hash(user.password, form.password.data):
                        session['current_menu'] = ''
                        login_user(user, remember=True)
                        return redirect(url_for('forecast.forecasting'))
                        # return redirect(url_for('views.dashboard'))
                    else:
                        flash('Invalid email or password', category='error')
                else:
                    flash('No record found', category='info')
    return render_template("/auth/login.html", form = form)

@auth.route('/logout')
@login_required
def logout():
    session.pop('current_menu', None)
    logout_user()
    return redirect(url_for('auth.login'))

@auth.route('/send-link', methods=['POST'])
def send_link(user):
    token = user.get_reset_token()
    print(token)
    msg = Message('Password Reset Request', sender='support@wetechsupport.online', recipients=[user.email])
    msg.body = f'''To reset your password, visit following link
    {url_for('.reset_token', token=token, _external=True)}
    If you did not make this request then simply ignore this email and no changes will be made
    '''
    mail.send(msg)

@auth.route('/forgot-password', methods=['GET', 'POST'])
def reset_request():
    if current_user.is_authenticated:
        return redirect(url_for('views.dashboard'))
    else:
        if request.method == 'POST':
            request_email = request.json
            user = User.query.filter_by(email=request_email['email']).first()
            if user == None:
                return ''
            send_link(user)
            return 'false' if user == None else 'true'
        return render_template("auth/forgot-password.html")

@auth.route('/forgot-password/<token>', methods=['GET', 'POST'])
def reset_token(token):
    if current_user.is_authenticated:
        return redirect(url_for('views.dashboard'))

    user = User.verify_reset_token(token)
    if user is None:
        flash('Link is invalid or expired', 'warning')
        return redirect(url_for('.reset_request'))

    if request.method == 'POST':
        # request_password = request.json
        # hashed_password = generate_password_hash(request_password['password'], method="sha256")
        hashed_password = generate_password_hash(request.form['password'], method="sha256")
        user.password = hashed_password
        user.fp_token = None
        db.session.commit()
        db.session.close()
        flash('Password Successfully Changed', 'info')
        return redirect(url_for('auth.login'))

    if request.method == 'GET':
        return render_template("auth/reset-password.html", rp_token=token)

@auth.route('/sign-up', methods=['GET','POST'])
def sign_up():
    if request.method == 'POST':
        email = request.form.get('email')
        firstname = request.form.get('first_name')
        password = request.form.get('password')
    return "<h1>Sign up Here</h1>"

"""
    FLASH MESSAGES
        1. import flash
        2. flash('message_str', category='any_category_name')
        3. display the message:
            {% with messages = get_flashed_messages(with_categories=true) %}
                {% if messages %}
                    {% for category, message in messages %}
                        <div>
                            {{ message }}
                            {{ category }}
                        </div>
                        
                    {% endfor %}
                {% endif %}
            {% endwith %}
            
    HASH PASSWORD
        password = generate_password_hash(password1, method="sha256")
    
    
    db.session(model)
    db.session.commit()
"""