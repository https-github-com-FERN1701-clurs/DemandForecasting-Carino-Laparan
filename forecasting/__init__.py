from flask import Flask, session
from flask_sqlalchemy import SQLAlchemy
from .secrets import DB_HOST, DB_USERNAME, DB_PASSWORD, DB_DATABASENAME, DB_PORT, UPLOAD_FOLDER, SECRET_KEY,ALLOWED_EXTENSIONS
from os import path
from flask_login import LoginManager
from flask_marshmallow import Marshmallow
from flask_wtf.csrf import CSRFProtect
from datetime import timedelta
from flask_mail import Mail

db = SQLAlchemy()
marsh = Marshmallow()
csrf = CSRFProtect()
mail = Mail()
app = Flask(__name__)
DB_NAME = "database.db"
conn = "mysql+pymysql://{0}:{1}@{2}:{3}/{4}".format(DB_USERNAME, DB_PASSWORD, DB_HOST, DB_PORT, DB_DATABASENAME)
# conn = "mysql+mysqldb://{0}:{1}@{2}:{3}/{4}".format(DB_USERNAME, DB_PASSWORD, DB_HOST, DB_PORT, DB_DATABASENAME)

def create_app():
    conn
    # app = Flask(__name__)
    app.config['SECRET_KEY'] = SECRET_KEY
    app.config['SQLALCHEMY_DATABASE_URI'] = conn
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    # app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['SQLALCHEMY_POOL_RECYCLE'] = 100
    app.config['SQLALCHEMY_POOL_TIMEOUT'] = 600

    app.config['MAIL_SERVER']='mail.wetechsupport.online'
    app.config['MAIL_PORT'] = 465
    app.config['MAIL_USERNAME'] = 'support@wetechsupport.online'
    app.config['MAIL_PASSWORD'] = 'superadmin2022'
    app.config['MAIL_USE_TLS'] = False
    app.config['MAIL_USE_SSL'] = True

    # db = SQLAlchemy(app)
    db.init_app(app)
    marsh.init_app(app)
    csrf.init_app(app)
    mail.init_app(app)

    from .views import views
    from .auth import auth
    from .forecasting import forecast

    app.register_blueprint(views, url_prefix='/')
    app.register_blueprint(auth, url_prefix='/')
    app.register_blueprint(forecast, url_prefix='/')

    from .models import User
    from .models import Permissions
    from .models import Uploads
    create_database(app)

    login_manager = LoginManager()
    login_manager.login_view = 'auth.login'
    login_manager.init_app(app)

    login_manager.refresh_view = 'auth.login'
    login_manager.needs_refresh_message = (u"Session timedout, please re-login")
    login_manager.needs_refresh_message_category = "info"

    # @app.before_request
    # def before_request():
    #     session.permanent = True
    #     app.permanent_session_lifetime = timedelta(minutes=1)

    @login_manager.user_loader
    def load_user(id):
        return User.query.get(int(id))

    return app

def create_database(app):
    if not path.exists('forecasting/' + DB_NAME):
        db.create_all(app=app)
        print('Created Database!')