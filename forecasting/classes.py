from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, PasswordField, EmailField, FileField, HiddenField
from wtforms.validators import ValidationError, DataRequired, InputRequired, Email

class GetLoginDetails(FlaskForm):
    email = EmailField(label=('Email'), validators=[DataRequired(), Email()])
    password = PasswordField(label=('Password'), validators=[DataRequired()])
    submit = SubmitField(label=('Submit'))

class ForgotPasswordForm(FlaskForm):
    email = EmailField(label=('Email'), validators=[DataRequired(), Email()])
    submit = SubmitField(label=('Submit'))

class UploadCSV(FlaskForm):
    # csrf_ = HiddenField(id="csrf_token", name="csrf_token", default="{{ csrf_token() }}")
    file = FileField(label=('File'), id="file", name="file", render_kw={'@input':'onSelectedFile'})
    submit = SubmitField(label=('Submit'))

class UploadValidationCSV(FlaskForm):
    # csrf_ = HiddenField(id="csrf_token", name="csrf_token", default="{{ csrf_token() }}")
    file = FileField(label=('File'), id="file_validation", name="file", render_kw={'@input':'onSelectedFileValidation'})
    submit = SubmitField(label=('Submit'))

class UploadRetrainingCSV(FlaskForm):
    # csrf_ = HiddenField(id="csrf_token", name="csrf_token", default="{{ csrf_token() }}")
    file = FileField(label=('File'), id="file_retraining", name="file", render_kw={'@input':'onSelectedFileRetraining', 'multiple': True})
    submit = SubmitField(label=('Submit'))