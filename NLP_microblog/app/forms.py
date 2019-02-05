# module to store my web form classes

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField
from wtforms import TextAreaField
from wtforms.validators import DataRequired

class PostForm(FlaskForm):
    title = StringField('Title:', validators=[DataRequired()])
    content = TextAreaField('Content:',
                            id="content-area",
                            validators=[DataRequired()])
    submit = SubmitField('Tag me')
