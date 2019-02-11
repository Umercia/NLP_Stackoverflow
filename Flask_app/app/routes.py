## Home page route
# The routes are the different URLs that the application implements. In Flask,
# handlers for the application routes are written as Python functions, called
# view functions. View functions are mapped to one or more route URLs so that
# Flask knows what logic to execute when a client requests a given URL.

from flask import render_template
from flask import flash, redirect
from app import app
from app.forms import PostForm   # our form created in forms.py
from .utils import tagline

@app.route('/', methods=['GET', 'POST'])
@app.route('/tag_me', methods=['GET', 'POST'])                                  # tells Flask that this view function accepts GET and POST requests, overriding the default, which is to accept only GET requests
def login():                                                                    # OST requests are typically used when the browser submits form data to the server
    form = PostForm()
    if form.validate_on_submit():
        tag_pred_sup = tagline(form.title.data, form.content.data, model_type="sup")
        tag_pred_uns = tagline(form.title.data, form.content.data, model_type="uns")       # if at least one field fails validation, then the function will return False
        tag_pred_sem = tagline(form.title.data, form.content.data, model_type="sem")
        flash("Supervised model tag suggestion: {}".format(tag_pred_sup))                  # When you call the flash() function, Flask stores the message, but flashed messages will not magically appear in web pages. The templates of the application need to render these flashed messages in a way that works for the site layout
        # flash("semi-supervised model tag suggestion: {}".format(tag_pred_sem))
        flash("Unsupervised model tag suggestion: {}".format(tag_pred_uns))
        # return redirect('/tag_me')                                               # This function instructs the client web browser to automatically navigate to a different page, given as an argument
    return render_template('tag_me.html', title='tag me', form=form)
