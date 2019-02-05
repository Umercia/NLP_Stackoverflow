## Flask application instance
# In Python, a sub-directory that includes a __init__.py file is considered a
# package, and can be imported. When you import a package, the __init__.py
# executes and defines what symbols the package exposes to the outside world.


from flask import Flask
from config import Config  # contain variables to run our flask app

app = Flask(__name__)
app.config.from_object(Config)

from app import routes
