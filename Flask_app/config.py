# There are several formats for the application to specify configuration
# options. The most basic solution is to define your variables as keys in
# app.config, which uses a dictionary style to work with variables

import os

class Config(object):
    # to avoid to hard code the secret key ionside the application we prefer
    # to point to an external variable using os.environ.get("SECRET_KEY")
    SECRET_KEY = os.environ.get("SECRET_KEY") or "you-will-never-guess"
