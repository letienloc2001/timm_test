# import os
# from werkzeug.utils import import_string

# from flask import Flask
# from flask_cors import CORS
# from flask_jwt_extended import JWTManager
# from flask_sqlalchemy import SQLAlchemy
# from flask_migrate import Migrate

# from app.controllers.quality_image import quality_image_bp
# from app.controllers.classification import classification_bp
# from app.controllers.detection import detection_bp

# db = SQLAlchemy()
# migrate = Migrate()
# jwt = JWTManager()

# def create_app():
#     app = Flask(__name__)
    
#     cfg = import_string(f'settings.{os.getenv("FLASK_ENV").capitalize()}Config')
#     app.config.from_object(cfg)
    
#     from app import models
#     db.init_app(app)
#     migrate.init_app(app, db)
#     jwt.init_app(app)

#     api_prefix = app.config["API_PREFIX"]
#     CORS(app, resources={f"{api_prefix}": {"origins": app.config['BACKEND_URL']}})
    
#     from app.services import init_models
#     app.models = init_models(app.config["MODELS_CONFIG"])
        
#     # Import a module / component using its blueprint handler variable
#     app.register_blueprint(quality_image_bp, url_prefix=api_prefix)
#     app.register_blueprint(classification_bp, url_prefix=api_prefix)
#     app.register_blueprint(detection_bp, url_prefix=api_prefix)
#     return app
    
    
from flask import Flask
from app.controllers.quality_image import quality_image_bp
from app.controllers.classification import classification_bp
from app.controllers.detection import detection_bp
from app.services import init_models
# from flask_sqlalchemy import SQLAlchemy
# from flask_migrate import Migrate

CONFIG_FILENAME = "config" # TODO: move to os env
URL_PREFIX = "/coreai/v1"

MODELS_CONFIG = {
    "PRODUCT_DETECTION": {
        "instance": "InferenceDetection",
        "checkpoint_path": "yolov5l-nutifood.pt",
    },
    "SHELF_DETECTION": {
        "instance": "InferenceDetection",
        "checkpoint_path": "yolov5s-nutifood-shelf.pt",
    },
    "CAN_DETECTION": {
        "instance": "InferenceDetection",
        "checkpoint_path": "yolov5s-milk-can.pt",
    },
    "ANTI_SPOOFING_CLASSIFICATION": {
        "instance": "Classification",
        "checkpoint_path": "",
        # "checkpoint_path": "mixnet_s-anti-spoofing.pth",
    },
}

def create_app():
    # app = Flask(__name__)
    # app.config.from_object(CONFIG_FILENAME)
    # app.secret_key = app.config["SECRET_KEY"]
    
    models = init_models(MODELS_CONFIG)
        
    # Import a module / component using its blueprint handler variable
    # app.register_blueprint(quality_image_bp, url_prefix=URL_PREFIX)
    # app.register_blueprint(classification_bp, url_prefix=URL_PREFIX)
    # app.register_blueprint(detection_bp, url_prefix=URL_PREFIX)
    return models

# create_app()