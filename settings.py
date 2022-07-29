from os import environ 
from os.path import join, dirname
from dotenv import load_dotenv

load_dotenv(join(dirname(__file__), f'.env.{environ.get("FLASK_ENV")}'))

class Config():
    API_PREFIX = '/coreai/v1'
    SECRET_KEY = environ.get('SECRET_KEY')
    
    DB_USER = environ.get("DB_USER")
    DB_PASSWORD = environ.get("DB_PASSWORD")
    DB_HOST = environ.get("DB_HOST")
    DB_NAME = environ.get("DB_NAME")
    SQLALCHEMY_DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}"
    
    #TODO: move to DB
    MODELS_CONFIG = {
        "PRODUCT_DETECTION": {
            "instance": "InferenceDetection",
            "checkpoint_path": "/tessel_coreai/checkpoint/yolov5l-nutifood.pt",
        },
        "SHELF_DETECTION": {
            "instance": "InferenceDetection",
            "checkpoint_path": "/tessel_coreai/checkpoint/yolov5s-nutifood-shelf.pt",
        },
        "CAN_DETECTION": {
            "instance": "InferenceDetection",
            "checkpoint_path": "/tessel_coreai/checkpoint/yolov5s-milk-can.pt",
        },
        "ANTI_SPOOFING_CLASSIFICATION": {
            "instance": "InferenceClassification",
            "checkpoint_path": "/tessel_coreai/checkpoint/mixnet_s-anti-spoofing.pth",
        },
    }

class DevelopmentConfig(Config):
    FRONTEND_URL = environ.get("FRONTEND_URL")
    COREAI_URL = environ.get("COREAI_URL")
    BACKEND_URL = environ.get("BACKEND_URL")
    
    SQLALCHEMY_ECHO = True
    SQLALCHEMY_TRACK_MODIFICATIONS = False

class ProductionConfig(Config):
    FLASK_ENV = 'production'