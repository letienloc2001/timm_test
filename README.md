# TESSEL.CORE_AI
This repository is the Core AI Tessel Project for Viettel, containing 2 services for auto-training and inference models. 

Two services intergrates mutually by gRPC protocol (currently is HTTP) in 2 different servers.

COREAI require GPU computing for training and inferencing Deeplearning models. Please configure NVIDIA Driver first!

## How to start project
1- Create env by conda, docker or virtual env.

2- Export os environment parameter

For development environment
```bash
export FLASK_APP=app
export FLASK_RUN_PORT=8080
export FLASK_ENV=development
export FLASK_DEBUG=True
```

For production environment
```bash
export FLASK_APP=app
export FLASK_RUN_PORT=8080
export FLASK_ENV=production
export FLASK_DEBUG=False
```

3- Install packages by 
```bash
pip install -r requirements.txt
```

4- Import .env file to root tree and change config in environment file

5- Start service (will update migrate DB later)
```bash
flask run
```


