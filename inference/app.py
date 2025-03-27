from fastapi import FastAPI, Request
from typing import Union, Optional, List
from pydantic import BaseModel

import os, logging
import mlflow

import torch

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print('Using computing device:', DEVICE)
PYTHON_PORT = int(os.environ.get('PYTHON_PORT', None))
MODEL_NAME = os.environ.get('MODEL_NAME', None)
MLFLOW_TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', None)

app = FastAPI()

mlflow.set_tracking_uri(uri=MLFLOW_TRACKING_URI)
loaded_model = mlflow.pytorch.load_model( f"models:/{MODEL_NAME}/latest", map_location=DEVICE)
loaded_model.device = DEVICE

def predict(objects: dict) -> dict:

    print(objects['texts'])
    return loaded_model.predict(objects['texts'])


@app.post("/api/predict")
def inference(request: dict) -> Union[dict,None]:

    response = None
    try:
        response = predict(objects = request)
        status = 200
    except Exception as e:
        logging.error(e)
        status = 500

    return {'response': [int(i) for i in response], 'status': status}

@app.get("/health")
def health():
    return {"health": "UP"}

if __name__ == "__main__":
    import uvicorn
    log_config = uvicorn.config.LOGGING_CONFIG
    log_config["formatters"]["access"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    log_config["formatters"]["default"]["fmt"] = "%(asctime)s - %(levelname)s - %(message)s"
    uvicorn.run(app, host="0.0.0.0", port=PYTHON_PORT, log_config=log_config)