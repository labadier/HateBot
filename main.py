#%%
import random; random.seed(0)
import numpy as np; np.random.seed(0)

from models.SeqModel import train_model_dev, SeqModel
import pickle,  pandas as pd
import seaborn as sns

from tqdm import tqdm
import optuna, os
import mlflow

def get_or_create_experiment(experiment_name):
    """
    Retrieve the ID of an existing MLflow experiment or create a new one if it doesn't exist.

    This function checks if an experiment with the given name exists within MLflow.
    If it does, the function returns its ID. If not, it creates a new experiment
    with the provided name and returns its ID.

    Parameters:
    - experiment_name (str): Name of the MLflow experiment.

    Returns:
    - str: ID of the existing or newly created MLflow experiment.
    """

    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)

def run_training(datatrain: pd.DataFrame, datadev: pd.DataFrame, settings : dict) -> dict:


    history = train_model_dev(settings['model_name'],
                             data_train = datatrain,
                             data_dev = datadev, 
                             task = settings['task'], 
                             epoches = settings['epoch'],
                             batch_size = settings['batch_size'], 
                             interm_layer_size = settings['interm_layer_size'],
                             lr = settings['lr'],
                            decay=settings['decay'], output=settings['output'])
    return history




df_test = pd.read_csv('dataset/test.tsv', sep='\t').fillna(' ').sample(100)
df_train = pd.read_csv('dataset/train.tsv', sep='\t').fillna(' ').sample(100)


with mlflow.start_run() as mlrun:
        
    settings = {'model_name': 'bert-base-uncased',
                'task': 'offensive',
                'epoch': 2,
                'batch_size': 32,
                'output': '.'}

    mapping = {i:j for j, i in enumerate(sorted(df_train[settings['task']].unique()))}

    df_test[settings['task']] = df_test[settings['task']].map(mapping)
    df_train[settings['task']] = df_train[settings['task']].map(mapping)

    hyperparams = {}
    hyperparams['interm_layer_size'] = 256
    hyperparams['lr'] = 2e-5
    hyperparams['decay'] = 1e-6 

    mlflow.log_params(hyperparams)

    history = run_training(df_train, df_test, settings | hyperparams )

    # mlflow.log_metric('macro-f1', 0.6)#max(history[1]['dev_acc']))
    # mlflow.log_metric('macro-f1', 0.2)#max(history[1]['dev_acc']))
    # mlflow.log_metric('macro-f1', 0.3)#max(history[1]['dev_acc']))
    # mlflow.log_metric('macro-f1', 0.8)#max(history[1]['dev_acc']))

    model = SeqModel(hyperparams['interm_layer_size'], 
                        settings['model_name'], 
                        settings['task'])

    model.load(os.path.join(settings['output'], f"{settings['model_name'].split('/')[-1]}_best.pt"))
    signature = mlflow.models.signature.infer_signature(df_train['text'].to_list(), 
                                                        model.predict(data = df_train['text'].to_list()))

    model_info = mlflow.pytorch.log_model(model, artifact_path="ofenseval_learn", 
                                            signature=signature,
                                            registered_model_name="offenseval_learn_quickstart")


# %%
model_uri = "models:/offenseval_learn_quickstart/latest"

mlflow.set_tracking_uri(uri='http://localhost:8080')

experiment_id = get_or_create_experiment('hate_learn')
mlflow.set_experiment(experiment_id=experiment_id)

loaded_model = mlflow.pytorch.load_model(model_info.model_uri)
predictions = loaded_model.predict(df_test['text'].to_list())


# result = pd.DataFrame(X_test, columns=iris_feature_names)
# result["actual_class"] = y_test
# result["predicted_class"] = predictions

# %%
