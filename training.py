import random; random.seed(0)
import numpy as np; np.random.seed(0)

from models import train_model_dev, SeqModel
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
    
def champion_callback(study, frozen_trial):
    winner = study.user_attrs.get("winner", None)

    if study.best_value and winner != study.best_value:
        study.set_user_attr("winner", study.best_value)
        if winner:
            print(
                f"Trial {frozen_trial.number:2d} achieved value: {frozen_trial.value:.3f}"
            )
        else:
            print(f"Initial trial {frozen_trial.number} achieved value: {frozen_trial.value}")
    

def run_training(datatrain, datadev, settings : dict ) -> dict:


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

def optuna_reward( trial: optuna.Trial, settings: dict, datatrain: pd.DataFrame, 
                  datadev: pd.DataFrame) -> float:
    
    with mlflow.start_run(nested=True):

        hyperparameters = {
            "interm_layer_size": trial.suggest_int('interm_layer_size', 1, 100),
            "lr": trial.suggest_float('lr', 1e-6, 5e-5),
            "decay": trial.suggest_float('decay', 1e-6, 1e-3),
        }

        mlflow.log_params(hyperparameters)
        history = run_training(datatrain, datadev, settings | hyperparameters)

        return max(history['dev_acc'])

if __name__ == '__main__':

    optuna.logging.set_verbosity(optuna.logging.ERROR)

    #load data
    df_test = pd.read_csv('dataset/test.tsv', sep='\t')
    df_train = pd.read_csv('dataset/train.tsv', sep='\t')

    mlflow.set_tracking_uri(uri='http://localhost:8080')
    mlflow.set_experiment('offensiveval')

    with mlflow.start_run(experiment_id=get_or_create_experiment('offensiveval'),
                          run_name='optuna', nested=True):
        settings = {'model_name': 'bert-base-uncased',
                    'task': 'offensive',
                    'epoch': 24,
                    'batch_size': 32,
                    'output': '.'}
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: optuna_reward(trial, settings, df_train, df_test), n_trials=10)

        mlflow.log_params(study.best_params)
        mlflow.log_metric('best_f1', study.best_value)
        _ = run_training(df_train, df_test, settings | study.best_params)

        model = SeqModel(study.best_params['interm_layer_size'], 
                            settings['model_name'], 
                            settings['task'])
        
        model.load(os.path.join(settings['output'], f"{settings['model_name'].split('/')[-1]}_best.pt"))
        signature = mlflow.models.signature.infer_signature(df_train['text'].to_list(), 
                                                    model.predict(data = df_train['text'].to_list()))

        model_info = mlflow.pytorch.log_model(model, artifact_path="ofenseval_learn", 
                                        signature=signature,
                                        registered_model_name="offenseval_learn_quickstart")


    ## run optuna
    ## take the best hyperparameters
    ## run that model and save the weights