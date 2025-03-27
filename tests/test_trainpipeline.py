import pytest
import pandas as pd, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))
from models.SeqModel import train_model_dev


def test_train_model():

    data_train = pd.DataFrame({
        'text': ['I hate you', 'I love you']*10,
        'label': [1, 0]*10
    })

    x = train_model_dev('prajjwal1/bert-tiny',
                             data_train = data_train,
                             data_dev = data_train, 
                             task = 'label', 
                             epoches = 5,
                             batch_size = 4, 
                             interm_layer_size = 32,
                             lr = 1e-4,
                            decay=1e-4, output='.')[-1]
    
    assert max(x['dev_acc']) > 0.6