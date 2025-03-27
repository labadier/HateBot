import pytest, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'training'))
from models.SeqModel import SeqModel

def test_train_model():

    model = SeqModel(32, 'prajjwal1/bert-tiny', 'label')
        
    model.load('bert-tiny_best.pt')
    x = model.predict(data = ['hate', 'love'])
    
    assert len(x) == 2
    assert x[0] == 1 and x[1] == 0