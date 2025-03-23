import random; random.seed(42)
import numpy as np; np.random.seed(42)
import torch; torch.manual_seed(42)
import os

from sklearn.model_selection import StratifiedKFold
from transformers import AutoModel, AutoTokenizer, BloomForCausalLM
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, mean_squared_error
from tqdm import tqdm

import pandas as pd
import mlflow

class bcolors:
  HEADER = '\033[95m'
  OKBLUE = '\033[94m'
  OKCYAN = '\033[96m'
  OKGREEN = '\033[92m'
  WARNING = '\033[93m'
  FAIL = '\033[91m'
  ENDC = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


def HugginFaceLoad(model_name):

  if 'bloom' in model_name:
    model = BloomForCausalLM.from_pretrained(model_name)
  else:
    model = AutoModel.from_pretrained(model_name)
    
  tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True, TOKENIZERS_PARALLELISM=True)

  return model, tokenizer


def seed_worker(worker_id):
  worker_seed = torch.initial_seed() % 2**32
  np.random.seed(worker_seed)
  random.seed(worker_seed)

class Data(Dataset):

  def __init__(self, data):

    self.data = data
    
  def __len__(self):
    return len(self.data['text'])

  def __getitem__(self, idx):

    if torch.is_tensor(idx):
      idx = idx.tolist()
      
    ret = {key: self.data.iloc[idx][key] for key in self.data.keys()}
    return ret
   

class SeqModel(torch.nn.Module):

  def __init__(self, interm_size, model, task):

    super(SeqModel, self).__init__()
		
    self.best_acc = None
    self.max_length = 128
    self.task = task
    self.interm_neurons = interm_size
    self.transformer, self.tokenizer = HugginFaceLoad( model )
    self.model = model
    self.intermediate = torch.nn.Sequential(torch.nn.Linear(in_features= 768 if 'bloom' not in model else 1536, out_features=self.interm_neurons), torch.nn.LeakyReLU(),
                                            torch.nn.Linear(in_features=self.interm_neurons, out_features=self.interm_neurons>>1),
                                            torch.nn.LeakyReLU())
    
    self.classifier = torch.nn.Linear(in_features=self.interm_neurons>>1, out_features=2)
    self.loss_criterion = torch.nn.CrossEntropyLoss()
    
    self.device = torch.device("cuda:0") if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    self.to(device=self.device)

  def forward(self, data):

    ids = self.tokenizer(data, return_tensors='pt', truncation=True, padding=True, max_length=self.max_length).to(device=self.device)

    if 'bloom' not in self.model:
      X = self.transformer(**ids)[0][:,0]
    else:
      X = self.transformer.transformer(**ids).last_hidden_state[:,-1,:]

    enc = self.intermediate(X)
    output = self.classifier(enc)

    return output 

  def load(self, path):
    print(f"{bcolors.OKCYAN}{bcolors.BOLD}Weights Loaded{bcolors.ENDC}") 
    self.load_state_dict(torch.load(path, map_location=self.device))

  def save(self, path):
    torch.save(self.state_dict(), path)

  def makeOptimizer(self, lr=1e-5, decay=2e-5, multiplier=1, increase=0.1):

    if 'bloom' in self.model:
      return torch.optim.RMSprop(self.parameters(), lr, weight_decay=decay)

    params = [] 
    for l in self.transformer.encoder.layer:
      
      params.append({'params':l.parameters(), 'lr':lr*multiplier}) 
      multiplier += increase

    try:
      params.append({'params':self.transformer.pooler.parameters(), 'lr':lr*multiplier})
    except:
      print(f'{bcolors.WARNING}Warning: No Pooler layer found{bcolors.ENDC}')

    params.append({'params':self.intermediate.parameters(), 'lr':lr*multiplier})
    params.append({'params':self.classifier.parameters(), 'lr':lr*multiplier})

    return torch.optim.RMSprop(params, lr=lr*multiplier, weight_decay=decay)
  
  def predict(self, 
    data: np.array,
    batch_size: int = 64):

    devloader = DataLoader(Data(pd.DataFrame({'text': data})),
                            batch_size=batch_size, shuffle=False)#, num_workers=4, worker_init_fn=seed_worker)
    itera = tqdm(enumerate(devloader, 0))

    running_stats = {'outputs':None, 'indexes':None}
    for j, data in itera:

        torch.cuda.empty_cache()            
        outputs = self.forward(data['text'])

        if running_stats['outputs'] is None:
          running_stats['outputs'] = outputs.detach().cpu()
          # running_stats['indexes'] = data['index']
        else:
          running_stats['outputs'] = torch.cat((running_stats['outputs'], outputs.detach().cpu()), dim=0)
          # running_stats['indexes'] = torch.cat((running_stats['indexes'], data['index']), dim=0)

    out = {'out': list(torch.max(running_stats['outputs'], 1).indices.detach().cpu().numpy())}#'index': list(running_stats['indexes'].detach().cpu().numpy()),
    return out['out']


def measurement(running_stats, task):
    
    score = -1
    p = torch.max(running_stats['outputs'], 1).indices.detach().cpu()
    l = running_stats['labels'].detach().cpu()
    score = f1_score(l, p)

    return score


def train_model(model_name, model, trainloader, devloader, epoches, lr, decay, output, task):
  
    eloss, eacc, edev_loss, edev_acc = [], [], [], []

    optimizer = model.makeOptimizer(lr=lr, decay=decay)
    batches = len(trainloader)

    for epoch in range(epoches):

        running_stats = {'outputs':None, 'labels':None}
        model.train()

        itera = tqdm(enumerate(trainloader, 0), total=batches)
        itera.set_description(f'Epoch: {epoch:3d}')

        for j, data in itera:

            torch.cuda.empty_cache()         
            labels = data[task].to(model.device)     

            optimizer.zero_grad()
            outputs = model(data['text'])
            loss = model.loss_criterion(outputs, labels)

            if running_stats['outputs'] is None:
                running_stats['outputs'] = outputs.detach().cpu()
                running_stats['labels'] = data[task]
            else:
                running_stats['outputs'] = torch.cat((running_stats['outputs'], outputs.detach().cpu()), dim=0)
                running_stats['labels'] = torch.cat((running_stats['labels'], data[task]), dim=0)

            loss.backward()
            optimizer.step()

            train_loss = model.loss_criterion(running_stats['outputs'], running_stats['labels']).item()
            train_measure = measurement(running_stats, task)
            itera.set_postfix_str(f"loss:{train_loss:.3f} measure:{train_measure:.3f}") 

            if j == batches-1:
                eloss += [train_loss]
                eacc += [train_measure]

                model.eval()
                with torch.no_grad():
                    
                    running_dev = {'outputs': None, 'labels': None}
                    for k, data_batch_dev in enumerate(devloader, 0):
                        torch.cuda.empty_cache() 
  
                        outputs = model(data_batch_dev['text'])
                        
                        if running_dev['outputs'] is None:
                            running_dev['outputs'] = outputs.detach().cpu()
                            running_dev['labels'] = data_batch_dev[task]
                        else:
                            running_dev['outputs'] = torch.cat((running_dev['outputs'], outputs.detach().cpu()), dim=0)
                            running_dev['labels'] = torch.cat((running_dev['labels'], data_batch_dev[task]), dim=0)

                    dev_loss = model.loss_criterion(running_dev['outputs'], running_dev['labels']).item()
                    dev_measure = measurement(running_dev, task)
                    
                if model.best_acc is None or model.best_acc < dev_measure:
                    model.save(os.path.join(output, f"{model_name.split('/')[-1]}_best.pt"))
                    model.best_acc = dev_measure

                mlflow.log_metric('train_loss', train_loss)
                mlflow.log_metric('train_f1', train_measure)
                mlflow.log_metric('dev_loss', dev_loss)
                mlflow.log_metric('dev_f1', dev_measure)

                itera.set_postfix_str(f"loss:{train_loss:.3f} measure:{train_measure:.3f} \
                                      dev_loss:{dev_loss:.3f} dev_measure: {dev_measure:.3f}") 
                edev_loss += [dev_loss]
                edev_acc += [dev_measure]
    return {'loss': eloss, 'acc': eacc, 'dev_loss': edev_loss, 'dev_acc': edev_acc}
        

def train_model_dev(model_name, data_train, data_dev, task = 'classification', epoches = 4, batch_size = 8, 
                    interm_layer_size = 64, lr = 1e-5,  decay=2e-5, output='logs'):

  history = []

  model = SeqModel(interm_layer_size, model_name, task)

  trainloader = DataLoader(Data(data_train), batch_size=batch_size, shuffle=True)#, num_workers=4, worker_init_fn=seed_worker)
  devloader = DataLoader(Data(data_dev), batch_size=batch_size, shuffle=False)#, num_workers=4, worker_init_fn=seed_worker)

  history.append(train_model(f'{model_name}', model, trainloader, devloader, epoches, lr, decay, output, task))

  del trainloader
  del model
  del devloader
  return history

   