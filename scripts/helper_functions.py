import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm

#from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import glob
from torch.utils.tensorboard import SummaryWriter
import random
from tqdm import tqdm

##################################################################################################
# First things first! Set a seed for reproducibility.
# https://www.cs.mcgill.ca/~ksinha4/practices_for_reproducibility/
def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def __get_device__() :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    print('Device available is', device)
    return device

def __contextual_rep__(device, train_path_stutter, train_path_fluent) :
    # Get wav2vec2.0
    bundle = torchaudio.pipelines.WAV2VEC2_BASE
    print("Sample Rate of model:", bundle.sample_rate)
    
    model_wav2vec = bundle.get_model().to(device)
    ## Convert audio to numpy to wav2vec feature encodings
    def conv_audio_data (filename) :
        waveform, sample_rate = torchaudio.load(filename)
        waveform = waveform.to(device)
        if sample_rate != bundle.sample_rate:
            print('Mismatched sample rate')
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)
        emission, _ = model_wav2vec(waveform)
        emission = emission.cpu().detach().numpy()
        return emission
    
    x_f = []
    y_f = []
    x_s = []
    y_s = []
    
    discarded_s = 0
    # Convert to the embeddings for training data
    for filename in glob.glob(os.path.join(train_path_stutter, '*.wav')):
        stutter_np = conv_audio_data(filename)
        # fluent_np --> (1, 149, 768)
        if ((np.shape(stutter_np)[0] != 1) |(np.shape(stutter_np)[1] != 149) | (np.shape(stutter_np)[2] != 768)) :
            discarded_s += 1
        else:
            x_s.append(stutter_np)
            y_s.append(1)
    
    discarded = 0
    for filename in glob.glob(os.path.join(train_path_fluent, '*.wav')):
        fluent_np = conv_audio_data(filename)
        # fluent_np --> (1, 149, 768)
        if ((np.shape(fluent_np)[0] != 1) |(np.shape(fluent_np)[1] != 149) | (np.shape(fluent_np)[2] != 768)) :
            discarded += 1
        else:
            x_f.append(fluent_np)
            y_f.append(0)
    return x_f, y_f, x_s, y_s     

def __contextual_rep_test__():
    x_t_f = []
    y_t_f = []
    x_t_s = []
    y_t_s = []
    
    discarded_t_s = 0
    # Convert to the embeddings for test data
    for filename in glob.glob(os.path.join(test_path_stutter, '*.wav')):
        stutter_np = conv_audio_data(filename)
        # stutter_np --> (1, 149, 768)
        if ((np.shape(stutter_np)[0] != 1) |(np.shape(stutter_np)[1] != 149) | (np.shape(stutter_np)[2] != 768)) :
            discarded_t_s += 1
        else:
            x_t_s.append(stutter_np)
            y_t_s.append(1)
    
    discarded_t = 0
    for filename in glob.glob(os.path.join(test_path_fluent, '*.wav')):
        fluent_np = conv_audio_data(filename)
        # fluent_np --> (1, 149, 768)
        if ((np.shape(fluent_np)[0] != 1) |(np.shape(fluent_np)[1] != 149) | (np.shape(fluent_np)[2] != 768)) :
            discarded_t += 1
        else:
            x_t_f.append(fluent_np)
            y_t_f.append(0)
    return x_t_f, y_t_f, x_t_s, y_t_s           


## Shuffle and pick a quarter of the data
def __shuffle_pick_quarter_data__ (x_f, y_f, x_s, y_s) :
    # Take only quarter dataset
    random.shuffle(x_f)
    random.shuffle(x_s)
    x_s_h = x_s[0:int(len(x_s)/4)]
    y_s_h = y_s[0:int(len(x_s)/4)]
    
    
    # Comment this for unblanced data
    # Stutter is less than fluent
    x_f_h = x_f[0:len(x_s_h)]
    y_f_h = y_f[0:len(x_s_h)]
    
    x_train = x_s_h + x_f_h
    y_train = y_s_h + y_f_h
    return x_train, y_train


## Pick test set in abalanced fashion
# Comment this for unblanced data
# Stutter is less than fluent
def __test_balanced_data__(x_t_f, x_t_s, y_t_f, y_t_s):
    random.shuffle(x_t_f)
    random.shuffle(x_t_s)
    x_t_f = x_t_f[0:len(x_t_s)]
    y_t_f = y_t_f[0:len(x_t_s)]
    
    x_test = x_t_s + x_t_f
    y_test = y_t_s + y_t_f

def train(epoch):
  print('\nEpoch : %d'%epoch)
  
  model.train()

  running_loss=0
  correct=0
  total=0

  for data in train_loader:
    
    inputs,labels=data[0].to(device),data[1].to(device)
    # forward pass
    outputs=model(inputs)
    loss=criterion(outputs,labels)

    # backward and optimise
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    
    # Compute Training Accuracy
    _, predicted_labels = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += predicted_labels.eq(labels).sum().item()
      
  train_loss=running_loss/len(train_loader)
  
  accu=100.*correct/total
  # visualisation
  writer.add_scalar("Loss/train", loss, epoch)  
  writer.add_scalar("Accuracy/train", accu, epoch)  
  
  train_accu.append(accu)
  train_losses.append(train_loss)
  print('Train Loss: %.3f | Accuracy: %.3f'%(train_loss,accu))

def test(epoch):
  model.eval()

  running_loss=0
  correct=0
  total=0

  with torch.no_grad():
    for data in valid_loader:
      features,labels=data[0].to(device),data[1].to(device)
      
      outputs=model(features)

      _, predicted_valid = torch.max(outputs.data, 1)

      loss= criterion(outputs,labels)
      running_loss+=loss.item()
      total += labels.size(0)
      correct += predicted_valid.eq(labels).sum().item()
  
  test_loss=running_loss/len(valid_loader)
  accu=100.*correct/total
  
  # visualisation
  writer.add_scalar("Loss/test", loss, epoch)  
  writer.add_scalar("Accuracy/test", accu, epoch)  

  eval_losses.append(test_loss)
  eval_accu.append(accu)

  print('Validation Loss: %.3f | Accuracy: %.3f'%(test_loss,accu))  