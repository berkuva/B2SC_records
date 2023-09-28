import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import models
import data2
from data2 import *
import losses
import torch.nn as nn
import umap
import pdb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
import copy
import tqdm
import scanpy as sc
import collections

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'

# Define hyperparameters and other settings
input_dim = 32738
hidden_dim = 700
z_dim = 9
epochs = 1001

np.random.seed(4)

model = models.ANN(input_dim, hidden_dim, z_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
train_loader = data2.loader 
'''
def ANN_train(model, X_train, y_train, X_val, y_val):
    model.to(device) 
    train_loss = 0

    y_train = y_train.float()
    y_val = y_val.float()

    X_train = X_train.to(device)  
    y_train = y_train.to(device)
    X_val = X_val.to(device)  
    y_val = y_val.to(device) 
    

    # loss function and optimizer
    loss_fn = nn.BCELoss()  # binary cross entropy
    optimizer = optim.Adam(model.parameters(), lr=0.000001)

    n_epochs = 30   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)

    # Hold the best model
    best_acc = - np.inf   # init to negative infinity
    best_weights = None

    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            
            if (epoch+1) % 10 == 0:
                print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')

            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)


                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                train_loss += loss.item()
                optimizer.step()
                # print progress
                acc = (y_pred.round() == y_batch).float().mean()
                bar.set_postfix(
                    loss=float(loss),
                    acc=float(acc)
                )
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_val)
        acc = (y_pred.round() == y_val).float().mean()
        acc = float(acc)
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    return best_acc
    '''
def ANN_train(model, data_train, labels_train, data_val, labels_val):
    model.to(device) 

    data_train = data_train.to(device)  
    labels_train = labels_train.to(device)
    data_val = data_val.to(device)  
    labels_val = labels_val.to(device) 

    model.train()
    train_loss = 0

    n_epochs = 100   # number of epochs to run
    batch_size = 32  # size of each batch
 
    # Hold the best model
    best_acc = -np.inf   # init to negative infinity
    best_weights = None
 
    for epoch in range(n_epochs):
        
        if (epoch+1) % 10 == 0:
            print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
        
    
        if (epoch+1) % 500 == 0:
            # save model checkpoint
            torch.save(model.cpu().state_dict(), f"model_{epoch+1}.pt")
            model.to(device)  # Move the model back to the desired device

        model.train()

        # batch
        data_batch = data_train[epoch:epoch+batch_size]
        labels_batch = labels_train[epoch:epoch+batch_size]

        # forward pass
        y_pred = model(data_batch)
        loss = loss_fn(y_pred, labels_batch.float())

        # backprop
        optimizer.zero_grad()
        loss.backward()

        # update weights
        train_loss = 0
        train_loss += loss.item()
        optimizer.step()

        # print progress
        acc = (y_pred.round() == labels_batch).float().mean()

        # evaluate accuracy at end of each epoch
        model.eval()
        labels_pred = model(data_val)
        acc = (labels_pred.round() == labels_val).float().mean()
        acc = float(acc)
        
        #remember accuracy
        if acc > best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())
    # restore model and return best accuracy

    model.load_state_dict(best_weights)
    return best_acc





for batch_idx, (data,labels) in enumerate(train_loader):
        data = data.to(device)
        # model = nn.DataParallel(model).to(device)
        labels = labels.view(-1,1).to(device)
        optimizer.zero_grad()
        loss_fn = nn.BCELoss()  # binary cross entropy



kfold = StratifiedKFold(n_splits=5, shuffle=True)
cv_scores = []

#turn into numpy arrays then back into tensors eventually
for train, test in kfold.split(data.cpu(), labels.cpu()):
    # create model, train, and get accuracy
    model = models.ANN(input_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
    acc = ANN_train(model, data[train], labels[train], data[test], labels[test])
    print("Accuracy (wide): %.2f" % acc)
    cv_scores.append(acc)


# evaluate the model
acc = np.mean(cv_scores)
std = np.std(cv_scores)
print("Model accuracy: %.2f%% (+/- %.2f%%)" % (acc*100, std*100))