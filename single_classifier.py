import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import models
import data3
from data3 import *
import losses
import torch.nn as nn
import umap
import pdb
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report
import copy
import tqdm
import scanpy as sc
import collections

import gc

gc.collect()


# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device='cpu'
torch.cuda.empty_cache()
# Define hyperparameters and other settings
input_dim = 32738
hidden_dim = 70
z_dim = 1
epochs = 1001

np.random.seed(4)

model = models.binary_ANN(input_dim, hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-3)
train_loader = data3.loader 

def ANN_train(model, data_train, labels_train, data_val, labels_val):
    model.to(device) 

    data_train = data_train.to(device)  
    labels_train = labels_train.to(device)
    data_val = data_val.to(device)  
    labels_val = labels_val.to(device) 

    model.train()
    train_loss = 0

    n_epochs = 100   # number of epochs to run
    batch_size = 20  # size of each batch
 
    # Hold the best model
    best_acc = -np.inf   # init to negative infinity
    best_weights = None
 
    for epoch in range(n_epochs):
        
        if (epoch+1) % 10 == 0:
            print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}')
        
    
        if (epoch+1) % 500 == 0:
            # save model checkpoint
            # torch.save(model.cpu().state_dict(), f"model_{epoch+1}.pt")
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
all_labels_val = []
all_labels_pred = []

#turn into numpy arrays then back into tensors eventually
for train, test in kfold.split(data.cpu(), labels.cpu()):
    # create model, train, and get accuracy
    acc = ANN_train(model, data[train], labels[train], data[test], labels[test])
    print("Accuracy (wide): %.2f" % acc)
    cv_scores.append(acc)
    model.eval()
    labels_val_fold = labels[test]
    labels_pred_fold = model(data[test]).round().cpu().detach().numpy()

    all_labels_val.extend(labels_val_fold)
    all_labels_pred.extend(labels_pred_fold)

    #size of all_labels_val
    print(len(all_labels_val))

# Convert lists to numpy arrays
all_labels_val_np = [tensor.cpu().numpy() for tensor in all_labels_val]
all_labels_pred = np.array(all_labels_pred)

# Compute the overall confusion matrix
conf_matrix_overall = confusion_matrix(all_labels_val_np, all_labels_pred)

# Print the overall confusion matrix
print("Overall Confusion Matrix:")
print(pd.DataFrame(conf_matrix_overall, columns=['Predicted Generated', 'Predicted Real'], index=['Actual Generated', 'Actual Real']))

# Print classification report for more detailed information
print("Classification Report:")
print(classification_report(all_labels_val_np, all_labels_pred))

#ROC Curve
from sklearn.metrics import roc_curve, auc
print("plotting ROC Curve")
fpr, tpr, thresholds = roc_curve(all_labels_val_np, all_labels_pred)
roc_auc = auc(fpr, tpr)

print("FPR:", fpr)
print("TPR:", tpr)
print("Thresholds:", thresholds)
print("ROC AUC:", roc_auc)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Real Data (area = %0.2f)' % roc_auc)
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([-0.05,1.05])
plt.ylim([-0.05,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig("B2SC/figures/ROC_curve.png")
plt.show()




# evaluate the model
acc = np.mean(cv_scores)
std = np.std(cv_scores)
print("Model accuracy: %.2f%% (+/- %.2f%%)" % (acc*100, std*100))