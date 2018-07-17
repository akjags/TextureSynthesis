import numpy as np
import os

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

#import seaborn as sns

###################
#
# split training data into test and train
def split_train_data(X, Y):
  y0_idx = np.where(Y==0)[0]
  y1_idx = np.where(Y==1)[0]

  np.random.shuffle(y0_idx)
  np.random.shuffle(y1_idx)

  # Select 1000 trials of 1 and 1000 trials of 0 for training.
  X_train = np.concatenate((X[y1_idx[:1000],:], X[y0_idx[:1000],:]))
  y_train = np.concatenate((Y[y1_idx[:1000]], Y[y0_idx[:1000]]))
  X_test = np.concatenate((X[y1_idx[1000:], :], X[y0_idx[1000:],:]))
  y_test = np.concatenate((Y[y1_idx[1000:]], Y[y0_idx[1000:]]))

  return X_train, X_test, y_train, y_test

###############################################

# Main code

#############################################
### DEFINE WORKING DIRECTORY
wd = '/home/users/akshayj/TextureSynthesis'
gmdir = '/scratch/PI/jlg/gram_mtxs'

# Function shortcuts
npand = np.logical_and
sqz = np.squeeze

### Load behavioral data
bd = loadmat(wd + '/behav_data/s352.mat')
td = np.load(wd+'/behav_data/trialDistances.npy')

# Preprocess the behavioral data a bit.
fix_str_vec = lambda vec: np.array([str(vec[0][i][0]) for i in range(len(vec[0]))], dtype='string_')
for field in ['imNames', 'rfNames', 'layerNames']:
  bd[field] = fix_str_vec(bd[field])
for field in ['layer', 'image', 'rf_size', 'corr_trials', 'ecc']:
  bd[field] = sqz(bd[field])
bd['rf_sz_deg'] = 6.0 / bd['rf_size']
bd['nLayers'] = len(np.unique(bd['layer']))
bd['nRfSizes'] = len(np.unique(bd['rf_size']))

nTrials = int(bd['nTrials'])
#nTrials = 10000

# Observer model
obs_RFs = bd['rfNames']
obs_layers = bd['layerNames']
# Split out by eccentricity
all_eccs = np.unique(bd['ecc'])

r2s = np.zeros((len(all_eccs), len(obs_RFs), len(obs_layers)))

correct = bd['corr_trials'][:nTrials]
# Loop through each eccentricity, fit separate model for data from each ecc.
for eccI in range(len(all_eccs)):
  this_ecc = all_eccs[eccI]
  whichEccs = (bd['ecc'] == this_ecc)[:nTrials]
  corr_ecc = correct[whichEccs] # Take the subset of trainlabels with this ecc.
    
  cntr = 1
  printStrs = []
  print '~~~ Eccentricity: %d degrees' % (this_ecc)
  for rfI in range(len(obs_RFs)):
    obs_rf = obs_RFs[rfI]
    for layI in range(len(obs_layers)):
      obs_lay = obs_layers[layI]
      trialDist = td[layI, rfI, whichEccs].reshape(-1,1) # Take the subset of trials with this ecc
 
      # Get all variables
      #trialDist = td[:,:,whichEccs].reshape(len(obs_RFs)*len(obs_layers), np.sum(whichEccs)).T

      # Split data into train and test sets, then train logistic regression
      X_train, X_test, y_train, y_test = train_test_split(trialDist, corr_ecc, test_size=.1)
      logReg = LogisticRegression(penalty='l1', class_weight='balanced')

      # Calculate accuracy on the held-out test set.
      logReg.fit(X_train, y_train)
      preds = logReg.predict(X_test)
      probs = logReg.predict_proba(X_test)
      accuracy = 100.0*np.sum(y_test==preds) / len(y_test)

      r2 = (np.corrcoef(probs[:,1], y_test)[0,1])**2
      r2s[eccI,rfI,layI] = r2

      ll = log_loss(y_test, probs)

      printStr = 'Model %2i: Observer Layer = %s, RFSize = %s : \t Accuracy = %.2f%%; R2 = %.3f; LogLoss = %.3f' % (cntr, obs_lay, obs_rf, accuracy, r2, ll)
      print printStr
      printStrs.append(printStr)
 
      cntr = cntr + 1

      #if not (np.sum(preds) == len(preds) or np.sum(preds) == 0):
      #  print('Hooray! Model isnt guessing all 1s or all 0s')
  r2 = np.ravel(r2s[eccI,:,:])
  print 'BEST Model: %s' % (printStrs[np.argmax(r2)])
