import numpy as np
import os

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from scipy.io import loadmat
from sklearn.model_selection import train_test_split

#import seaborn as sns

### DEFINE WORKING DIRECTORY
wd = '/home/users/akshayj/TextureSynthesis'
gmdir = '/scratch/PI/jlg/gram_mtxs'

# Function shortcuts
npand = np.logical_and
sqz = np.squeeze

### Load behavioral data
bd = loadmat(wd + '/behav_data/s352.mat')

# Preprocess the behavioral data a bit.
fix_str_vec = lambda vec: np.array([str(vec[0][i][0]) for i in range(len(vec[0]))], dtype='string_')
for field in ['imNames', 'rfNames', 'layerNames']:
  bd[field] = fix_str_vec(bd[field])
for field in ['layer', 'image', 'rf_size', 'corr_trials']:
  bd[field] = sqz(bd[field])
bd['rf_sz_deg'] = 6.0 / bd['rf_size']
bd['nLayers'] = len(np.unique(bd['layer']))
bd['nRfSizes'] = len(np.unique(bd['rf_size']))

nTrials = int(bd['nTrials'])
#nTrials = 1000

# Observer model
obs_RFs = bd['rfNames']
obs_rf = obs_RFs[3] # Set observer receptive field size to 4x4
obs_layers = bd['layerNames']
obs_lay = obs_layers[2] # Set observer layer to Pool 4
distIdx = 1 # Get only first distractor

td = np.zeros((len(obs_layers), len(obs_RFs), nTrials))
cntr = 1;
for rfI in range(len(obs_RFs)):
  obs_rf = obs_RFs[rfI]
  for layI in range(len(obs_layers)):
    obs_lay = obs_layers[layI]
    print '---Model %i: Observer Layer = %s, RFSize = %s---' % (cntr, obs_lay, obs_rf)
    cntr = cntr + 1
    trialDist = np.zeros(nTrials)
    for i in range(nTrials):

      # Extract this trial's image, layer, and distractor rfSize
      thisRF = bd['rfNames'][bd['rf_size'][i]-1]
      thisLayer = bd['layerNames'][bd['layer'][i]-1]
      thisImg = bd['imNames'][bd['image'][i]-1]

      # Load gram matrix for original image
      origGM = np.load(gmdir + '/gram_' + obs_rf + '_0_' + thisImg + '.npy').item()
      distGM = np.load(gmdir + '/gram_' + obs_rf + '_' + str(distIdx) + '_' + thisRF +\
                        '_' + thisLayer + '_' + thisImg + '.npy').item()

      # Compute vector distance between gram matrix of original and distractor
      origGM_ol = origGM[obs_lay]
      distGM_ol = distGM[obs_lay]
      nGMs = origGM_ol.shape[2]
      dist = np.zeros(nGMs)
      for j in range(nGMs): # iterate through each gram matrix
        thisOrig = np.ravel(origGM_ol[:,:,j])
        thisDist = np.ravel(distGM_ol[:,:,j])
        dist[j] = np.linalg.norm(thisOrig - thisDist)

      trialDist[i] = np.mean(dist)
      
      if i % 100 == 0:
        print 'Trial #:', i+1, '; Image:', thisImg, '; Layer:', thisLayer, \
              '; RFSize:', thisRF, '; Distance:', trialDist[i]
    trialDist = trialDist.reshape(-1,1)
    correct = bd['corr_trials'][:nTrials]

    X_train, X_test, y_train, y_test = train_test_split(trialDist, correct)
    logReg = LogisticRegression()

    logReg.fit(X_train, y_train)
    preds = logReg.predict(X_test)
    accuracy = 100.0*np.sum(y_test==preds) / len(y_test)

    print 'Accuracy: %g %%' % (accuracy)

    td[layI, rfI, :] = np.ravel(trialDist)

np.save(wd+'/behav_data/trialDistances.npy', td)

