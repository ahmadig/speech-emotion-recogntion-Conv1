import librosa
import os
import numpy as np
import joblib
# dataset path
path = '\\Users\\ahmad\\SER\\Dataset\\' 
lst = []

count = 0

for subdir, dirs, files in os.walk(path):
  for file in files:
      try:
        #Load librosa array, obtain mfcss, store the file and the mcss information in a new array
        X, sample_rate = librosa.load(os.path.join(subdir,file), res_type='kaiser_fast')
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0) 
        # The instruction below converts the labels (from 1 to 8) to a series from 0 to 7
        # This is because our predictor needs to start from 0 otherwise it will try to predict also 0.
        file = int(file[7:8]) - 1 
        count = count+1
        print("file : " + str(count) + " / 5252")
        arr = mfccs, file
        lst.append(arr)
      # If the file is not valid, skip it
      except ValueError:
        continue


# we use zip to take each sounds array in lst and where it belongs to(class type) and make a new array of 5252
X, y = zip(*lst)
# to convert array into 5252,40
X = np.asarray(X)
# to convert array into 5252,
y = np.asarray(y)

# save the data into joblib file to load the data faster.
X_name = 'X.joblib'
y_name = 'y.joblib'
save_dir = '\\Users\\ahmad\\SER\\joblib\\'
savedX = joblib.dump(X, os.path.join(save_dir, X_name))
savedy = joblib.dump(y, os.path.join(save_dir, y_name))






