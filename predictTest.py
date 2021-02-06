import keras
import numpy as np
import librosa

class livePredictions:

# this method is used for init the path of file
    def __init__(self, path, file):
        self.path = path
        self.file = file

# this method used for load the model
    def load_model(self):
        self.loaded_model = keras.models.load_model(self.path)
        return self.loaded_model.summary()

# this method is used to make prediction of the input sound file
    def makepredictions(self):
        data, sampling_rate = librosa.load(self.file)
        mfccs = np.mean(librosa.feature.mfcc(y=data, sr=sampling_rate, n_mfcc=40).T, axis=0)
        x = np.expand_dims(mfccs, axis=1)
        x = np.expand_dims(x, axis=0)
        predictions = self.loaded_model.predict_classes(x)
        print("Prediction is", " ", self.convertclasstoemotion(predictions))

# this method is used for convert the class from 0-7 to the human reaction in words
    @staticmethod
    def convertclasstoemotion(pred):
        label_conversion = {'0': 'natural',
                            '1': 'calm',
                            '2': 'happy',
                            '3': 'sad',
                            '4': 'angry',
                            '5': 'fearful',
                            '6': 'disgust',
                            '7': 'surprised'}

        for key, value in label_conversion.items():
            if int(key) == pred:
                label = value
        return label
    

# the model path
modelPath='\\Users\\ahmad\SER\\model\\model.h5'
# file path
filePath='\\Users\\ahmad\SER\\TestedSounds\\'

# angry sound
pred = livePredictions(path=modelPath,file=filePath+'angry.wav')
pred.load_model()
pred.makepredictions()

#calm sound
pred = livePredictions(path=modelPath,file=filePath+'calm.wav')
pred.load_model()
pred.makepredictions()

#disgust dound
pred = livePredictions(path=modelPath,file=filePath+'disgust.wav')
pred.load_model()
pred.makepredictions()

#fearful sound
pred = livePredictions(path=modelPath,file=filePath+'fearsful.wav')
pred.load_model()
pred.makepredictions()

#happy sound
pred = livePredictions(path=modelPath,file=filePath+'happy.wav')
pred.load_model()
pred.makepredictions()

#natural sound
pred = livePredictions(path=modelPath,file=filePath+'natural.wav')
pred.load_model()
pred.makepredictions()

# sad sound
pred = livePredictions(path=modelPath,file=filePath+'sad.wav')
pred.load_model()
pred.makepredictions()

#surprised sound
pred = livePredictions(path=modelPath,file=filePath+'surprised.wav')
pred.load_model()
pred.makepredictions()
