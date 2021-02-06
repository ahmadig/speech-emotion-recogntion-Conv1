from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import joblib
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D

X = joblib.load('\\Users\\ahmad\\SER\\joblib\\X.joblib')
y = joblib.load('\\Users\\ahmad\\SER\\joblib\\y.joblib')

# new data 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

# make the data into 3d to work with cnn
x_traincnn = np.expand_dims(X_train, axis=2)
x_testcnn = np.expand_dims(X_test, axis=2)

#our cnn model
model = Sequential()
# We use con1d because its very effective when you expect to generate interesting attributes shorter (Constant length) 
model.add(Conv1D(64,5,padding='same',input_shape=(40,1)))
# relu used for to change the value of negative numbers to 0
model.add(Activation('relu'))
#we use dropout to remove randomly selected neurons that ignored during training
model.add(Dropout(0.1))
# we use maxpoling to take the maximum of 4 in data
model.add(MaxPooling1D(pool_size=(4)))
model.add(Conv1D(128, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
model.add(MaxPooling1D(pool_size=(4)))
model.add(Conv1D(256, 5,padding='same',))
model.add(Activation('relu'))
model.add(Dropout(0.1))
# we use flatten to  take the data to one array to use them to the next layer(Dense).
model.add(Flatten())
# we use dense to connect each neuron in one layer for each neuron in another layer
model.add(Dense(8))
# we use softmax to convert the output of the last layer in our neural network into an essentially probability distribution
model.add(Activation('softmax'))
#we use this optmizer to utilizes the magnitude of recent gradients to normalize the gradients.
opt = keras.optimizers.RMSprop(lr=0.00005)
model.summary()
#Computes the cross-entropy loss between true labels and predicted labels. 
model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

# learning
cnnhistory=model.fit(x_traincnn, y_train, batch_size=16, epochs=200, validation_data=(x_testcnn, y_test))


# loss picture adn save it
plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Pic\\modelLose.png')
plt.show()

# accuracy picture and save it
plt.plot(cnnhistory.history['accuracy'])
plt.plot(cnnhistory.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Pic\\modelAcc.png')
plt.show()

# model prediction
predictions = model.predict_classes(x_testcnn)
new_Ytest = y_test.astype(int)

# the report of our model
report = classification_report(new_Ytest, predictions)
print(report)

# confusion matrix
matrix = confusion_matrix(new_Ytest, predictions)
print (matrix)


# save the model
model.save('model\\model.h5')
print("MODEL SAVED")


# load the model
new_model=keras.models.load_model('\\Users\\ahmad\\SER\\model\\model.h5')
new_model.summary()


# accuracy
loss, acc = new_model.evaluate(x_testcnn, y_test)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))



