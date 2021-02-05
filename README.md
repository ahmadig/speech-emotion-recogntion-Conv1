# Speech-Emotion-Recogntion-Conv1
In this project we talk about (SER) machine learning we use CNN (Conv1d), (SER) is  analysis of any kind of signal data over a fixed-length period (such as audio signals).

# Prepare before Running
Now i want to guide you how to download this project , you have to download the fills , i work with python3.8 also anaconda prompt to download some Libraries. 
To download the libraries , you must have to write in anaconda prompt,exe:(pip install numpy) if you don't have a pip go to cmd and download it from there.

# Download DataSet
To download the dataset that i used, you should to click in this link and download it from google drive() .
another file that you can download https://zenodo.org/record/1188976#.YB0_rmgzZPY , scroll down and download the file of the speech
 maybe you will got accuracy 60% , to up accuracy you must have to add more folders for the speech.
# how to run the code 
Firstly after you upload the code to python and save the dataset and do all changes pathe by where you saved the files , first you run Datasetload creates a folder in name joblib that save the Datasetload after you run this section of code 
Secondlly you run file model in this file you see thet we use Conv1D, and we save the model in folder by name model.h5 after we run this section in the console we see the accuracy that we got from the test.
Finally we run predictTest , in this file do tests for 8 sounds and we gקt the output in the console of the sounds to do this you ust have to put the file TestedSounds with the files . 
