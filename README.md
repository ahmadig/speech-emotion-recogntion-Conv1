# Speech-Emotion-Recogntion-Conv1
In this project we talk about (SER) machine learning we use CNN (Conv1d), (SER) is  analysis of any kind of signal data over a fixed-length period (such as audio signals).


![Screenshot 2021-02-06 205012](https://user-images.githubusercontent.com/65724677/107127197-25f73e00-68bd-11eb-8e07-e46ebb8aabbf.png)

# Prepare before Running
Now i want to guide you how to download this project , you have to download the files , i work with python3.8 also anaconda prompt to download some Libraries. 
To download the libraries , you must have to write in anaconda prompt,exe:(pip install numpy) if you don't have a pip go to cmd and download it from there.

# Download DataSet
Click on the link to download Dataset https://zenodo.org/record/1188976#.YB0_rmgzZPY , scroll down and download the file zip speech
 maybe you will got accuracy 60% , to up accuracy you must have to add more folders sounds for the speech.
# How to run the code 
After you upload the code to python and save the dataset and do all changes path by where you saved the files , firstly you run Datasetload.py it will create a folder in name joblib that save the Datasetload after you run this section of code .
Secondlly you run code model.py in this file you see thet we use Conv1D, and we save the model in folder by name model.h5 after we run this section of code in the console we see the accuracy that we get.
Finally we run predictTest.py , in this file we do tests for 8 sounds and we get the output in the console to do this you have to put the file TestedSounds with the folder files see the path . 

# Accuracy 

![Screenshot 2021-02-06 205122](https://user-images.githubusercontent.com/65724677/107127284-ca798000-68bd-11eb-8ecd-264ba78a0d43.png)
