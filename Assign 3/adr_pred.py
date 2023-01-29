import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # turning of logging WARNINGS and INFO

import tensorflow as tf
import numpy as np
import librosa
from librosa.display import specshow
import matplotlib.pyplot as plt
import cv2.cv2 as cv2

def plots(path):
    for i in os.listdir(path):
        wav,sr = librosa.load(os.path.join(path,i))
        mfc = librosa.feature.melspectrogram(wav,sr)
        specshow(mfc)
        plt.axis('off')
        plt.savefig((i.replace(".wav","")+".png"),bbox_inches='tight', transparent=True, pad_inches=0.0)

def arr(path):
    pic = []
    for i in os.listdir(path):
        img = cv2.imread(os.path.join(path,i),cv2.IMREAD_COLOR)
        j = cv2.resize(img,(100,100))
        j = np.asarray(j)
        pic.append(j)
    pic = np.asarray(pic)
    return pic

if __name__ == "__main__":
    """
    we have 10 audio samples each for a digit
    we will test our model using these test audio
    first we will make mel spectograph plots of 10 audio samples
    then we will save the spectograph as images
    """
    plots("C:\\Users\\rhin1\\Downloads\\DSP\\dig1")
    """
    we are loading the 10 image files saved
    we are resizing the images to 100x100
    we are making a numpy array with BGR channel values
    """
    pred = arr("C:\\Users\\rhin1\\Downloads\\DSP\\meltest1")

    print(pred.shape) # print numpy array shape
    model = tf.keras.models.load_model("adr_model.h5") #loading saved model
    predic = model.predict(pred) # making a prediction on test array
    print(predic) # printing all model probabilities
    val = [np.argmax(i) for i in predic] # segregating index of max probability
    print(val) # printing the index value indirectly the digit recognised