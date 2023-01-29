import librosa
import os
import numpy as np
import matplotlib.pyplot as plt
from librosa.display import specshow
import cv2.cv2 as cv2

def printpic(path):
    for i in os.listdir(path):
        p = os.path.join(path,i)
        for j in os.listdir(p):
            pa = os.path.join(p,j)
            wav,sr = librosa.load(pa)
            mfc = librosa.feature.melspectrogram(wav,sr)
            specshow(mfc)
            plt.axis('off')
            plt.savefig((j.replace(".wav","")+".png"),bbox_inches='tight'
                        , transparent=True, pad_inches=0.0)

def createnpy(path):
    data = []
    label = []
    fg = 0
    for i in os.listdir(path):
        p = os.path.join(path,i)
        for j in os.listdir(p):
            pa = os.path.join(p,j)
            img = cv2.imread(pa,cv2.IMREAD_COLOR)
            # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            im = cv2.resize(img,(100,100))
            pic = np.asarray(im)
            data.append(pic)
            label.append(fg)
        fg += 1
    dat = np.asarray(data)
    lab = np.asarray(label)
    return dat,lab
    
if __name__ == "__main__":
    """
    First we read all wav files
    we then plot their respective mel spectographs using librosa
    The spectographs images are then stored in png format without axes
    """

    printpic("C:\\Users\\rhin1\\Downloads\\DSP\\wavs")

    """
    once we have the mel spectograph images, we import the images using opencv
    we are reading the images in BGR channel
    all images are resized to a standard shape (100,100)
    we create a npy of resized images (save file of numpy array)
    for both the data and the corresponding label 
    the npy files help importing all the image data easily and efficiently
    """
    d,l = createnpy("C:\\Users\\rhin1\\Downloads\\DSP\\melpic")
    print(d.shape,l.shape) #get the shape of image data array and label array
    np.save('d.npy',d) # create npy file of data
    np.save('l.npy',l) #create npy file of label