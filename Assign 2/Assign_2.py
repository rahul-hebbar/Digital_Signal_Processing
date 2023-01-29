import librosa
import librosa.display as ld
import python_speech_features as psf
import matplotlib.pyplot as plt
import numpy as np
import os
path = "C:\\Users\\rhin1\\Downloads\\DSP\\Assign 2\\Wavs"
j = 1
for i in os.listdir(path):
    name = os.path.join(path,i)
    y,sr = librosa.load(name) #Load Wav
    lfe = psf.logfbank(y,sr,nfft = 1024)  # Log filterbank energies
    mfcc = psf.mfcc(y,sr,nfft = 1024, winfunc=np.hamming) # MFCC
    time = np.linspace(0, len(y) / sr, num=len(y)) #To plot waveform wrt Time
    plt.figure(j)
    plt.suptitle(i)
    j = j+1
    plt.subplot(3,1,1)
    plt.title('waveform')
    plt.plot(time,y)
    plt.subplot(3,1,2)
    plt.title('log filterbank energies')
    ld.specshow(lfe, x_axis = 'time')
    plt.colorbar()
    plt.tight_layout()
    plt.subplot(3,1,3)
    plt.title('MFCC')
    # mfcc = np.swapaxes(mfcc,0,1)
    ld.specshow(mfcc, x_axis = 'time')
    plt.colorbar()
    plt.tight_layout()
plt.show()

