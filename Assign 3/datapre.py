import librosa
import os
import numpy as np
import python_speech_features as psf

def maxsample(path):
    maxsamp = 0
    for i in os.listdir(path):
        p = os.path.join(path,i)
        for j in os.listdir(p):
            pa = os.path.join(p,j)
            wav,sr = librosa.load(pa)
            maxsamp = len(wav) if (len(wav) > maxsamp) else maxsamp
    return (maxsamp,sr)

def mfccout(path,m):
    main_arr = []
    for i in os.listdir(path):
        p = os.path.join(path,i)
        new = []
        for j in os.listdir(p):
            pa = os.path.join(p,j)
            wav,sr = librosa.load(pa)
            l = wav.shape[0]
            app = m-l if l<m else l
            wav = np.pad(wav,(0,app),'constant')
            mfc = psf.mfcc(wav,sr,nfft=1024)
            n = [mfc,int(i)]
            new = np.asarray(n)
            main_arr.append(new)
    main_ar = np.asarray(main_arr)
    return main_ar

if __name__ == "__main__":
    path = "C:\\Users\\rhin1\\Downloads\\DSP\\wavs"
    (high,s) = maxsample(path)
    fg = ((high//s)+1)*s
    arr = mfccout(path,fg)
    np.save('datan.npy',arr)