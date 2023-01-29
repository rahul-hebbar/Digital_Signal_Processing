import librosa
import numpy as np
import DTW
import sounddevice as sd
import matplotlib.pyplot as plt
import noisereduce as nr
import os
import fea
import fastdtw
from scipy.spatial.distance import euclidean
import python_speech_features as pysf

def rec():
    main_fs = 22050
    main_sec = 0.907
    print('tell a dig btw 0-9')
    myrec = sd.rec(int(main_fs*main_sec),samplerate=main_fs, channels=1)
    sd.wait()
    print('done')
    librosa.output.write_wav("mine.wav", myrec, sr = main_fs)

def noise_red(wav,sr):
    new = nr.reduce_noise(wav,wav,verbose=True)
    librosa.output.write_wav("noiseless.wav", new, sr =sr)

def endpt(wav):
    wav = wav*(1/max(wav))
    win =  fea.Win(wav,"rect")
    e = fea.Norm_Short_Time_Energ(wav,win)
    st,ed,fg = 0,0,0
    for i in range(len(e)):
        if fg == 0:
            if e[i] > 0.0005:
                st,fg = i,1
        else:
            if e[i] < 0.0005:
                ed = i
                break
    wav = e[st:ed]
    return wav

if __name__ == "__main__":
    path = "C:\\Users\\rhin1\\Downloads\\DSP\\dig"
    wa,sr = librosa.load("noiseless.wav")

    nwa = endpt(wa)
    l_wa = len(nwa)

    # mfccwa = pysf.mfcc(wa,sr,nfft=1024)

    # plt.figure(1)
    # plt.plot(nwa)
    # plt.plot(wa)
    # j = 2

    wrp = []
    for i in os.listdir(path):
        sample,sr1 = librosa.load(os.path.join(path,i))
        nsample = endpt(sample)

        l_sa = len(nsample)
        ratio = l_sa/l_wa if (l_sa>l_wa) else l_wa/l_sa
        ratio = round(ratio,2)
        win1 = round(l_wa/(ratio*100))
        win2 = round(l_sa/(ratio*100))
        dis = 0
        print("Started")
        for j in range(int(ratio*100)):
            x = nwa[j:j+win1]
            y = nsample[j:j+win2]
            tb = DTW.DwtTable(x,y)
            d,ind = DTW.WarpPath(tb)
            dis += d
        print("Stopped")

        # mfccsa = pysf.mfcc(sample,sr1,nfft = 1024)
        # print("st")
        # dis,pat = fastdtw.fastdtw(mfccwa,mfccsa, dist=euclidean)
        # print("stop")

        # plt.figure(j)
        # plt.plot(sample)
        # plt.plot(nsample)
        # j += 1

        wrp.append(dis)
    m = min(wrp)
    print(wrp.index(m))

    # plt.show()