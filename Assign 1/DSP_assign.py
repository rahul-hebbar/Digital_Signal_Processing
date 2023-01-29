from playsound import playsound
import time
import matplotlib.pyplot as plt
import librosa 
import numpy

def sgn(x):
  y = numpy.zeros_like(x)
  y[numpy.where(x >= 0)] = 1.0
  y[numpy.where(x < 0)] = -1.0
  return y

if __name__ == "__main__":
  song_li = ["beep_fs_10000.wav","s5_synthetic.wav","test_16k.wav","beep_fs_16000.wav","should.wav"]
  for i in song_li:
    playsound(i)
    time.sleep(1)
  li = ["s5.wav","should.wav"]
  for i in range(2):
    wav,fs = librosa.load(li[i])
    n_wav = librosa.util.normalize(wav)
    win = numpy.hamming(max(1,len(n_wav)//64))
    win = win/len(win)
    mag = numpy.convolve(numpy.abs(n_wav),win,mode = "same")*200
    eng = numpy.convolve(numpy.square(n_wav),numpy.square(win),mode = "same")*2000
    x = numpy.roll(n_wav, 1)
    x[0] = 0.0
    abs_diff = numpy.abs(sgn(n_wav) - sgn(x))
    zc = numpy.convolve(abs_diff, win, mode="same")*0.5
    plt.figure(i+1)
    plt.suptitle("waveform " + str(i+1))
    plt.subplot(411)
    plt.title(li[i])
    plt.plot(wav)
    plt.suptitle("Feature extraction")
    plt.subplot(412)
    plt.title("STM")
    plt.plot(mag)
    plt.subplot(413)
    plt.title("STE")
    plt.plot(eng)
    plt.subplot(414)
    plt.title("ZCR")
    plt.plot(zc)

  plt.show()