import numpy as np

l = [7,6,6,7,3,1]
z = np.fft.fft(l, n = 6)
in_z = np.fft.ifft(z,n = 6)
z = np.around(z,decimals = 3)
print(z)