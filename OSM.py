import numpy as np
import sounddevice as sd
from scipy.signal import firwin, fftconvolve
import matplotlib.pyplot as plt

# Task 1: Record Audio Signal with Noise
Fs = 8000  # Sampling frequency
recordTime = 5  # Recording time in seconds

print('Start recording...')
x = sd.rec(int(Fs * recordTime), samplerate=Fs, channels=1, blocking=True)
print('Recording finished.')

# Task 2: Play the recorded signal
sd.play(x, Fs)
sd.wait()

# Task 3: Design FIR Low Pass Filter
Fpass = 1500  # Passband frequency in Hz
Fstop = 2500  # Stopband frequency in Hz
N = 101  # Filter order (adjust as needed)

h = firwin(N, cutoff=Fpass, fs=Fs, pass_zero=True)

# Task 4: Filter the audio signal using FFT-based Overlap Add
overlap = N - 1  # Overlap size
hopSize = N - overlap
numBlocks = int(np.ceil(len(x) / hopSize))
# Ensure y has enough samples for convolution
y = np.zeros(len(x) + N - 1)

for i in range(numBlocks):
    startIndex = i * hopSize
    endIndex = min(startIndex + N, len(x))
    block = x[startIndex:endIndex].flatten()
    h_1d = h.ravel()  # Flatten the filter kernel to 1D array
    yBlock = fftconvolve(block, h_1d, mode='same')
    y[startIndex:startIndex + len(yBlock)] += yBlock

# Trim y to match the original length
y = y[:len(x)]

# Task 5: Play the filtered signal
sd.play(y, Fs)
sd.wait()

# Task 6: Plot Magnitude Spectrum of Input Signal
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(np.abs(np.fft.fft(x)))
plt.title('Magnitude Spectrum of Input Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

# Task 7: Plot Magnitude Spectrum of Filtered Output Signal
plt.subplot(2, 1, 2)
plt.plot(np.abs(np.fft.fft(y)))
plt.title('Magnitude Spectrum of Filtered Output Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')

plt.tight_layout()
plt.show()
