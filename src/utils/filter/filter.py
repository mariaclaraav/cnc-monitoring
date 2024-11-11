import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt

def plot_cwt_spectrogram(time, signal, scale_range, wavelet, sampling_rate, ax=None, title=None, cmap='seismic'):
    """
    Plots a CWT spectrogram given a time array, signal array, scale range, wavelet type, and sampling rate.

    Parameters:
    time (np.ndarray): Array of time values.
    signal (np.ndarray): Array of signal values.
    scale_range (np.arraye): Range of scales for CWT 
    wavelet (str): Type of wavelet to use.
    sampling_rate (int): Sampling rate of the signal (Hz).
    title (str, optional): Title of the plot. Default is None.

    Returns:
    None: Displays the CWT spectrogram.
    """   

    # Compute the CWT
    cwtmatr, freqs = pywt.cwt(signal, scale_range, wavelet, sampling_period=1/sampling_rate)
    cwtmatr = np.abs(cwtmatr[:-1, :-1])
    
    # Create a new figure and plot
    fig, ax = plt.subplots(figsize=(10, 3))
    cax = ax.pcolormesh(time, freqs, cwtmatr, cmap=cmap, shading='auto', vmin=0, vmax=700)

    fig.colorbar(cax, ax=ax, label='Magnitude')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    if title is not None:
        ax.set_title(title)

    # Show the plot
    plt.show()
        
def plot_wavelet_decomposition(y, wavelet='coif8', level=3):
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(y, wavelet, level=level)
    
    # Extract approximation and details coefficients
    cA = coeffs[0]
    details = coeffs[1:]
    
    # Resample the decomposed coefficients back to the original signal length
    cA_upsampled = pywt.upcoef('a', cA, wavelet, level=level, take=len(y))
    details_upsampled = [pywt.upcoef('d', detail, wavelet, level=i+1, take=len(y)) for i, detail in enumerate(details)]

    # Plot the original signal and the decomposition
    plt.figure(figsize=(10, 5 + 3 * level))

    plt.subplot(level + 2, 1, 1)
    plt.plot(y, 'r')
    plt.title('Original Signal')

    plt.subplot(level + 2, 1, 2)
    plt.plot(cA_upsampled, 'g')
    plt.title(f'Approximation Coefficients (cA{level})')

    for i, detail in enumerate(details_upsampled):
        plt.subplot(level + 2, 1, i + 3)
        plt.plot(detail, 'b')
        plt.title(f'Detail Coefficients (cD{level - i})')

    plt.tight_layout()
    plt.show()
    return cA_upsampled, details_upsampled
        
        
# Design the Butterworth high-pass filter
def butter_highpass(cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff, fs, order=4):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Design the Butterworth bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def perform_fft(time, signal, fs):
    N = len(time)
    
    # Calculate FFT
    yf = fft(signal)
    xf = fftfreq(N, 1/fs)[:N//2]
    
    # Remove negative part and DC component
    yf = yf[1:N//2]
    
    x_fft = xf[1:]
    y_fft = 2.0/N * np.abs(yf)
    
    return x_fft, y_fft