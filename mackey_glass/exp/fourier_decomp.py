import numpy as np
from scipy import fftpack
import matplotlib.pyplot as plt

def generate_composite_signal(Fs=100, duration=1.0):
    """
    Create a composite signal from 3 sine waves of different frequencies and amplitudes.
    Returns time vector t, the composite signal y, and its frequency domain y_fft.
    """
    t = np.arange(0, duration, 1 / Fs)

    # Define sine components
    A1, f1 = 0.5, 2  # amplitude, frequency
    A2, f2 = 1.0, 5
    A3, f3 = 2.0, 10

    # Construct signal
    y1 = A1 * np.sin(2 * np.pi * f1 * t)
    y2 = A2 * np.sin(2 * np.pi * f2 * t)
    y3 = A3 * np.sin(2 * np.pi * f3 * t)
    y = y1 + y2 + y3

    return t, y, (y1, y2, y3)

def apply_fft(y, Fs):
    """
    Perform FFT and return frequency components and magnitudes
    """
    y_fft = fftpack.fft(y)
    n = len(y)
    freqs = Fs / 2 * np.linspace(0, 1, n // 2)
    magnitudes = 2 / n * np.abs(y_fft[:n // 2])
    return freqs, magnitudes, y_fft

def reconstruct_signal(y_fft):
    """
    Reconstruct signal from FFT using inverse FFT
    """
    return np.real(fftpack.ifft(y_fft))

def plot_all(t, y, freqs, magnitudes, y_reconstructed, components):
    """
    Plot time-domain signal, frequency-domain, and reconstructed signal
    """
    y1, y2, y3 = components

    fig, ax = plt.subplots(2, 2, figsize=(15, 7.5))

    # Time domain (original signal)
    ax[0, 0].plot(t, y)
    ax[0, 0].set_title("Original Time-Domain Signal")

    # Frequency domain
    ax[0, 1].stem(freqs, magnitudes)
    ax[0, 1].set_title("Frequency Domain (FFT)")

    # Individual components
    ax[1, 0].plot(t, y1, label="2 Hz")
    ax[1, 0].plot(t, y2, label="5 Hz")
    ax[1, 0].plot(t, y3, label="10 Hz")
    ax[1, 0].set_title("Individual Sine Components")
    ax[1, 0].legend()

    # Reconstructed signal
    ax[1, 1].plot(t, y_reconstructed)
    ax[1, 1].set_title("Reconstructed Signal from IFFT")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    Fs = 100  # Sampling rate
    t, y, components = generate_composite_signal(Fs)
    freqs, magnitudes, y_fft = apply_fft(y, Fs)
    y_reconstructed = reconstruct_signal(y_fft)
    plot_all(t, y, freqs, magnitudes, y_reconstructed, components)
