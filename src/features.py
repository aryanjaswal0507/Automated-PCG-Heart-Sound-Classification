import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from scipy.signal import hilbert
import librosa
import pywt
from .decomposition import EmpiricalWaveletTransform

def extract_statistical_features(signal):
    """Extracts mean, std, skewness, kurtosis, rms, and envelope energy."""
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    skew_val = skew(signal)
    kurt_val = kurtosis(signal)
    rms_val = np.sqrt(np.mean(signal**2))
    envelope = np.abs(hilbert(signal))
    envelope_energy = np.sum(envelope**2)
    
    return [mean_val, std_val, skew_val, kurt_val, rms_val, envelope_energy]

def extract_spectral_features(signal, fs=1000):
    """Extracts spectral centroid, bandwidth, dominant frequency, and band power."""
    length = len(signal)
    spectrum = np.abs(fft(signal))[:length//2]
    freqs = fftfreq(length, 1/fs)[:length//2]
    
    spectral_centroid = np.sum(freqs * spectrum) / np.sum(spectrum)
    spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid)**2) * spectrum) / np.sum(spectrum))
    dominant_freq = freqs[np.argmax(spectrum)]
    
    # Power band calculation (e.g., 20-150Hz)
    band_mask = (freqs >= 20) & (freqs <= 150)
    # Using trapezoidal rule for numerical integration
    band_power = np.trapezoid(spectrum[band_mask]**2, freqs[band_mask]) if np.any(band_mask) else 0
    
    return [spectral_centroid, spectral_bandwidth, dominant_freq, band_power]

def extract_mfcc_features(signal, fs=1000, n_mfcc=5):
    """Extracts MFCC means from the signal."""
    # Resample signal to 22050 for librosa
    signal_resampled = librosa.resample(signal.astype(np.float32), orig_sr=fs, target_sr=22050)
    mfccs = librosa.feature.mfcc(y=signal_resampled, sr=22050, n_mfcc=n_mfcc)
    
    return np.mean(mfccs, axis=1).tolist()

def extract_wavelet_features(signal, wavelet='db4', level=3):
    """Extracts wavelet energy features from three detail levels (levels 1-3)."""
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    # coeffs[0] is approximation, coeffs[1:] are details (highest first)
    # Levels 1, 2, 3 detail coefficients are in coeffs[1], coeffs[2], coeffs[3]
    energies = []
    for coeff in coeffs[1:4]:
        energies.append(np.sum(np.square(coeff)))
        
    return energies

def extract_all_features_for_signal(signal, n_modes=5, fs=1000):
    """Processes a single signal through decomposition and extracts all feature sets."""
    ewt = EmpiricalWaveletTransform(n_modes=n_modes, sampling_rate=fs)
    modes = ewt.decompose(signal)
    
    features = []
    
    # 1. Statistical features for each mode
    for mode in modes:
        features.extend(extract_statistical_features(mode))
        
    # 2. Spectral features for the full signal
    features.extend(extract_spectral_features(signal, fs=fs))
    
    # 3. MFCC features for the full signal
    features.extend(extract_mfcc_features(signal, fs=fs))
    
    # 4. Wavelet features for the full signal
    features.extend(extract_wavelet_features(signal))
    
    return features

def get_feature_column_names(n_modes=5):
    """Generates a list of column names for the extracted features."""
    stat_names = ['mean', 'std', 'skew', 'kurtosis', 'rms', 'envelope_energy']
    spec_names = ['spectral_centroid', 'spectral_bandwidth', 'dominant_freq', 'power_band_20_150Hz']
    mfcc_names = [f'mfcc_{i+1}_mean' for i in range(5)]
    wavelet_names = [f'wavelet_L{i}_energy' for i in range(1, 4)]
    
    col_names = []
    for m in range(n_modes):
        for name in stat_names:
            col_names.append(f"{name}_mode{m+1}")
            
    col_names += spec_names + mfcc_names + wavelet_names
    return col_names
