import numpy as np
from scipy.fft import fft, ifft, fftfreq
from scipy.signal import find_peaks

class EmpiricalWaveletTransform:
    """Class implementing Empirical Wavelet Transform (EWT) for 1D signals."""
    
    def __init__(self, n_modes=5, max_freq=300, sampling_rate=1000):
        self.n_modes = n_modes
        self.max_freq = max_freq
        self.sampling_rate = sampling_rate
        self.boundaries = None

    def _smooth_spectrum(self, spectrum, window_len=11, sigma=5):
        """Applies Gaussian smoothing to the frequency spectrum."""
        s = np.r_[spectrum[window_len-1:0:-1], spectrum, spectrum[-2:-window_len-1:-1]]
        w = np.exp(-0.5 * (np.arange(window_len) - (window_len-1)/2)**2 / sigma**2)
        w /= w.sum()
        return np.convolve(w, s, mode='valid')[(window_len//2):-(window_len//2)]

    def detect_boundaries(self, signal):
        """Detects frequency boundaries for decomposition."""
        length = len(signal)
        spectrum = np.abs(fft(signal))[:length//2]
        freqs = fftfreq(length, 1/self.sampling_rate)[:length//2]
        
        # Focus on the relevant frequency range (e.g., 0-300Hz for PCG)
        valid_idx = (freqs >= 0) & (freqs <= self.max_freq)
        spectrum = spectrum[valid_idx]
        freqs = freqs[valid_idx]
        
        # Smooth and find peaks
        smooth_spec = self._smooth_spectrum(spectrum)
        peaks, _ = find_peaks(smooth_spec)
        
        if len(peaks) >= self.n_modes - 1:
            peak_vals = smooth_spec[peaks]
            # Use top peaks as boundaries
            # In the original code, boundaries were selected from top peaks
            top_peaks = np.sort(peaks[np.argsort(peak_vals)[-(self.n_modes-1):]])
            self.boundaries = np.concatenate(([0], top_peaks, [len(spectrum)-1]))
        else:
            # Fallback to linear division if not enough peaks
            self.boundaries = np.linspace(0, len(spectrum)-1, self.n_modes+1, dtype=int)
            
        return self.boundaries

    def decompose(self, signal):
        """Decomposes the signal into n_modes using EWT."""
        length = len(signal)
        fft_signal = fft(signal)
        freqs = fftfreq(length, 1/self.sampling_rate)
        
        if self.boundaries is None:
            self.detect_boundaries(signal)
            
        modes = []
        for i in range(self.n_modes):
            mask = np.zeros(length, dtype=complex)
            start_freq = freqs[self.boundaries[i]]
            end_freq = freqs[self.boundaries[i+1]]
            
            # Mask both positive and negative frequencies
            mask_idx = (np.abs(freqs) >= abs(start_freq)) & (np.abs(freqs) < abs(end_freq))
            mask[mask_idx] = 1
            
            mode = np.real(ifft(fft_signal * mask))
            modes.append(mode)
            
        return modes
