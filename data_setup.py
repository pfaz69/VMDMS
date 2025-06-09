# Code and algorithm design by Paolo Fazzini (paolo.fazzini@cnr.it)

import pandas as pd
import numpy as np
from scipy import signal


def load_data():
    
    T=1000
    fs=500
    t = np.arange(T) / fs
    
    freq_config = {
        # [frequency, bandwidth, amplitude] for each mode in each channel
        0: [(87.3, 0.5, 0.6), (30.1, 0.5, 1.5), (22.5, 0.5, 0.2)],
        1: [(26.8, 10.0, 0.7), (79.7, 0.5, 1.4), (100.0, 8.7, 0.5)],
        2: [(47.8, 3.9, 0.1), (30.4, 0.5, 1.5), (19.9, 5.4, 0.1)],
    }

    # Generate signals
    signals = np.zeros((len(freq_config), T))
    ground_truth_modes = np.zeros((len(freq_config), T, 3))
    
    for ch in range(len(freq_config)):
        for mode_idx, (freq, bw, amp) in enumerate(freq_config[ch]):
            if amp > 0:  # Skip modes with zero amplitude
                
                # Create a band-limited signal using filtered noise
                noise = np.random.randn(T)
                sos = signal.butter(4, [freq-bw, freq+bw], btype='bandpass', fs=fs, output='sos')
                mode = amp * signal.sosfilt(sos, noise)
                # Normalize to maintain consistent amplitude
                mode = mode / np.std(mode) * amp
                
                ground_truth_modes[ch, :, mode_idx] = mode
                signals[ch] += mode
    
    # Add some noise
    noise_level = 0.05
    signals += noise_level * np.random.randn(*signals.shape)
    
    return signals, ground_truth_modes, t, freq_config



def create_sequences(data, time_steps, forecasting_steps, increment):
    X, y = [], []

    for i in range(0, len(data) - time_steps - forecasting_steps + 1, increment):
        # Extract input sequence (X)
        input_sequence = data[i:i + time_steps, :]

        # Extract target sequence (y)
        target_sequence = data[i + time_steps:i + time_steps + forecasting_steps, :]

        X.append(input_sequence)
        y.append(target_sequence)

    return np.array(X), np.array(y)



def split_data(data, split):
    train = data[:-split]
    test = data[-split:]
    return train, test


def split_data_multitest(data, num_tests, step_test):
    train = data[:-step_test*num_tests]
    tests = []
    for i in range(num_tests):
        start = -step_test*(num_tests - i)
        stop = -step_test*(num_tests - i - 1)
        stop = None if stop == 0 else stop
        tests.append(data[start:stop])
    return train, tests



