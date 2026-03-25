import glob
import numpy as np
import random
from natsort import natsorted

def make_data(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    data = []
    for si in lines:
        data.append([float(i) for i in si.strip().split(' ')])
    return data

def make_inputs_and_labels(input_path, label_path):
    inputs = []
    labels = []
    for i in range(len(input_path)):
        input_data = make_data(input_path[i])
        label_data = make_data(label_path[i])
        inputs.extend(input_data)
        labels.extend(label_data)
    return inputs, labels

class DataPreprocessor:
    def __init__(self, use_median_filter=True, train=True, use_power_spectrum=True):
        self.use_median_filter = use_median_filter
        self.use_power_spectrum = use_power_spectrum
        self.input_real_path = natsorted(glob.glob('learning_data/2025_7_1_5/input/real/real_input*.txt'))
        self.input_imag_path = natsorted(glob.glob('learning_data/2025_7_1_5/input/imag/imag_input*.txt'))
        self.label_real_path = natsorted(glob.glob('learning_data/2025_7_1_5/label/real/real_label*.txt'))
        self.label_imag_path = natsorted(glob.glob('learning_data/2025_7_1_5/label/imag/imag_label*.txt'))

        self.real_inputs, self.real_labels = make_inputs_and_labels(self.input_real_path, self.label_real_path)
        self.imag_inputs, self.imag_labels = make_inputs_and_labels(self.input_imag_path, self.label_imag_path)
        
        self.real_inputs = np.array(self.real_inputs)
        self.real_labels = np.array(self.real_labels)
        self.imag_inputs = np.array(self.imag_inputs)
        self.imag_labels = np.array(self.imag_labels)

        if self.use_power_spectrum:
            abs_inputs = np.sqrt(self.real_inputs**2 + self.imag_inputs**2)
            abs_labels = np.sqrt(self.real_labels**2 + self.imag_labels**2)
            self.input_data = 10 * np.log10(abs_inputs**2)
            self.label_data = 10 * np.log10(abs_labels**2)
        else:
            self.input_data = np.fft.fft(self.real_inputs + 1j * self.imag_inputs)
            self.label_data = np.fft.fft(self.real_labels + 1j * self.imag_labels)
            # 実部・虚部を分けて使いたい場合は適宜修正

        self.max_length = self.test_max_length(self.input_data)
        self.inputs, self.labels = self.normalize_array(self.input_data, self.label_data)
        
        if train:
            x = list(range(len(self.inputs)))
            random.shuffle(x)
            self.inputs = self.inputs[x]
            self.labels = self.labels[x]

    def test_max_length(self, data):
        return max(len(d) for d in data)
    
    def median_filter(self, inputs):
        for idx, signal in enumerate(inputs):
            thres = 100 * np.median(np.abs(signal))
            signal[np.abs(signal) > thres] = 0
            inputs[idx] = signal
        return inputs

    def normalize_array(self, inputs, labels):
        norm_input = []
        norm_label = []
        if self.use_median_filter:
            inputs = self.median_filter(inputs)
        for idx in range(len(inputs)):
            norm_val = np.sqrt(np.max(np.abs(inputs[idx])**2))
            norm_input.append(inputs[idx] / norm_val)
            norm_label.append(labels[idx] / norm_val)
        return np.array(norm_input), np.array(norm_label)