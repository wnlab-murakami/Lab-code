import tensorflow as tf
import numpy as np
import scipy.io
import glob
import datetime
import time
import os
from natsort import natsorted
from statistics import mean, stdev
import rnn_transformer_copy

radar = rnn_transformer_copy.Radar(mod='False', make_data=True)
Number_of_samples = 512
Number_of_chirps = 16
dt_now = datetime.datetime.today()

save_path_real = 'model/Attention_biLSTM/Attention_biLSTM_20250629/res'
mkdir = 'suppression_data/%d%d%d' % (dt_now.year, dt_now.month, dt_now.day)
if not os.path.exists(mkdir):
    os.makedirs(mkdir)

def get_max_value(inputs):
    max_value = []
    for signal in inputs:
        max_val = np.sqrt(max([i * i for i in signal]))
        max_value.append(max_val)
    return max_value

def make_data(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    data = []
    for si in lines:
        data.append([float(i) for i in si.strip().split(' ')])
    return data

def pad_data(inputs, Number_of_samples):
    pad_inputs = []
    for signal_input in inputs:
        if len(signal_input) > Number_of_samples:
            pad_inputs.append(signal_input[:Number_of_samples])
        else:
            pad_inputs.append(signal_input + [0] * (Number_of_samples - len(signal_input)))
    return pad_inputs

def normalize(inputs, inputs_max):
    norm_inputs = []
    for channel in range(1):
        for ramp in range(Number_of_chirps):
            norm_inputs.append([float(i) / inputs_max[channel][ramp] for i in inputs[channel][ramp]])
    return norm_inputs

def denormalize(test_inputs, inputs_max):
    norm_test = []
    for channel in range(1):
        for ramp in range(Number_of_chirps):
            norm_test.append([float(i) * inputs_max[channel][ramp] for i in test_inputs[channel][ramp]])
    norm_test = np.transpose(np.reshape(np.array(norm_test), (1, Number_of_chirps, Number_of_samples)), (2, 1, 0))
    return norm_test

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  
with tf.compat.v1.Session(config=config) as sess:
    radar.saver.restore(sess, save_path_real)
    times = []
    for radar_waveform in range(1, 2):
        for radar_data in range(1, 101):
            imag_data_path = natsorted(glob.glob('test_data/2025_6_30_0/input/imag/imag_input_%d_%d.txt' % (radar_waveform, radar_data)))
            real_data_path = natsorted(glob.glob('test_data/2025_6_30_0/input/real/real_input_%d_%d.txt' % (radar_waveform, radar_data))) 
            signal_index = radar_waveform
            random_index = radar_data

            real_data = [make_data(real_data_path[0])]
            imag_data = [make_data(imag_data_path[0])]
            abs_data = np.sqrt(np.array(real_data) ** 2 + np.array(imag_data) ** 2)
            test_inputs = 10 * np.log10(abs_data ** 2)

            inputs_max = [get_max_value(test_inputs[0])]

            norm_test = normalize(test_inputs, inputs_max)
            matlab_data = np.reshape(np.array(pad_data(norm_test, Number_of_samples)), (1, Number_of_chirps, Number_of_samples))

            time_start = time.perf_counter()
            test_matlab = [sess.run(radar.logits, feed_dict={
                radar.signal_input: matlab_data[0], 
                radar.rnn_keep_prob: 1,
                radar.dense_drop_rate: 0
            })]
            time_end = time.perf_counter()

            test_output = denormalize(test_matlab, inputs_max)

            scipy.io.savemat('suppression_data/%d%d%d/%d%d%d%d_sup_db_tgt1_ifr1_%d_%d.mat' % (
                dt_now.year, dt_now.month, dt_now.day, 
                dt_now.year, dt_now.month, dt_now.day, dt_now.hour, signal_index, random_index), mdict={'arr': test_output})

            time_miti = time_end - time_start
            print('inference time is:%f' % time_miti)
            if radar_data != 1:
                times.append(time_miti)
    ave_time = mean(times)
    std_time = stdev(times)
    print('average_time is', ave_time, ',std is:', std_time)
