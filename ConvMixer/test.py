# import the necessary packages
import h5py
import numpy as np
import math
import datetime
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from networks.ConvMixer import ConvMixerLayer
from networks.CV_FCN import CV_FCN
from networks.CV_ConvMixer import CV_ConvMixer

def load_data(testfile='FMCW_test', signal_length=None, scale_num=None):
    # load data
    print("[INFO] Loading testing data ...")
    f = h5py.File('./dataset/' + testfile + '.hdf5', 'r')
    test_X_real = f['X_real']
    test_X_imag = f['X_imag']

    print('Before reshaping: Shape of test_X (real part):' + str(np.shape(test_X_real)))

    # reshape
    test_X = np.zeros(shape=(np.shape(test_X_real)[0], 2, 256, 256))
    test_X[:, 0, :, :] = test_X_real[:, :, :]
    test_X[:, 1, :, :] = test_X_imag[:, :, :]

    print('After reshaping: Shape of test_X :' + str(np.shape(test_X)))

    f.close()

    # Data Normalization
    print("[INFO] Data Normalization ...")

    norm = []
    for i in range(np.shape(test_X)[0]):
        temp_X = np.sqrt(pow(test_X[i, 0, :, :], 2) + pow(test_X[i, 1, :, :], 2))
        max_X = np.max(temp_X)
        norm.append(max_X)
        test_X[i] = (test_X[i] / max_X) * scale_num
    test_X = test_X.astype(np.float32)
    return test_X, norm


def test(dim, depth, saving_path, signal_length, test_X_input, norm, scale_num):
    # load trained model
    model = ConvMixerLayer(2, dim, depth)
    # model = CV_ConvMixer(1, dim, depth)
    # model = CV_FCN(depth, dim, 3)
    summary(model, input_size=(1, 2, 256, 256))
    # load weights
    para = saving_path
    if scale_num == 1000:
        model.load_state_dict(torch.load('trained_models/' + para + '.pth'))
    else:
        model.load_state_dict(torch.load('trained_models/' + para + '.pth'))
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # View the output figure of the middle layer
    # intermediate_layer_model = Model(inputs=model.get_input_at(0),
    #                                  outputs=model.get_layer('subtract_10').output)
    # pred_Y = intermediate_layer_model.predict(test_X_input, batch_size=32, verbose=1)

    # data loader 
    test_X_input = DataLoader(test_X_input, batch_size=32, shuffle=False)
    # split mini-batch


    # predict
    model.eval()
    pred_Y = None
    
    with torch.no_grad():  # メモリ使用量を減らすために勾配計算を無効にする
        starttime = datetime.datetime.now()
        for i, data in enumerate(test_X_input):
            data = data.to(device).float()  # データをGPUに移動
            # with torch.no_grad():  # メモリ使用量を減らすために勾配計算を無効にする
            if i == 0:
                pred_Y = model(data)
            else:
                pred_Y = torch.cat((pred_Y, model(data)), 0)
        endtime = datetime.datetime.now()

    print("[INFO] Testing data shape: " + str(np.shape(pred_Y)))
    # denorm
    for i in range(np.shape(pred_Y)[0]):
        pred_Y[i] = (pred_Y[i] * norm[i]) / scale_num

    # Tensor to numpy
    pred_Y = pred_Y.cpu().detach().numpy()
    pred_Y_real = pred_Y[:, 0, :, :]
    pred_Y_imag = pred_Y[:, 1, :, :]

    # create hdf5 file
    if scale_num == 1000:
        if signal_length == 16129:
            f = h5py.File('./dataset/realdata/pred_chimney_resnet_sar.hdf5', 'w')
        else:
            f = h5py.File('./dataset/' + saving_path + '_' + str(signal_length) + '_exp' + '.hdf5', 'w')
    else:
        f = h5py.File('./dataset/' + saving_path + '_' + str(signal_length) + '_exp'+ '.hdf5', 'w')
    d1 = f.create_dataset('Y_real', data=pred_Y_real)
    d2 = f.create_dataset('Y_imag', data=pred_Y_imag)
    d1[...] = pred_Y_real[:, :, :]
    d2[...] = pred_Y_imag[:, :, :]
    f.close()

    print("Testing time(ms):" + str(((endtime-starttime).seconds * 1000 + (endtime-starttime).microseconds / 1000)))
    print("Average time(ms):" + str(((endtime-starttime).seconds * 1000 + (endtime-starttime).microseconds / 1000) / np.shape(pred_Y)[0]))
    print("[INFO] Completed")


if __name__ == '__main__':

    # load data
    signal_length = 256
    scale_num = 1000

    if signal_length == 256:
        test_file = 'FMCW_test_exp3'
    elif signal_length == 1025:
        test_file = 'FMCW_test2'
    elif signal_length == 16129:
        test_file = 'realdata/chimney_small'
    else:
        raise Exception("The signal length is wrong!")

    test_X_input, norm = load_data(testfile=test_file, signal_length=signal_length, scale_num=scale_num)
    test(32, 5, 'ConvMixer_wpmc/model', signal_length, test_X_input, norm, scale_num)
    # test(16, 11, 'CVFCN_2509_awgnlbl/model_100', signal_length, test_X_input, norm, scale_num)
    # test(32, 5, 'ConvMixer_2509_awgnlbl/model_100', signal_length, test_X_input, norm, scale_num)
