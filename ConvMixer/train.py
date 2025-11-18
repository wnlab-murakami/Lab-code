import os
import h5py
import math
import numpy as np
import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from sklearn.model_selection import train_test_split
from networks.ConvMixer import ConvMixerLayer
from networks.CV_ConvMixer import CV_ConvMixer
from networks.CV_FCN import CV_FCN

# set the processing configurations
ap = argparse.ArgumentParser(description="processing configurations ...")
ap.add_argument("-d", "--dataset", type=str, default="./dataset/",
                help="path to dataset for training")
ap.add_argument("-m", "--model", type=str, default="trained_models/",
                help="path to trained model")
ap.add_argument("-f", "--figure", type=str, default="figure/",
                help="path to loss/metrics plot")
args = vars(ap.parse_args())  # 'vars' returns the properties and property values of the object

MAX_EPOCHS = 100
LR = 0.001
BS = 32
WARMUP_EPOCHS = 5

def lr_lambda(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / WARMUP_EPOCHS  # Linear Warm-up
    else:
        return 0.5 * (1 + math.cos(math.pi * (epoch - WARMUP_EPOCHS) / (MAX_EPOCHS - WARMUP_EPOCHS)))  # Cosine Annealing

def load_data(train_file=None, scale_num=None):

    # load training data
    print("[INFO] Loading Training Data ...")
    from_filename = args["dataset"] + train_file + '.hdf5'
    f = h5py.File(from_filename, 'r')
    train_X_real = f['X_real']
    train_X_imag = f['X_imag']
    train_Y_real = f['Y_real']
    train_Y_imag = f['Y_imag']

    # distribute the real and imaginary part into two channels
    train_X = np.zeros(shape=(np.shape(train_X_real)[0], 2, np.shape(train_X_real)[1], np.shape(train_X_real)[2]))
    train_X[:, 0, :, :] = train_X_real[:, :, :]
    train_X[:, 1, :, :] = train_X_imag[:, :, :]
    train_Y = np.zeros(shape=(np.shape(train_Y_real)[0], 2, np.shape(train_Y_real)[1], np.shape(train_Y_real)[2]))
    train_Y[:, 0, :, :] = train_Y_real[:, :, :]
    train_Y[:, 1, :, :] = train_Y_imag[:, :, :]

    # clear
    f.close()

    # data normalization
    print("[INFO] Data Normalization ...")

    average = 0
    for i in range(np.shape(train_X)[0]):
        temp_X = np.sqrt(pow(train_X[i, 0, :, :], 2) + pow(train_X[i, 1, :, :], 2))
        temp_Y = np.sqrt(pow(train_Y[i, 0, :, :], 2) + pow(train_Y[i, 1, :, :], 2))
        max_X = np.max(temp_X)
        max_Y = np.max(temp_Y)
        max_complex = max(max_X, max_Y)
        train_X[i] = (train_X[i] / max_complex) * scale_num
        train_Y[i] = (train_Y[i] / max_complex) * scale_num
        average += max_complex
    # float
    train_X = train_X.astype(np.float32)
    train_Y = train_Y.astype(np.float32)
    print(average / np.shape(train_X)[0])

    # data shuffle
    print("[INFO] Data shuffle ...")
    index = [i for i in range(len(train_X))]
    np.random.seed(2020)
    np.random.shuffle(index)
    train_X = train_X[index, ...]
    train_Y = train_Y[index, ...]

    return train_X, train_Y

def FMCW_loss(b):
    mse = nn.MSELoss()
    def loss_fn(y_pred, y_true):
        loss1 = 2 * mse(y_pred, y_true)
        pred_real, pred_imag = torch.chunk(y_pred, 2, dim=1)
        # L21 norm
        # norm2 = torch.norm(pred_real, p=2, dim=2) + torch.norm(pred_imag, p=2, dim=2) # L2 norm along the time dimension
        norm2 = torch.sum(torch.sqrt(pred_real**2 + pred_imag**2), dim=2)  # L1 norm along the time dimension
        loss2 = torch.sum(norm2, dim=2) 
        loss2 = torch.mean(loss2)
        return loss1 + b * loss2
    return loss_fn

def train(dim, depth, c, saving_path, train_X_input, train_Y_input):
    print("[INFO] Data split")
    X_train, X_test, y_train, y_test = train_test_split(train_X_input, train_Y_input, test_size=0.2, random_state=42)
    print("shape of X_train: ", X_train.shape)
    print("shape of X_test: ", X_test.shape)
    print("shape of y_train: ", y_train.shape)
    print("shape of y_test: ", y_test.shape)

    train_loader = DataLoader(list(zip(X_train, y_train)), batch_size=BS, shuffle=True)
    test_loader = DataLoader(list(zip(X_test, y_test)), batch_size=BS, shuffle=False)

    print("[INFO] Model")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = ConvMixerLayer(2, dim, depth)
    # model = CV_FCN(depth, dim, 3)

    summary(model, input_size=(BS, 2, 256, 256))

    model_path = os.path.join(saving_path, "model.pth")
    
    model = model.to(device)

    print("[INFO] Loss and Optimizer")
    loss_fn = FMCW_loss(c)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    # optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)



    print("[INFO] Training")
    logs = []
    
    
    minmum_loss = 1000000
    for epoch in range(MAX_EPOCHS):
        print("-"*10, f"Epoch {epoch}", "-"*10)
        model.train()
        train_loss = 0
        for i, (X_batch, y_batch) in enumerate(train_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Loss: {train_loss/len(train_loader)}")

        model.eval()
        test_loss = 0  
        for i, (X_batch, y_batch) in enumerate(test_loader):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            test_loss += loss.item()
            
        print(f"Test Loss: {test_loss/len(test_loader)}")

        log={"epoch": epoch, "train_loss": train_loss/len(train_loader), "test_loss": test_loss/len(test_loader)}
        logs.append(log)
        if test_loss < minmum_loss:
                        minmum_loss = test_loss
                        torch.save(model.state_dict(), model_path)
                        print(f"Model saved at {model_path}")
        if (epoch+1) % 10 == 0:
            model_path_ = os.path.join(saving_path, f"model_{epoch+1}.pth")
            torch.save(model.state_dict(), model_path_)
            print(f"Model saved at {model_path_}")
        
        # Update the learning rate
        # scheduler.step()

    print("[INFO] Plot figure")
    fig, ax = plt.subplots()
    ax.plot([log["train_loss"] for log in logs], label="train_loss")
    ax.plot([log["test_loss"] for log in logs], label="test_loss")
    ax.legend()
    plt.savefig(os.path.join(args["figure"], "model.png"))

if __name__ == "__main__":
    train_X, train_Y = load_data("FMCW_train_wpmc", 1000)
    # train(16, 11, 0, args["model"], train_X, train_Y) # CV_FCN
    # train(16, 5, 0, args["model"], train_X, train_Y) # CV_ConvMixer
    train(32, 5, 0, args["model"], train_X, train_Y) # ConvMixer
