#!/usr/bin/evn python
# -*- coding: utf-8 -*-

import sys
import torch
import numpy as np
import cv2
from torch import optim
from fc_net import Fc_Net
from torch.autograd import Variable
from scipy.stats import mode
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pickle
import csv
from progressbar import *
import random
from config import *


def predict(model, x_val):
    x = Variable(x_val.cuda(), requires_grad=False)
    output = model.forward(x)
    if type(output) == tuple:
        return output[0].cpu().data.numpy().argmax(axis=1)
    return output.cpu().data.numpy().argmax(axis=1)

def predict_all(model, x_val):
    x = Variable(x_val.cuda(), requires_grad=False)
    output = model.forward(x)
    if type(output) == tuple:
        return output[0].cpu().data.numpy().argmax(axis=1)
    return output.cpu().data.numpy()


def predict_feature(model, x_val):
    x = Variable(x_val.cuda(), requires_grad=False)
    output = model.forward(x)
    if type(output) == tuple:
        print output[1].size()
        return output[1].cpu().data.numpy()
    return output.cpu().data.numpy().argmax(axis=1)


def train_model(model, loss, optimizer, x_val, y_val):
    x = Variable(x_val.cuda(), requires_grad=False)
    y = Variable(y_val.cuda(), requires_grad=False)
    optimizer.zero_grad()
    fx = model.forward(x)
    # inecption 网络特有
    if type(fx) == tuple:
        t_y = fx[0].cpu().data.numpy().argmax(axis=1)
        acc = 1. * np.mean(t_y == y_val.numpy())
        output = loss.forward(fx[0], y)
    else:
        t_y = fx.cpu().data.numpy().argmax(axis=1)
        acc = 1. * np.mean(t_y == y_val.numpy())
        output = loss.forward(fx, y)
    output.backward()
    optimizer.step()
    return output.cuda().data[0], acc


def predict_ens():
    inception_data = np.load('feature_test_inception_v3.npy').astype(np.float)
    densenet_data = np.load('feature_test_densenet161.npy').astype(np.float)
    resnet_data = np.load('feature_test_resnet101.npy').astype(np.float)
    lable = np.load('lable_test_resnet101.npy')
    add_data = (inception_data + resnet_data)
    _inception_data = inception_data.copy()
    _resnet_data = resnet_data.copy()
    _inception_data = _inception_data[..., np.newaxis]
    _resnet_data = _resnet_data[..., np.newaxis]
    max_data = np.concatenate((_inception_data, _resnet_data), axis=2)
    max_data = max_data.max(axis=2)
    all_data = np.concatenate((inception_data, densenet_data, resnet_data, add_data), axis=1)
    model = torch.load('models/fcnet_model_shuffle_SGD_123_4.pkl')
    model.training = False
    batch_size = 128
    predict_lable = np.zeros((0))
    num_batches_train = int(all_data.shape[0] / batch_size) + 1
    for i in range(num_batches_train):
        start, end = i * batch_size, (i + 1) * batch_size
        batch_trX = all_data[start:end]
        predY = predict(model, torch.from_numpy(batch_trX).float())
        predict_lable = np.concatenate((predict_lable, predY), axis=0)
    print predict_lable.shape
    predict_lable = predict_lable[:len(lable)]
    print predict_lable.shape
    dog_key = os.listdir(Image_Path)
    key_map = {dog_key[x]: x for x in range(100)}

    for i in range(len(lable)):
        for key, value in key_map.iteritems():
            if value == predict_lable[i]:
                with open('predict_dog_ens_1.txt', 'a') as f:
                    f.write('%s\t%s\n' % (key, lable[i]))


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = 0.05 * (0.95 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def CV_train():
    inception_data = np.load('feature_inception_v3.npy').astype(np.float)
    densenet_data = np.load('feature_densenet161.npy').astype(np.float)
    resnet_data = np.load('feature_resnet101.npy').astype(np.float)
    lable = np.load('lable_resnet101.npy')

    inception_test_data = np.load('feature_test_inception_v3.npy').astype(np.float)
    densenet_test_data = np.load('feature_test_densenet161.npy').astype(np.float)
    resnet_test_data = np.load('feature_test_resnet101.npy').astype(np.float)
    lable_test = np.load('lable_test_resnet101.npy')

    add__test_data = (inception_test_data + resnet_test_data)
    all_test_data = np.concatenate((inception_test_data, densenet_test_data, resnet_test_data, add__test_data), axis=1)


    add_data = (inception_data + resnet_data)
    all_data = np.concatenate((inception_data, densenet_data, resnet_data, add_data), axis=1)
    # nn = range(len(all_data))
    # np.random.shuffle(nn)
    # all_data = all_data[nn]
    # lable = lable[nn]
    test_preds = []
    skf = StratifiedKFold(n_splits=10)
    for train_index, test_index in skf.split(all_data, lable):
        train_X = all_data[train_index]
        test_X = all_data[test_index]
        train_Y = lable[train_index]
        test_Y = lable[test_index]

        batch_size = 128

        model = Fc_Net(all_data.shape[1], 100)
        model = model.cuda()
        loss = torch.nn.CrossEntropyLoss(size_average=True)
        loss = loss.cuda()

        optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.75, weight_decay=1e-4)
        last_acc = 0.0
        epochs = 40
        for e in range(epochs):
            adjust_learning_rate(optimizer, e)
            num_batches_train = int(train_X.shape[0] / batch_size) + 1
            train_acc = 0.0
            cost = 0.0
            widgets = ['Progress: ', Percentage(), ' ', Bar(marker=RotatingMarker('>'))]
            pbar = ProgressBar(widgets=widgets, maxval=num_batches_train)
            pbar.start()
            model.training = True
            for i in range(num_batches_train):
                start, end = i * batch_size, (i + 1) * batch_size
                batch_trX = train_X[start:end]
                batch_trY = train_Y[start:end]
                tor_batch_trX = torch.from_numpy(batch_trX).float()
                tor_batch_trY = torch.from_numpy(batch_trY).long()
                cost_temp, acc_temp = train_model(model, loss, optimizer, tor_batch_trX, tor_batch_trY)
                train_acc += acc_temp
                cost += cost_temp
                pbar.update(i)
            pbar.finish()
            print 'Epoch %d ,all average train loss is : %f' % (e, cost / (num_batches_train))
            print 'Epoch %d ,all average train acc is : %f' % (e, train_acc / (num_batches_train))
            model.training = False
            acc = 0.0
            num_batches_test = int(test_X.shape[0] / batch_size) + 1
            for j in range(num_batches_test):
                start, end = j * batch_size, (j + 1) * batch_size
                predY = predict(model, torch.from_numpy(test_X[start:end]).float())
                acc += 1. * np.mean(predY == test_Y[start:end])

            print 'Epoch %d ,all test acc is : %f' % (e, acc / num_batches_test)
            last_acc = acc / num_batches_test
            # torch.save(model, 'models/fcnet_model_shuffle_%s_%s_4.pkl' % ('SGD', str(e)))
        model.training = False
        #预测样本
        predict_lable = np.zeros((0))
        num_batches_train = int(all_test_data.shape[0] / batch_size) + 1
        for i in range(num_batches_train):
            start, end = i * batch_size, (i + 1) * batch_size
            batch_trX = all_test_data[start:end]
            predY = predict(model, torch.from_numpy(batch_trX).float())
            predict_lable = np.concatenate((predict_lable, predY), axis=0)
        print predict_lable.shape
        predict_lable = predict_lable[:len(lable_test)]
        print predict_lable.shape
        if (last_acc) > 0.75:
            test_preds.append(predict_lable)
    test_preds = np.array(test_preds)
    dog_key = os.listdir(Image_Path)
    key_map = {dog_key[x]: x for x in range(100)}

    for i in range(len(lable_test)):
        for key, value in key_map.iteritems():
            mode_value = mode(test_preds[:,i])[0][0]
            if value == mode_value:
                with open('predict_dog_ens_4.txt', 'a') as f:
                    f.write('%s\t%s\n' % (key, lable_test[i]))

def pig_predict():
    feature_file = open('inception_resnet_v2_testA_feature.pkl','rb')
    all_feature = pickle.load(feature_file)
    feature_file.close()

    all_data = []
    all_label = []
    for key,value in all_feature.iteritems():
        lable = int(key.split('.')[0])
        all_label.append(lable)
        all_data.append(value)
    all_data = np.array(all_data)
    lable = np.array(all_label)
    testX = torch.from_numpy(all_data).float()
    model = torch.load('models/fcnet_model_shuffle_SGD_112_0.pkl')
    model.training =False
    lable_Y = predict_all(model, testX)
    tag = []
    for index in range(len(all_label)):
        image_name= int(all_label[index])
        for y in range(len(lable_Y[index])):
            label = lable_Y[index,y]
            tag.append([image_name,y+1,str('%.8f'%(label))])
    with open('out.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
    for x in tag:
        writer.writerow(x)

    print lable_Y[0]
    print lable_Y[10].sum()
    print lable_Y[100].sum()
    print lable_Y[1000].sum()
    print lable_Y[2000].sum()
    print lable_Y[2500].sum()
    print lable_Y[2501].sum()

def train():
    feature_file = open('inception_resnet_v2_feature.pkl','rb')
    all_feature = pickle.load(feature_file)
    feature_file.close()
    # all_feature={}
    all_data = []
    all_label = []
    for key,value in all_feature.iteritems():
        lable = int(key.split('_')[0])
        all_label.append(lable)
        all_data.append(value)
    all_data = np.array(all_data)
    lable = np.array(all_label)
    nn = range(len(all_data))
    np.random.shuffle(nn)
    all_data = all_data[nn]
    lable = lable[nn]
    proportion = 0.85
    batch_size = 256
    train_X = all_data[:int(all_data.shape[0] * proportion)]
    test_X = all_data[int(all_data.shape[0] * proportion):]

    train_Y = lable[:int(lable.shape[0] * proportion)]
    test_Y = lable[int(lable.shape[0] * proportion):]
    print all_data.shape
    print lable.shape
    model = Fc_Net(all_data.shape[1], 30)
    model = model.cuda()
    loss = torch.nn.CrossEntropyLoss(size_average=True)
    loss = loss.cuda()

    optimizer = optim.SGD(model.parameters(), lr=0.0005, momentum=0.75, weight_decay=1e-4)

    epochs = 1000
    for e in range(epochs):
        adjust_learning_rate(optimizer, e)
        num_batches_train = int(train_X.shape[0] / batch_size) + 1
        train_acc = 0.0
        cost = 0.0
        widgets = ['Progress: ', Percentage(), ' ', Bar(marker=RotatingMarker('>'))]
        pbar = ProgressBar(widgets=widgets, maxval=num_batches_train)
        pbar.start()
        model.training = True
        for i in range(num_batches_train):
            start, end = i * batch_size, (i + 1) * batch_size
            batch_trX = train_X[start:end]
            batch_trY = train_Y[start:end]
            tor_batch_trX = torch.from_numpy(batch_trX).float()
            tor_batch_trY = torch.from_numpy(batch_trY).long()
            cost_temp, acc_temp = train_model(model, loss, optimizer, tor_batch_trX, tor_batch_trY)
            train_acc += acc_temp
            cost += cost_temp
            pbar.update(i)
        pbar.finish()
        print 'Epoch %d ,all average train loss is : %f' % (e, cost / (num_batches_train))
        print 'Epoch %d ,all average train acc is : %f' % (e, train_acc / (num_batches_train))
        model.training = False
        acc = 0.0
        num_batches_test = int(test_X.shape[0] / batch_size) + 1
        for j in range(num_batches_test):
            start, end = j * batch_size, (j + 1) * batch_size
            predY = predict(model, torch.from_numpy(test_X[start:end]).float())
            acc += 1. * np.mean(predY == test_Y[start:end])

        print 'Epoch %d ,all test acc is : %f' % (e, acc / num_batches_test)
        torch.save(model, 'models/fcnet_model_shuffle_%s_%s_0.pkl' % ('SGD', str(e)))


if __name__ == '__main__':
    pig_predict()
