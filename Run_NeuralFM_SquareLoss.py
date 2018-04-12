# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:20:10 2018

@author: minjiang
"""
import tensorflow as tf
import LoadData as DATA
from NeuralFM_Model import NeuralFM
import numpy as np
from time import time

#定义参数
#对于square_loss，下列超参数较合理
#对于log_loss，下列超参数不合理，需调整
Path = 'D:/Recommender System/neural_factorization_machine-master/data/frappe/'
Epoch = 50
Batch_size = 128
Hidden_factor = 64
Layers = '[64]'
Keep_prob = '[0.8, 0.5]'
Lamda = float(0)
Lr = 0.05
Optimizer = 'AdagradOptimizer'
Verbose = 1
Bn = 1
Activation = 'relu'
Loss_type = 'square_loss'
#Loss_type = 'log_loss'
Early_stop = 1

# 读取数据
data = DATA.MyLoadData(Path, Loss_type)

activation_function = tf.nn.relu
if Activation == 'sigmoid':
    activation_function = tf.sigmoid
elif Activation == 'tanh':
    activation_function == tf.tanh
elif Activation == 'identity':
    activation_function = tf.identity

# 训练
t1 = time()
model = NeuralFM(data.features_M, Hidden_factor, eval(Layers), Epoch, Batch_size, Lr, Lamda, eval(Keep_prob), Optimizer, Bn, activation_function, Loss_type, Verbose, Early_stop)
model.train(data.Train_data, data.Validation_data, data.Test_data)

# 找到使验证集误差最小的迭代次数
best_epoch  = np.argmin(model.valid_loss)
print ("Best Iter(validation)= %d\t train = %.4f, valid = %.4f, test = %.4f [%.1f s]" 
       %(best_epoch+1, model.train_loss[best_epoch], model.valid_loss[best_epoch], model.test_loss[best_epoch], time()-t1))