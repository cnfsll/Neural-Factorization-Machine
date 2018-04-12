# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:31:56 2018

定义NFM模型

@author: minjiang
"""

import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.contrib.layers import batch_norm
from sklearn.metrics import roc_auc_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import numpy as np
import math
from time import time

class NeuralFM(BaseEstimator, TransformerMixin):
    
    def __init__(self, features_M, hidden_factor, layers, epoch, batch_size, learning_rate, lamda_bilinear,
                 keep_prob, optimizer_type, bn, activation_function, loss_type, verbose, early_stop, random_seed=2016):
        '''
        features_M:特征维度
        hidden_factor: embedding维度
        layers:各隐藏层神经元数目
        epoch：迭代次数
        batch_size:batch的大小
        learning_rate:学习率
        lamda_bilinear:对bilinear层使用L2正则化，如果为0表示不使用
        keep_prob:对bilinear层和隐藏层使用dropout,1:no dropout
        optimizer_type:优化方法选择
        bn:是否对隐藏层使用BN，1：使用
        activation_function: 激活函数
        loss_type：损失函数类型 'square_loss' or 'log_loss'
        verbose:显示运行结果，每X次迭代显示一次
        early_stop:是否应用早停策略
        '''
        self.features_M = features_M    
        self.hidden_factor = hidden_factor
        self.layers = layers
        self.epoch = epoch        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lamda_bilinear = lamda_bilinear
        self.keep_prob = keep_prob
        self.optimizer_type = optimizer_type
        self.bn = bn
        self.activation_function = activation_function
        self.loss_type = loss_type
        self.verbose = verbose
        self.early_stop = early_stop
        self.random_seed = random_seed
        # 迭代误差
        self.train_loss, self.valid_loss, self.test_loss = [], [], [] 
        
        # init all variables in a tensorflow graph
        self._init_graph()    
    
    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default(): #tf.device('/cpu:0'):
            tf.set_random_seed(self.random_seed)
            # 输入数据
            self.train_features = tf.placeholder(tf.int32, shape = [None, None]) # None * features_M
            self.train_labels = tf.placeholder(tf.float32, shape = [None, 1])
            self.dropout_keep = tf.placeholder(tf.float32, shape = [None])
            self.train_phase = tf.placeholder(tf.bool)
            
            # 变量
            self.weights = self._initialize_weights()
    
            # 模型定义
            # ————————————和的平方项————————————
            # 获取特征embeddings的和
            nonzero_embeddings = tf.nn.embedding_lookup(self.weights['feature_embeddings'], self.train_features)
            # weights['feature_embeddings']: features_M * K
            # train_features ids = [id1, id2, ..., id10] 0 <=id < features_M
            # None * shape(ids) * K
            self.summed_features_emb = tf.reduce_sum(nonzero_embeddings, 1) # None * K
            # 获取对K维向量中每个元素求平方的结果
            self.summed_features_emb_square = tf.square(self.summed_features_emb) # None * K
            
            # ___________平方的和项———————————————
            self.squared_features_emb = tf.square(nonzero_embeddings)
            self.squared_sum_features_emb = tf.reduce_sum(self.squared_features_emb, 1) #None * K
            
            # __________FM__________
            self.FM = 0.5 * tf.subtract(self.summed_features_emb_square, self.squared_sum_features_emb)  # None * K
            if self.bn:
                self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_fm')            
            self.FM = tf.nn.dropout(self.FM, self.dropout_keep[-1]) # 对bilinear层使用dropout
            
            # ________Deep Layers________
            for i in range(0, len(self.layers)):
                self.FM = tf.add(tf.matmul(self.FM, self.weights['layer_%d' %i]), self.weights['bias_%d' %i]) #None * layers[i] *1
                if self.bn:
                    self.FM = self.batch_norm_layer(self.FM, train_phase=self.train_phase, scope_bn='bn_%d' %i) # None * layers[i] * 1
                self.FM = tf.nn.dropout(self.FM, self.dropout_keep[i]) # dropout at each Deep layer
                self.FM = self.activation_function(self.FM)
            self.FM = tf.matmul(self.FM, self.weights['prediction'])     # None * 1

            # ________输出________
            Feature_Inter = tf.reduce_sum(self.FM, 1, keep_dims=True)  # None * 1
            self.Feature_bias = tf.reduce_sum(tf.nn.embedding_lookup(self.weights['feature_bias'], self.train_features) , 1)  # None * 1
            Bias = self.weights['bias'] * tf.ones_like(self.train_labels)  # None * 1
            self.out = tf.add_n([Feature_Inter, self.Feature_bias, Bias])  # None * 1

            # 计算损失
            if self.loss_type == 'square_loss':
                if self.lamda_bilinear > 0:
                    self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out)) + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['feature_embeddings'])  # regulizer
                else:
                    self.loss = tf.nn.l2_loss(tf.subtract(self.train_labels, self.out))
            elif self.loss_type == 'log_loss':
                self.out = tf.sigmoid(self.out)
                if self.lamda_bilinear >0:
                    self.loss = tf.losses.log_loss(self.train_labels, self.out) + tf.contrib.layers.l2_regularizer(self.lamda_bilinear)(self.weights['feature_embeddings'])
                else:
                    self.loss = tf.losses.log_loss(self.train_labels, self.out)
                    
            # 优化方法
            if self.optimizer_type == 'AdamOptimizer':
                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'AdagradOptimizer':
                self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate, initial_accumulator_value=1e-8).minimize(self.loss)
            elif self.optimizer_type == 'GradientDescentOptimizer':
                self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
            elif self.optimizer_type == 'MomentumOptimizer':
                self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate, momentum=0.95).minimize(self.loss)             
            
            # 初始化
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            
            # 参数数目
            total_parameters = 0
            for variable in self.weights.values():
                shape = variable.get_shape()
                variable_parameters = 1
                for dim in shape:
                    variable_parameters *= dim.value
                total_parameters += variable_parameters
            if self.verbose > 0:
                print ("#params: %d" %total_parameters)             
            
            
    def _initialize_weights(self):
        all_weights = dict()
        all_weights['feature_embeddings'] = tf.Variable(
            tf.random_normal([self.features_M, self.hidden_factor], 0.0, 0.01), name='feature_embeddings')  # features_M * K
        all_weights['feature_bias'] = tf.Variable(tf.random_uniform([self.features_M, 1], 0.0, 0.0), name='feature_bias')  # features_M * 1
        all_weights['bias'] = tf.Variable(tf.constant(0.0), name='bias')  # 1 * 1
        # deep layers
        num_layer = len(self.layers)
        if num_layer > 0:
            glorot = np.sqrt(2.0 / (self.hidden_factor + self.layers[0]))
            all_weights['layer_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.hidden_factor, self.layers[0])), dtype=np.float32)
            all_weights['bias_0'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(1, self.layers[0])), dtype=np.float32)  # 1 * layers[0]
            for i in range(1, num_layer):
                glorot = np.sqrt(2.0 / (self.layers[i-1] + self.layers[i]))
                all_weights['layer_%d' %i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(self.layers[i-1], self.layers[i])), dtype=np.float32)  # layers[i-1]*layers[i]
                all_weights['bias_%d' %i] = tf.Variable(
                    np.random.normal(loc=0, scale=glorot, size=(1, self.layers[i])), dtype=np.float32)  # 1 * layer[i]
	       # prediction layer
            glorot = np.sqrt(2.0 / (self.layers[-1] + 1))
            all_weights['prediction'] = tf.Variable(np.random.normal(loc=0, scale=glorot, size=(self.layers[-1], 1)), dtype=np.float32)  # layers[-1] * 1
        else:
            all_weights['prediction'] = tf.Variable(np.ones((self.hidden_factor, 1), dtype=np.float32))  # hidden_factor * 1       
        return all_weights
    
    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=0.9, center=True, scale=True, updates_collections=None,
            is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z        

    def partial_fit(self, data):  # 对一个batch的数据进行训练
        feed_dict = {self.train_features: data['X'], self.train_labels: data['Y'], self.dropout_keep: self.keep_prob, self.train_phase: True}
        loss, opt = self.sess.run((self.loss, self.optimizer), feed_dict=feed_dict)
        return loss
        
    def get_random_block_from_data(self, data, batch_size):  # 从训练数据中产生随机产生一个batch的数据
        start_index = np.random.randint(0, len(data['Y']) - batch_size)
        X , Y = [], []
        # forward get sample
        i = start_index
        while len(X) < batch_size and i < len(data['X']):
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i + 1
            else:
                break
        # backward get sample
        i = start_index
        while len(X) < batch_size and i >= 0:
            if len(data['X'][i]) == len(data['X'][start_index]):
                Y.append([data['Y'][i]])
                X.append(data['X'][i])
                i = i - 1
            else:
                break
        return {'X': X, 'Y': Y}        
        
    def shuffle_in_unison_scary(self, a, b): #将数据打乱，特征和标签执行相同的打乱
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)


    def train(self, Train_data, Validation_data, Test_data):
        # 初始性能检测
        if self.verbose > 0:
            t2 = time()
            init_train = self.evaluate(Train_data)
            init_valid = self.evaluate(Validation_data)
            init_test = self.evaluate(Test_data)
            print("Init: \t train=%.4f, validation=%.4f, test=%.4f [%.1f s]" %(init_train, init_valid, init_test, time()-t2))
        Epoch = []
        for epoch in range(self.epoch):
            t1 = time()
            self.shuffle_in_unison_scary(Train_data['X'], Train_data['Y']) #数据打乱
            total_batch = int(len(Train_data['Y']) / self.batch_size)
            #loss_i = []
            for i in range(total_batch):
                # 产生一个batch数据
                batch_xs = self.get_random_block_from_data(Train_data, self.batch_size)
                # 训练
                self.partial_fit(batch_xs)
                #loss_i = self.partial_fit(batch_xs) #如有必要可以查看每一个batch的训练误差
                #print(loss_i)
            t2 = time()
            
            # 损失计算
            train_result = self.evaluate(Train_data)
            valid_result = self.evaluate(Validation_data)
            test_result = self.evaluate(Test_data)
            
            Epoch.append(epoch)
            self.train_loss.append(train_result)
            self.valid_loss.append(valid_result)
            self.test_loss.append(test_result)
            if self.verbose > 0 and epoch % self.verbose == 0:
                print("Epoch %d [train_time%.1f s]\ttrain=%.4f, validation=%.4f, test=%.4f [eval_time%.1f s]" 
                      %(epoch+1, t2-t1, train_result, valid_result, test_result, time()-t2))
            if self.early_stop > 0 and self.eva_termination(self.valid_loss):
                print ("Early stop at %d based on validation result." %(epoch+1))
                break
        plt.plot(Epoch, self.train_loss, 'r-', label = 'Train_loss')
        plt.plot(Epoch, self.valid_loss, 'b-', label = 'Valid_loss')        
        plt.plot(Epoch, self.test_loss, 'y-', label = 'Test_loss')
        plt.title('训练集、验证集、测试集误差变化')
        plt.ylabel('Rmse')
        plt.xlabel('epoch')
        plt.legend()
        plt.show()
            
    def eva_termination(self, valid): #早停
        if len(valid) > 5:
            if valid[-1] > valid[-2] and valid[-2] > valid[-3] and valid[-3] > valid[-4] and valid[-4] > valid[-5]:
                return True
        return False

    def evaluate(self, data):
        num_example = len(data['X'])
        feed_dict = {self.train_features: data['X'], self.train_labels: [[y] for y in data['Y']], self.dropout_keep: [1], self.train_phase: False}
        predictions = self.sess.run((self.out), feed_dict=feed_dict)
        y_pred = np.reshape(predictions, (num_example,))
        y_true = np.reshape(data['Y'], (num_example,))
        
        if self.loss_type == 'square_loss':    
            predictions_bounded = np.maximum(y_pred, np.ones(num_example) * min(y_true))  # bound the lower values
            predictions_bounded = np.minimum(predictions_bounded, np.ones(num_example) * max(y_true))  # bound the higher values
            RMSE = math.sqrt(mean_squared_error(y_true, predictions_bounded))
            return RMSE
        elif self.loss_type == 'log_loss':
            logloss = log_loss(y_true, y_pred)
            return logloss
        '''
        logloss = log_loss(y_true, y_pred)
        #logloss = roc_auc_score(y_true, y_pred)
        #logloss = -np.sum((y_true*np.log(y_pred) + (1-y_true)*np.log(1-y_pred))) / (num_example)
        return logloss        
        '''        