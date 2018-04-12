# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:00:42 2018

@author: minjiang
"""
import numpy as np
import os

class MyLoadData(object):
    '''
    给定数据的路径，读取相应的训练、验证和测试集数据
    输入：path
         loss_type
    输出：Train_data, Validation_data, Test_data 
    '''
 
    def __init__(self, path, loss_type):
    # _init_是错误的，__init__才是正确的
        self.path = path
        self.trainfile = self.path + "frappe.train.libfm"
        self.validationfile = self.path + "frappe.validation.libfm"
        self.testfile = self.path + "frappe.test.libfm"
        self.features_M = self.map_features( )
        self.Train_data, self.Validation_data, self.Test_data = self.construct_data( loss_type )
        
    def map_features(self):
        # 将三个数据集的特征整合
        self.features = {}
        self.read_features(self.trainfile)
        self.read_features(self.testfile)
        self.read_features(self.validationfile)
        return  len(self.features)
  
    def read_features(self, file):
        '''
        读取文件中的特征
        对于每一行，例如451:1表示one-hot编码中第451位为1
        故通过一个字典数据，将451:1映射为451
        '''
        f = open(file)
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            for item in items[1:]:
                if item not in self.features:
                    self.features[item] = int(item[:-2])
            line = f.readline()
        f.close()

    def construct_data(self, loss_type):
        # 分别构造训练集、验证集和测试集数据
        X_, Y_ , Y_logloss= self.read_data(self.trainfile)

        if loss_type == 'log_loss':
            Train_data = self.construct_dataset(X_, Y_logloss)
        else:
            Train_data = self.construct_dataset(X_, Y_)
        print("# of training:" , len(Y_))

        X_, Y_ , Y_for_logloss= self.read_data(self.validationfile)
        if loss_type == 'log_loss':
            Validation_data = self.construct_dataset(X_, Y_logloss)
        else:
            Validation_data = self.construct_dataset(X_, Y_)
        print("# of validation:", len(Y_))

        X_, Y_ , Y_for_logloss = self.read_data(self.testfile)
        if loss_type == 'log_loss':
            Test_data = self.construct_dataset(X_, Y_logloss)
        else:
            Test_data = self.construct_dataset(X_, Y_)
        print("# of test:", len(Y_))

        return Train_data,  Validation_data,  Test_data        

    def read_data(self, file):
        # 读取数据文件，对于一行数据，第一列为标签值
        # 原始标签值为1/-1
        # 其它列为相应的特征值，通过features中相应的value值来表示
        f = open( file )
        X_ = []
        Y_ = []
        Y_logloss = []
        line = f.readline()
        while line:
            items = line.strip().split(' ')
            Y_.append( 1.0*float(items[0]) )

            if float(items[0]) > 0:# > 0 as 1; others as 0
                v = 1.0
            else:
                v = 0.0
            Y_logloss.append( v )

            X_.append( [ self.features[item] for item in items[1:]] )
            line = f.readline()
        f.close()
        return X_, Y_, Y_logloss

    def construct_dataset(self, X_, Y_):
        Data_Dic = {}
        X_lens = [ len(line) for line in X_]
        #将样本按照特征类别的数目排序，最长特征类别数目为10
        indexs = np.argsort(X_lens) 
        Data_Dic['Y'] = [ Y_[i] for i in indexs]
        Data_Dic['X'] = [ X_[i] for i in indexs]
        return Data_Dic