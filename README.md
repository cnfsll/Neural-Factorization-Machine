# Neural-Factorization-Machine
基于TensorFlow实现Neural-Factorization-Machine
参考如下：
Xiangnan He and Tat-Seng Chua (2017). Neural Factorization Machines for Sparse Predictive Analytics. In Proceedings of SIGIR '17, Shinjuku, Tokyo, Japan, August 07-11, 2017.
https://github.com/hexiangnan/neural_factorization_machine

LoadData.py：数据读取
NeuralFM_Model.py：模型定义
Run_NeuralFM_SquareLoss.py：针对平方误差损失，训练模型
Run_NeuralFM_LogLoss.py：针对对数似然损失，训练模型（对于frappe数据集，采用该损失很难找到合适的超参数）

