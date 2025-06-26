import sys , os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist 
import numpy as np
import pickle

from activation_function import *

def get_data() :
    # 데이터 셋 받아오기.
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=True,one_hot_label=False)
    return x_test, t_test

def init_network() :
    with open(os.path.dirname(__file__) + "/sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
        
    return network

def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

x,t = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(x)) : 
    y = predict(network, x[i])
    p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i] :
        accuracy_cnt += 1

print("Accuracy :" + str(float(accuracy_cnt) / len(x)))