import sys , os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist 
import numpy as np
import pickle

from activation_function import *
from neuralnet_minist import get_data, init_network , predict

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0

for i in range(0, len(x), batch_size) :
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy: "+ str(float(accuracy_cnt)/len(x)))