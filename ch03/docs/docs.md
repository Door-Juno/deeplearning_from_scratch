# Chapter 03. 신경망 
신경망은 가중치 매개변수의 적절한 값을 데이터로부터 자동으로 학습하는 능력을 갖추고있다는 것이 가장 중요한 점이다.

## 3.1 신경망의 구조
![신경망](../assets/ANN.png)

신경망은 여려층의 노드의 배열로 주로 **입력층, 은닉층, 출력층**으로 구분된다.

입력층은 아무 연산이 이루어지지 않으며 입력을 받아서 다음 계층으로 넘기는 역할을 한다. 신경망에 주어지는 초기 데이터(피쳐)의 개수에 따라 노드의 수가 결정된다.

은닉층은 복잡한 문제를 해결하여 학습하는 핵심계층이다. 입력된 데이터에서 패턴을 추출하고 비선형적인 관계를 학습하는 역할을 한다. 은닉층의 개수와 각 은닉층의 노드 수는 신경망의 성능에 큰 영향을 미치며 이를 **신경망의 깊이**라고한다.

출력층은 은닉층에서의 신호를 외부로 출력하는데 사용된다.

## 3.2 활성화 함수
활성화 함수는 입력 신호의 총합을 출력신호로 변환하는 함수를 의미한다.
이 함수의 결과에 따라 다음 퍼셉트론으로 신호를 보낼지에 대한 여부와 신호의 강도를 결정하게 하므로 활성화 함수는 신경망의 복잡도를 높이는 아주 중요한 요소이다. 활성화 함수는 신경망에 **비선형성**을 부여하며, 신경망이 비선형적인 복잡한 문제도 학습할 수 있도록 한다.

### 3.2.1 시그모이드 함수 Sigmoid function

$h(x) = \frac{1} {1 + exp(-x)}$

다음 함수는 과거 신경망에서 자주 활성화되는 함수이다.

```py
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
```

![sigmoid](../assets/sigmoid.png)

시그모이드 함수의 치역은 (0,1)이며 연속적인 값을 가진다. 이는 출력을 확률처럼 해석할 수 있게 해주어 이진 분류 문제의 출력층에서 사용되기도 한다.

### 3.2.2 계단 함수
계단함수는 입력값이 0을 넘으면 1을 출력하고 그 이외에는 0을 출력하는 함수이다.
``` py
def step_funtion(x):
    y = x > 0 
    return y.astype(int)
```

![step_function](../assets/step.png)

이때, y는 `bool` 배열을 가지고있게 된다.
계단 함수는 0을 경계로 출력이 0에서 1로 바뀌는 형태를 가지고있다.

#### 활성화 함수의 역할과 비선형성
신경망에서 활성화 함수는 각 층의 출력을 비선형적으로 변환하는 핵심적인 역할을 한다.
만약 활성화 함수가 선형 함수라면, 여러 층을 쌀더라도 결국 하나의 선형 함수로 표현되어 신경망 학습의 한계가 생긴다. 비선형 활성화 함수를 사용하여야지만 신경망이 복잡한 비선형 패턴을 학습하고, XOR 문제와 같이 선형적으로 분리 불가능한 문제들을 해결할 수 있게 된다.

### 3.2.5 시그모이드 함수와 계단 함수의 비교

![compare_step&sigmoid](../assets/compare.png)

치역은 [0,1]로 동일하나 '매끄러움'의 차이를 가진다.
`sigmoid`함수는 부드로운 곡선으로 입력에 따라 출력이 연속적으로 변화하는 반면 `step`함수는 0을 경계로 출력이 급격하게 바뀌게 된다.
`sigmoid`함수의 이런 연속적인 매끈함은 신경망 학습에 매우 중요하다.

#### 기울기 소실 문제 (Vanishing Gradient Problem)
시그모이드 함수는 부드러운 곡선 형태를 가지지만, 입력값의 절댓값이 커질수록 기울기가 0개 가까워지는 **포화(Saturation)**영역이 존재한다. 이 영역에서는 기울기가 매우 작아지므로, 오차역전파 과정에서 여러 층을 거쳐 기울기가 전달될 때에 점차 작아져 결국 0으로 수렴하게 된다. 이를 **기울기 소실 문제**라고 하며, 이로 인해 특히 깊은 신경망에서 학습이 매우 느려지거나 멈추는 현상이 발생한다.

### 3.2.7 ReLU 함수
`ReLU(Rectified Linear Unit)` 함수는 0을 넘으면 그 입력을 그대로 출력하고 0 이하이면 0을 출력하는 함수이다.
``` py
def relu(x):
    return np.maximum(0,x)
```

![ReLU](../assets/ReLU.png)

$h(x) = 
\left\{\begin{matrix} x ( x >= 0 )
 \\ 0 ( x <= 0 )
\end{matrix}\right.$

ReLU함수는 양수 입력에 대해 항상 일정한 기울기를 가지므로 시그모이드 함수의 한계였던 기울기 소실 문제를 완화할 수 있다. 그러나 음수 입력에 대해서는 기울기가 0이므로, 한 번 비활성화된 뉴런은 학습중에 다시 활성화 되기 어렵다는 **죽은 ReLU 문제**가 발생할 수 있다.

## 3.4 3층 신경망 구현하기 
핵심은 신경망에서의 계산을 **행렬 계산**으로 처리할 수 있다는 것이다.

$$\begin{pmatrix}
1 & 2 \\
3 & 4 \\
\end{pmatrix} 
\begin{pmatrix}
5 & 6 \\
7 & 8 \\
\end{pmatrix} 
=
\begin{pmatrix}
19 & 22 \\
43 & 50 \\
\end{pmatrix}$$
``` python
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])

np.dot(A,B) # 곱 출력
>>>
(
    [19,22],
    [43,50]
)
```
다음과 같이 행렬의 곱은 왼쪽 행렬의 행과 오른쪽 행렬의 열을 원소별로 곱하고 그 값을 더해서 계산한다.

$$1 \times 5 +2 \times7 = 19$$
$$ 3 \times 5 + 4 \times 7 = 43 $$

이때 코드에서 `np.dot`은 행렬의 곱을 연산하는데 입력이 1차원 배열이면 벡터를, 2차원 배열이면 행렬 곱을 출력한다.

주의할 점은 행렬의 형상이다.
왼쪽행렬의 1번째 차원의 원소 수(열 수)와 오른쪽 행렬의 0번째 차원의 원소수(행 수)가 같아야한다.

즉, $A = m \times n$ 행렬이고 $B = n \times p$ 이여야지만 $A \times B$ 연산이 가능하며 결과 행렬은 $m \times p$가 된다.

### 3.4.1 신경망의 가중치와 편향 표기법
이때 이 그림은 뉴런 $x_2$에서 다음층의 뉴런인 ${a_1}^{(1)}$로 향하며 가중치는 $w_{12}^{(1)}$을 가지고 있다. 이 표기법에 대해 알아보자.

가중치의 지수부의 $(1)$은 1층의 가충치임을 명시하고있다. 또 아래의 숫자 $1$과 $2$는 앞층의 2번째 뉴런에서 다음 층의 1번째 뉴런으로 향함을 의미한다. 가중치 오른쪽 아래의 인덱스 번호는 '다음 층 번호, 앞 층 번호'순으로 적는다.

- **가중치($W$)**
  - 지수부의 $(1)$은 **1층의 가중치**임을 명시한다.
  - 아래첨자 $ij$는 $i$는 **다음 층의 뉴런 번호**, $j$는 **앞 층의 뉴런 번호**를 의미한다.
  - 입력층의 노두 수(열) X 다음층(1층)의 노드 수(행)이다.
- **편향($B$)**
  - 편향은 다음 층의 각 뉴런에 독립적으로 더해지는 값이다.
  - $b_{1}^{1}$는 1층의 1번째 뉴런에 더해지는 값이다.
  - 뉴런이 얼마나 쉽게 활성화 되는지 조절할 수 있다. 이는 선형 회귀에서 절편과 유사한 역할을 하여, 데이터가 원점을 지나지 않아도 모델이 유연하게 학습할 수 있도록 돕는다.
  
![3층신경망](../assets/make.png)

### 3.4.2 활성화 전 신호의 계산
그렇다면 ${a_1}^{(1)}$를 계산해보도록 하겠다.
$$a_{1}^{(1)} = w_{11}^{1} x_1 + w_{12}^{1}x_2 + b_{1}^{1}$$
여기서 행렬 곱을 이용해보자.
$$A^{(1)} = XW^{(1)}+B^{(1)}$$
로 간소화할 수 있다.
이때, 가중치부분은 다음과 같다.
$$
A^{(1)} = (a_{(1)}^{(1)} , a_{(2)}^{(1)}, a_{(3)}^{(1)}) (1층활성화 전 신호 벡터)
$$
$$
X = (x_1 , x_2) (입력 벡터)
$$
$$
B^{(1)} = (b_{1}^{(1)} , b_{2}^{(1)} , b_{2}^{(1)}) (1층 편향 벡터)
$$
$$
W^{(1)} = 
\begin{pmatrix}
w_{11}^{(1)} &w_{21}^{(1)}  &w_{31}^{(1)}  \\
w_{12}^{(1)} &w_{22}^{(1)}  &w_{32}^{(1)}  \\
\end{pmatrix} (1층 가중치 행렬)
$$

```python
X = np.array([1.0,0.5])
W1 = np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])

print(W1.shape) #   (2,3)
print(X.shape)  #   (2,)
print(B1.shape) #   (3,)

A1 = np.dot(X,W1) + B1
```
### 3.4.3 활성화 함수의 적용
이어서 1층의 활성화 함수의 처리를 살펴보자면 위와 같이 은닉층에서의 가중치 합을 $a$라고 표기하고 활성화 함수 $h()$의 처리결과를 $z$로 표기한다.
``` python
A1 = np.dot(X,W1) + B1
Z1 = sigmoid(A1)

print(A1)   #   [0.3 0.7 1.1]
print(Z1)   #   [0.57444252 0.66818777 0.75026011]
```
### 3.4.4 다음 층으로의 전달 (1층 -> 2층)
이어서 1층에서 2층으로 가는 과정을 구현해본다.
```python
W2 = np.array[[0.1, 0.4],[0.2, 0.5],[0.3, 0.6]]
B2 = np.array([0.1, 0.2])

print(Z1.shape) #   (3,)
print(W2.shape) #   (3,2)
print(B2.shape) #   (2,)

A2 = np.dot(Z1,W2) + B2 # 이전층의 출력은 다시 다음층의 입력이 된다.
Z2 = sigmoid(A2)
```
### 3.4.5 출력층으로의 전달
마지막으로, 은닉층에서 출력층으로의 전달을 구현해보자.
이전과의 차이점은 **활성화 함수가 다르다**.
``` python
W3 = np.array([0.1, 0.3],[0.2, 0.4])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2,W3) + B3
Y = identity_function(A3) # Y = A3 와 같다.
```
출력층의 활성화 함수는 풀고자 하는 문제의 성질에 맞게 정한다.
예를 들어 회귀에는 항등함수를, 2클래스 분류에는 시그모이드 함수를, 다중 클래스 분류에는 소프트 맥스를 사용하는 것이 일반적이다.
- **회귀(Regression)문제** -> 주택 가격 예측
- **2클래스 분류** -> 합격/불합격
- **다중 클래스 분류** -> 손글씨 숫자 인식 등

### 최종 구현 (전체 신경망 순전파)
```python

def init_network() :
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])

    return network

def forward(network, x):
    W1,W2,W3 = network['W1'],network['W2'],network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']

    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = a3
    return y

network = init_network()
x = np.array([1.0,0.5])
y = forward(network, x) #   [0.31682708 0.69627909]
print(y)
```
위 코드는 입력층에서 출력층 까지 신호가 한 방향으로 전달되는 순전파(Forward Propagation)과정을 구현하였다. 이 코드의 출력이 실제 정답과 얼마나 차이나는지 오차를 계산하여 역전파(Backpropagation)를 통해 가중치와 편향을 업데이트하며 학습을 진행한다.

## 3.5 출력층 설계하기
머신러닝 문제는 **분류(classfication)** 과 **회귀(regression)** 문제로 나뉜다.
- 분류 : 주어진 데이터가 어느 클래스에 속하느냐 하는 문제이다.
- 회귀 : 입력 데이터에서 수치를 예측하는 문제

이처럼 문제의 종류에 따라 신경망의 출력층에서 사용하는 활성화 함수를 다르게 설계해야한다.

### 3.5.1 항등 함수와 소프트 맥스 함수 구현하기
**항등 함수(identity function)**는 입력을 그대로 출력하는 함수이다.
입력된 $x$가 있다면 $x$를 그대로 출력하는 $f(x)=x$의 형태이다.

한편, 분류에서 사용하는 **소프트맥스 함수(softmax function)** 의 식은 다음과 같은데, 소프트맥스 함수는 다중 클래스 분류에서 각 클래스에 속할 확률을 출력하는데 유용하다.

$$
y_k =  \frac{\exp(a_{k})}{\sum_{i = 1}^{n} \exp(a_{i})}
$$
소프트맥스 함수의 분자는 입력신호 $a_k$의 지수 함수, 분모는 모든 입력 신호의 지수 함수의 합으로 구성된다.
``` py
def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y
```
다음과 같이 구현을 하면 **오버플로**가 발생하기 쉽다. 그렇기때문에 다른 수식을 사용해서 구현을 해보도록 하자.
$$
y_k =  \frac{\exp(a_{k})}{\sum_{i = 1}^{n} \exp(a_{i})} 
= \frac{C\exp(a_{k})}{C\sum_{i = 1}^{n} \exp(a_{i})} \\
= \frac{\exp(a_{k} + \log C)}{\sum_{i = 1}^{n} \exp(a_{i} + \log C)} \\
=  \frac{\exp(a_{k} + C')}{\sum_{i = 1}^{n} \exp(a_{i} + C')}
$$
분모와 분자에$C$라는 임의의 정수를 곱하고, 이를 지수 함수 $\exp()$안으로 옮겨 $\log C$로 만든 후, 이를 $C'$으로 바꾸어주었다.
``` py
def softmax(x):
    c = np.max(x) # 오버플로를 방지하기 위해 최댓값을 선택한다.
    exp_x = np.exp(x - c) 
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y
```

소프트맥스 함수의 출력은 0에서 1.0 사이의 실수이다. 또, 소프트맥스 함수 출력의 총 합은 1이다.
> 출력 총합이 1이 된다는 것으로 "확률"로 해석 가능하다.

신경망을 이용한 분류에서는 일반적으로 가장 큰 출력을 내는 뉴런에 해당하는 클래스로만 인식한다.
그리고 소프트맥스 함수를 적용해도 출력이 가장 큰 뉴런의 위치는 달라지지 않는다.
결과적으로, 신경망을 분류할 때에는 출력층의 소프트맥스 함수를 생략해도 된다. 현업에서도 지수 함수 계산에 드는 자원 낭비를 줄이고자 출력층의 소프트맥스 함수는 생략하는 것이 일반적이다.

### 3.5.4 출력층의 뉴런 수 정하기
출력층의 뉴런 수는 풀려는 문제에 맞게 적절하게 정해야한다. 분류에서는 분류하고자 하는 클래스의 수로 설정하는 것이 일반적이다.
예를 들어, 0~9등급으로 분류를 하겠다고 하면 출력층의 뉴런 수는 10개가 되는것이 일반적이다는 말이다.

이처럼 출력층의 설계는 머신러닝의 문제의 종류를 이해하고 그에 맞는 활성화 함수와 뉴런 수를 선택하는 것이 중요하다.

## 3.6 손글씨 숫자 인식
이제 배운 것들을 응용하여 손글씨 숫자 분류를 해보도록 하겠다.

### 3.6.1 MNIST 데이터셋
이번 실습에서 활용하는 데이터셋은 `MNIST`라는 손글씨 숫자 이미지 집합이다. MINIST는 머신러닝 분야에서 아주 유명한 데이터 셋이다.
![MNIST](../assets/MNIST.png)
다음과 같은 훈련이미지 60,000장과 시험 이미지 10,000장이 준비되어있다.

우리는 MNIST 데이터셋을 내려받아 이미지를 넘파이 배열로 변환하는 파이썬 스크립트를 사용하겠다.
``` py
import sys , os

# 부모 디렉토리의 파일을 가져올 수 있도록 커서의 현 위치를 수정( 개인 환경마다 다름 )
sys.path.append(os.path.join(os.path.dirname(__file__),'../..'))
# 옮긴 위치의 파일을 열어서 함수 불러오기 
from dataset.mnist import load_mnist 

# 데이터 셋 받아오기.
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=False)

print(x_train.shape)    #(60000, 784)
print(t_train.shape)    #(60000,)
print(x_test.shape)     #(10000, 784)
print(t_test.shape)     #(10000,)
```
만일 이 글을 보고 실습을 해보고 싶다면

[WegraLee's github](https://github.com/WegraLee/deep-learning-from-scratch)

에서 데이터셋을 받아올 수 있다.

데이터를 받아온거에서 그치지않고 이미지 파일을 실행해보자.
``` py
import sys , os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  
# 부모 디렉터리의 파일을 가져올 수 있도록 설정
# 옮긴 위치의 파일을 열어서 함수 불러오기 
from dataset.mnist import load_mnist 

import numpy as np
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

# 데이터 셋 받아오기.
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=False)

img = x_train[0]
label = t_train[0]
print(label) # 5

print(img.shape) # (784,)
img = img.reshape(28,28) # 원래 이미지 모양으로 변형
print(img.shape) #(28,28)

img_show(img)
```
![test](../assets/test.png)

다음과 같이 이미지가 잘 노출된다 !

### 3.6.2 신경망의 추론
이제 이 MNIST 데이터셋을 가지고 추론을 할 신경망을 구현해보자.

이 신경망의 입력층은 이미지 크기가 28 * 28 이라서 748로 설정을 하고, 출력층은 숫자 0에서 9를 구분하는 문제이므로 10개로 설정해주었다.

한편 은닉층은 총 두 개로 구성되며 첫 은닉층에는 50개의 뉴런을, 두 번째 은닉층에는 100개의 뉴런을 배치하겠다. (임의로 정한 값)
``` py
import sys , os

# 부모 디렉토리의 파일을 가져올 수 있도록 커서의 현 위치를 수정( 개인 환경마다 다름 )
sys.path.append(os.path.join(os.path.dirname(__file__),'../..'))
# 옮긴 위치의 파일을 열어서 함수 불러오기 
from dataset.mnist import load_mnist 

import numpy as np
import pickle

from activation_function import *

def get_data() :
    # 데이터 셋 받아오기.
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=True,one_hot_label=False)
    return x_test, t_test

def init_network() :
    with open(os.path.dirname(__file__) + "/sample_weigh.pkl",'rb') as f:
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

x,t = get_data()    # 데이터 불러오기
network = init_network()    # 신경망 구성하기

accuracy_cnt = 0

for i in range(len(x)) : 
    y = predict(network, x[i])
    p = np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i] :
        accuracy_cnt += 1

print("Accuracy :" + str(float(accuracy_cnt) / len(x))) 
# Accuracy :0.9352
```
- `init_network()`
  - pickle 파일인 `sampe_weight.pkl`에 저장된 학습된 가충치 매개변수를 읽어온다.
  - 이 파일에는 가중치와 편향 변수가 딕셔너리 변수로 저장되어있다.
- `load_mnist`의 인수인 `normalize`를 True로 설정하면 0~255범위인 픽셀 값들을 0.0 ~ 1.0의 범위로 정규화 한다.

정답을 판정하는 기준은 `np.argmax(y)`를 통해 해당 배열에서 확률이 가장 높은 원소의 인덱스를 구하여 예측 결과를 구하고, 이 결과(`p`)와 정답 레이블(`t[i]`)를 비교화여 같다면 정답으로 처리하는것이다.

위 함수의 실행결과 `Accuracy : 0.9352` 가 나온다. 올바르게 분류한 비율이 93.52% 이라는 의미이다.

### 3.6.3 배치 처리
이번에는 입력 데이터와 가중치 매개변수의 '형상'에 대해 주의 해 보도록 하겠다.
``` py
x , _ = get_data()
network = init_network()
w1, w2, w3 = network['W1'], network['W2'], network['W3']

print(x.shape)      # (10000,784)
print(x[0].shape)   # (784,)
print(w1.shape)     # (784,50)
print(w2.shape)     # (50,100)
print(w3.shape)     # (100,10)
```
이 결과에서 다차원 배열의 대응하는 차원의 원소 수가 일치함을 확인할 수 있다.

(784(입력층의 수) -> 50(은닉 1층의 수) -> 100(은닉 2층의 수) -> 10(출력층의 수))

$$
형상 : 784 ~~784\times50~~50\times100~~100\times10~~10
$$

