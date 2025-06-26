import sys , os

# 부모 디렉토리의 파일을 가져올 수 있도록 커서의 현 위치를 수정( 개인 환경마다 다름 )
sys.path.append(os.path.join(os.path.dirname(__file__),'../..'))
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