import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('../assets/test.png')

plt.imshow(img)
plt.show()