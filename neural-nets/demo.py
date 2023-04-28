from PIL import Image
import numpy as np

image = Image.open('2000.jpg')
array_img = np.array(image)
array_img = array_img/255

print(array_img)
