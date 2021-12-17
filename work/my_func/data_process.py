import pandas as pd
import numpy as np
from PIL import Image
from work.my_func.data_info import get_img_path

e = 2.71828

#1956x1934 size img, eye in the left, the transfered location
#mode == train
def transfer_img_location(index,lr,size,mode='train'):
    location = pd.read_excel('data/training/fovea_localization_training_GT.xlsx').iloc[index][1:] if mode == 'train' else None
    img =np.array(Image.open(get_img_path(index,mode=mode)))
    img = img.transpose(2,1,0)

    if mode == 'train':
        item_x,item_y = location
    else:
        item_x,item_y = 0,0

    if lr == 1:
        if size == 1:
            item_x = 1956 - (item_x - 518)
            item_y = item_y - 33
            img = img[:,518:518+1956,33:33+1934]
            img = img[:,::-1,:]

        else:
            item_x = 1956 - item_x
            img = img[:, ::-1, :]

    else:
        if size == 1:
            item_x = item_x - 518
            item_y = item_y - 33
            img = img[:, 518:518 + 1956, 33:33 + 1934]

        else:
            pass
    return img,[item_x,item_y]

#input location is transfered location,return 1,608,400img
#map center is 1140,1040
def get_gauss_map(location,sigma=1,r=100):
    x,y = location
    if x > 600:
        x,y = x - 1140 + 304, y - 1040 + 200
    gauss_map = np.zeros([1,608,400])
    for i in range(608):
        for j in range(400):
            distance = ((i-x)**2+(j-y)**2)**(1/2) / r
            if distance < 1:
                gauss_map[0][i][j] = 10 * e ** (-(distance/sigma)**2)
    return gauss_map

#input img 608,400,green channel
#return location
def dark_kernel_process(img,kernel_size=112):
    kernel = np.zeros([kernel_size,kernel_size])

    for i in range(kernel_size):
        for j in range(kernel_size):
            distance = ((i - kernel_size/2) ** 2 + (j - kernel_size/2) ** 2) ** (1 / 2)
            distance = distance * 2 / kernel_size
            kernel[i][j] = (1 - distance ** 2) if distance < 1 else 0
    
    max_value = 0
    for i in range((608-kernel_size)//2):
        for j in range((400-kernel_size)//2):
            value = np.mean(np.multiply(img[i*2:i*2+kernel_size,j*2:j*2+kernel_size],kernel))
            if value >= max_value:
                max_value = value
                x = i*2 + kernel_size/2
                y = j*2 + kernel_size/2
    return x,y

#统计所有点，求平均
def mean_map(map_img):
    x = 0
    y = 0
    for i in range(0,608,2):
        for j in range(0,400,2):
            x = x + i * map_img[i][j] / np.sum(map_img)
            y = y + j * map_img[i][j] / np.sum(map_img)
    return x*2,y*2

#取最大点
def max_map(map_img):
    old_value = 0
    for i in range(608//2):
        for j in range(400//2):
            if map_img[i][j] >= old_value:
                x,y = i,j
    return i,j



