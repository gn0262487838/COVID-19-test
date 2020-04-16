# /usr/bin/python3.6
# -*- coding:utf-8 -*-
# Author: HU REN BAO
# History:
#        1. first create on 20200214
#
# dataset source : https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/
#

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split



def load_data(NormalFile_dir=None, Covid19File_dir=None):
    '''

    標記:
        0 為 正常肺部Xray
        1 為 covid-19 肺部Xray

    params:
        NormalFile_dir : must be file and contain a lot of picture with normal type
        Covid19File_dir: must be file and contain a lot of picture with COVID-19 type
    
    output:
        dir of data with concate normal and COVID-19,and label be targeted

    '''

    if NormalFile_dir == None:
        NormalFile_dir = "covid19_nn/dataset/normal/"
    
    if Covid19File_dir == None:
        Covid19File_dir = "covid19_nn/dataset/covid/"

    print("========================")
    print("看一下目前工作路徑: ", os.getcwd())
    print("========================")
    
    dir_normal = glob.glob(os.path.join(os.getcwd(), NormalFile_dir) + "/" + "*")
    dir_covid = glob.glob(os.path.join(os.getcwd(), Covid19File_dir) + "/" + "*")

    print("========================")
    print("看一下normal file路徑: ", dir_normal)
    print("========================")
    
    dir_sum = dir_normal + dir_covid
    dir_sum_len = len(dir_sum)
    dir_normal_len = len(dir_normal)
    dir_covid19_len = dir_sum_len - dir_normal_len

    print(f"""
    ========================================
    正常Xray肺部照片數量    : {dir_normal_len}
    不正常Xray肺部照片數量  : {dir_covid19_len}
    總共照片數量            : {dir_sum_len}
    ========================================
    """)

    dict_ = {
        "PATH":dir_sum,
        "TARGET": [0] * dir_normal_len + [1] * dir_covid19_len
    }

    df = pd.DataFrame(dict_)
    df["TARGET"] = df["TARGET"].astype(str) # 因後面generator需要，所以要把這欄位值改為str!!!

    print(df)

    return df



def split_data(df, test_size=None, random_state=None, shuffle=True):
    '''

    切分training data, validation data, testing data

    params:
        df : DataFrame Type and must be two columns with "PATH" & "TARGET"

    output:
        return tuple like (train_X, test_X, train_Y, test_Y)

    '''
    
    if test_size == None:
        test_size = 0.25

    target_onehot = to_categorical(df["TARGET"])
    train_X, test_X, train_Y, test_Y = train_test_split(df["PATH"], df["TARGET"], shuffle=shuffle, test_size=test_size, random_state=random_state)

    return (train_X, test_X, train_Y, test_Y)



def genetator(data):
    '''

    1. 暫不考慮把字拿掉
    2. augmentation 增強數據並增加dataset，避免overfitting
    3. 不管R有沒有，都是肺部的圖片，依然都有一定的判斷作用在

    params:
        data : Type must be tuple and len equal to 4
        example.
                (train_X, test_X, train_Y, test_Y)
    
    output:
        return tuple with three generator
        example.
                (train_generator, valid_generator, test_generator)

    '''
    if not isinstance(data, tuple):
        raise TypeError("Type of data must be tuple and len equal to 4.")

    train_X, test_X, train_Y, test_Y = data

    # 因generator需要，故再合併。
    trainDf= pd.concat([train_X, train_Y], axis=1)
    testDf = pd.concat([test_X, test_Y], axis=1)

    trainDataGen = image.ImageDataGenerator(
        rotation_range=15, 
        width_shift_range=0.05, 
        height_shift_range=0.05, 
        zoom_range=0.2, 
        horizontal_flip=True,
        fill_mode='constant',
        rescale=1./255,
    )

    validationDataGen = image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="constant",
        rescale=1./255
    )

    testDataGen = image.ImageDataGenerator(
        rotation_range=30,
        rescale=1./255
    )


    train_generator = trainDataGen.flow_from_dataframe(
        trainDf, x_col="PATH", y_col="TARGET", target_size=(224, 224), batch_size=20, class_mode="categorical"
    )

    valid_generator = validationDataGen.flow_from_dataframe(
        testDf[:5], x_col="PATH", y_col="TARGET", target_size=(224, 224), batch_size=20, class_mode="categorical"
    )

    test_generator = testDataGen.flow_from_dataframe(
        testDf[5:], x_col="PATH", y_col="TARGET", target_size=(224, 224), batch_size=1, shuffle=False
    )

    return (train_generator, valid_generator, test_generator)



def show_picture(data_dir, generator, save=False):
    '''

    看一下資料集的圖片

    '''

    plt.figure(figsize=(15,20))
    i = 0
    j = 0

    nrow = 5
    ncol = 5
    order = 0
    for path in data_dir[0:5]:

        img = image.load_img(path, target_size=(224,224))
        img = image.img_to_array(img)
        img = img.reshape((1,) + img.shape)
        
        for idx, batch in enumerate(generator.flow(img, batch_size=1)):
            plt.subplot(nrow, ncol, order + idx + 1)
            plt.axis("off")
            plt.imshow(batch[0])
            i += 1
            j += 1
            if i == 5:
                i = 0
                order = j
                break
    
    if save == True:
        plt.savefig("./show.jpg")