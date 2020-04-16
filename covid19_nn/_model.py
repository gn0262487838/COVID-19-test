# /usr/bin/python3.6
# -*- coding:utf-8 -*-
# Author: HU REN BAO
# History:
#        1. first create on 20200214
#

import tensorflow.keras as tk



class ResNet():

    def __init__(self, learning_rate=None):
        
        self.input_shape = (224, 224, 3)

        if learning_rate == None:
            self.learning_rate = 0.001
        else:
            self.learning_rate = learning_rate
        

    def resNet50(self, include_top=False, weights="imagenet", freezeWeights=False):
        
        if weights != "imagenet":
            weights = None

        if include_top == False:
            self.resnet50 = tk.applications.resnet.ResNet50(include_top=False, weights="imagenet",input_shape=self.input_shape)
            
            if freezeWeights == True:
                for layer in self.resnet50.layers:
                    layer.trainable = False
            
            x = tk.layers.GlobalAveragePooling2D()(self.resnet50.output)
            x = tk.layers.Dense(2048)(x)
            x = tk.layers.Dropout(0.5)(x)
            x = tk.layers.Dense(2, activation="softmax")(x)
            self.resnet50 = tk.models.Model(inputs=self.resnet50.input, outputs=x)
        else:
            self.resnet50 = tk.applications.resnet.ResNet50(include_top=False, weights="imagenet",input_shape=self.input_shape)
            
            if freezeWeights == True:
                for layer in self.resnet50.layers:
                    layer.trainable = False
        

        return self.resnet50

    
    # 怎寫成屬性？ 類似model._compile()
    def _compile(self, model):

        resnetAdam = tk.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(
            optimizer=resnetAdam,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model


class DenseNet():

    def __init__(self, learning_rate=None):

        self.input_shape = (224, 224, 3)
        
        if learning_rate == None:
            self.learning_rate = 0.001
        else:
            self.learning_rate = learning_rate

    def denseNet121(self, include_top=False, weights="imagenet", freezeWeights=False):
        
        if weights != "imagenet":
            weights = None

        if include_top == False:
            self.densenet121 = tk.applications.densenet.DenseNet121(include_top=False, weights="imagenet", input_shape=self.input_shape)
            
            if freezeWeights == True:
                for layer in self.densenet121.layers:
                    layer.trainable = False

            x = tk.layers.GlobalAveragePooling2D()(self.densenet121.output)
            x = tk.layers.Dense(2048)(x)
            x = tk.layers.Dropout(0.5)(x)
            x = tk.layers.Dense(2, activation="softmax")(x)
            self.densenet121 = tk.models.Model(inputs=self.densenet121.input, outputs=x)
        else:
            self.densenet121 = tk.applications.densenet.DenseNet121(include_top=False, weights="imagenet", input_shape=self.input_shape)  
        
            if freezeWeights == True:
                for layer in self.densenet121.layers:
                    layer.trainable = False

        self.model = self.densenet121

        return self.densenet121


    # 怎寫成屬性？ 類似model._compile()
    def _compile(self, model):

        densenetAdam = tk.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(
            optimizer=densenetAdam,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        return model

class MobileNet():

    def __init__(self, learning_rate=None):
        
        self.input_shape = (224, 224, 3)

        if learning_rate == None:
            self.learning_rate = 0.001
        else:
            self.learning_rate = learning_rate
        
    def mobileNet(self, include_top=False, weights="imagenet", freezeWeights=False):

        if weights != "imagenet":
            weights = None

        if include_top == False:
            self.mobilenet = tk.applications.mobilenet.MobileNet(include_top=include_top, weights=weights, input_shape=self.input_shape)
            
            if freezeWeights == True:
                for layer in self.mobilenet.layers:
                    layer.trainable = False

            x = tk.layers.GlobalAveragePooling2D()(self.mobilenet.output)
            x = tk.layers.Dense(2048)(x)
            x = tk.layers.Dropout(0.5)(x)
            x = tk.layers.Dense(2, activation="softmax")(x)
            self.mobilenet = tk.models.Model(inputs=self.mobilenet.input, outputs=x)
        else:
            self.mobilenet = tk.applications.mobilenet.MobileNet(include_top=include_top, weights=weights, input_shape=self.input_shape)
            
            if freezeWeights == True:
                for layer in self.mobilenet.layers:
                    layer.trainable = False

        return self.mobilenet


    # 怎寫成屬性？ 類似model._compile()
    def _compile(self, model):

        mobilenetAdam = tk.optimizers.Adam(learning_rate=self.learning_rate)

        model.compile(
            optimizer=mobilenetAdam,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

        return model

