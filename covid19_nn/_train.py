# /usr/bin/python3.6
# -*- coding:utf-8 -*-
# Author: HU REN BAO
# History:
#        1. first create on 20200214
#

import tensorflow.keras as tk
from _model import ResNet, DenseNet, MobileNet
from _preprocessing import load_data, split_data, genetator 



def train(models, train_generator, valid_generator, **kwargs):

    if not isinstance(models, list):
        raise TypeError("Type of models must be list.")

    if len(models) != 3:
        raise ValueError("Number of models must be three.")

    allow_kwargs = {
            "MIN_DELTA",
            "MONITOR",
            "PATIENCE",
            "VERBOSE",
            "HISTOGRAM_FREQ"
        }

    for kwarg in kwargs:
        if kwarg not in allow_kwargs:
            raise TypeError("Keyword argument not understood: ", kwarg)
    
    MIN_DELTA = kwargs.get("MIN_DELTA")
    if not MIN_DELTA:
        MIN_DELTA = 0.0005

    MONITOR = kwargs.get("MONITOR")
    if not MONITOR:
        MONITOR = "val_loss"

    PATIENCE = kwargs.get("PATIENCE")
    if not PATIENCE:
        PATIENCE = 10
    
    VERBOSE = kwargs.get("VERBOSE")
    if not VERBOSE:
        VERBOSE = 1

    HISTOGRAM_FREQ = kwargs.get("HISTOGRAM_FREQ")
    if not HISTOGRAM_FREQ:    
        HISTOGRAM_FREQ = 10


    resnet50, densenet121, mobilenet = models

    history_resnet50 = resnet50.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        verbose=VERBOSE,
        workers= 4,
        validation_data=valid_generator,
        validation_steps=100,
        callbacks=[
                tk.callbacks.EarlyStopping(
                    monitor=MONITOR,
                    min_delta=MIN_DELTA,
                    patience=PATIENCE,
                    verbose=VERBOSE
                ),
                tk.callbacks.ModelCheckpoint(
                    filepath="covid-19/model_resnet_best_weight.h5",
                    monitor=MONITOR,
                    verbose=VERBOSE,
                    save_weights_only=True
                ),
                tk.callbacks.TensorBoard(
                    log_dir="covid-19/resnet50_logs",
                    write_images=True,
                    histogram_freq=HISTOGRAM_FREQ,
                )
        ]
    )


    history_densenet121 = densenet121.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        verbose=VERBOSE,
        workers= 4,
        validation_data=valid_generator,
        validation_steps=100,
        callbacks=[
                tk.callbacks.EarlyStopping(
                    monitor=MONITOR,
                    min_delta=MIN_DELTA,
                    patience=PATIENCE,
                    verbose=VERBOSE
                ),
                tk.callbacks.ModelCheckpoint(
                    filepath="covid-19/model_densenet_best_weight.h5",
                    monitor=MONITOR,
                    verbose=VERBOSE,
                    save_weights_only=True
                ),
                tk.callbacks.TensorBoard(
                    log_dir="covid-19/densenet121_logs",
                    write_images=True, 
                    histogram_freq=HISTOGRAM_FREQ
                )
        ]
    )

    history_mobilenet = mobilenet.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=100,
        verbose=VERBOSE,
        workers= 4,
        validation_data=valid_generator,
        validation_steps=100,
        callbacks=[
                tk.callbacks.EarlyStopping(
                    monitor=MONITOR,
                    min_delta=MIN_DELTA,
                    patience=PATIENCE,
                    verbose=VERBOSE
                ),
                tk.callbacks.ModelCheckpoint(
                    filepath="covid-19/model_mobilenet_best_weight.h5",
                    monitor=MONITOR,
                    verbose=VERBOSE,
                    save_weights_only=True,
                    
                ),
                tk.callbacks.TensorBoard(
                    log_dir="covid-19/mobilenet_logs",
                    write_images=True,
                    histogram_freq=HISTOGRAM_FREQ
                )
        ]
    )

    print("============================")
    print("All Processing Completely...")
    print("============================")

    return (history_resnet50, history_densenet121, history_mobilenet)


if __name__ == "__main__":

    # building model
    resnet50 = ResNet().resNet50()
    densenet121 = DenseNet().denseNet121()
    mobilenet = MobileNet().mobileNet()

    resnet50 = ResNet()._compile(resnet50)
    densenet121 = DenseNet()._compile(densenet121)
    mobilenet = MobileNet()._compile(mobilenet)

    models = [resnet50, densenet121, mobilenet]

    # load data and split for validation
    df = load_data()
    data = split_data(df)
    
    # create generator
    train_generator, valid_generator, _ = genetator(data)
    
    # training
    train(models, train_generator, valid_generator)