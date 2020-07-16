import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=DeprecationWarning)
import tensorflow as tf
import numpy as np
np.random.bit_generator = np.random._bit_generator
import matplotlib.pyplot as plt
import glob
import os
from deepposekit.io import TrainingGenerator, DataGenerator
from deepposekit.augment import FlipAxis
import imgaug.augmenters as iaa
import imgaug as ia
from deepposekit.models import (StackedDenseNet, DeepLabCut, StackedHourglass, LEAP)
from deepposekit.models import load_model
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from deepposekit.callbacks import Logger, ModelCheckpoint
from configparser import ConfigParser


def trainDPKmodel(dpkini,saveModelName,annotationP):
    config = ConfigParser()
    configFile = str(dpkini)
    config.read(configFile)
    project_folder = config.get('general DPK settings', 'project_folder')
    model_save_name = saveModelName
    model_save_name = model_save_name + '.h5'
    model_output_path = os.path.join(project_folder, 'models', model_save_name)
    annotationsPath = annotationP

    epochs = config.getint('train model settings', 'epochs')
    downsampleFactor = config.getint('train model settings', 'downsampleFactor')
    validation_split = config.getfloat('train model settings', 'validation_split')
    sigma = config.getint('train model settings', 'sigma')
    graph_scale = config.getint('train model settings', 'graph_scale')
    augmenterCheck = config.getboolean('train model settings', 'augmenterCheck')
    validation_batch_size = config.getint('train model settings', 'validation_batch_size')
    modelGrowthRate = config.getint('train model settings', 'modelGrowthRate')
    model_batch_size = config.getint('train model settings', 'model_batch_size')

    loggerCheck = config.getboolean('train model settings', 'loggerCheck')
    logger_validation_batch_size = config.getint('train model settings', 'logger_validation_batch_size')
    reducelrCheck = config.getboolean('train model settings', 'reducelrCheck')
    reduce_lr_factor = config.getfloat('train model settings', 'reduce_lr_factor')
    earlyStopCheck = config.getboolean('train model settings', 'earlyStopCheck')
    modelcheckPointCheck = config.getboolean('train model settings', 'modelcheckPointCheck')

    data_generator = DataGenerator(annotationsPath)
    NN_architecture = config.get('train model settings', 'NN_architecture')

    def augmentFunction(data_generator):
        augmenter = []
        augmenter.append(FlipAxis(data_generator, axis=0))
        augmenter.append(FlipAxis(data_generator, axis=1))
        sometimes = []
        sometimes.append(iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                                    translate_percent={'x': (-0.05, 0.05), 'y': (-0.05, 0.05)}, shear=(-8, 8), order=ia.ALL,
                                    cval=ia.ALL, mode=ia.ALL))
        sometimes.append(iaa.Affine(scale=(0.8, 1.2), mode=ia.ALL, order=ia.ALL, cval=ia.ALL))
        augmenter.append(iaa.Sometimes(0.75, sometimes))
        augmenter.append(iaa.Affine(rotate=(-180, 180), mode=ia.ALL, order=ia.ALL, cval=ia.ALL))
        augmenter = iaa.Sequential(augmenter)
        return augmenter

    if augmenterCheck == True:
        augmenter = augmentFunction(data_generator)
    if augmenterCheck == False:
        augmenter = None

    train_generator = TrainingGenerator(generator=data_generator, downsample_factor=downsampleFactor , augmenter=augmenter, sigma=sigma, validation_split=validation_split, use_graph=True, random_seed=1, graph_scale=graph_scale)
    train_generator.get_config()
    if NN_architecture == 'StackedDenseNet' or 'StackedHourglass':
        n_stacks = config.getint('StackedDenseNet/StackedHourglass settings', 'n_stacks')
        n_transitions = config.getint('StackedDenseNet/StackedHourglass settings', 'n_transitions')
        growth_rate = config.getint('StackedDenseNet/StackedHourglass settings', 'growth_rate')
        compression_factor = config.getfloat('StackedDenseNet/StackedHourglass settings', 'compression_factor')
        bottleneckfactor = config.getfloat('StackedDenseNet/StackedHourglass settings', 'bottleneckfactor')
        pretrained = config.get('StackedDenseNet/StackedHourglass settings', 'pretrained')
        subpixel = config.get('StackedDenseNet/StackedHourglass settings', 'subpixel')
        if NN_architecture == 'StackedDenseNet':
            model = StackedDenseNet(train_generator, n_stacks=n_stacks, bottleneck_factor=bottleneckfactor, n_transitions=n_transitions, growth_rate=modelGrowthRate, compression_factor=compression_factor, subpixel=subpixel, pretrained=True)
        if NN_architecture == 'StackedHourglass':
            model = StackedHourglass(train_generator, n_stacks=n_stacks, n_transitions=n_transitions, growth_rate=modelGrowthRate, compression_factor=compression_factor, subpixel=subpixel, pretrained=True)
    if NN_architecture == 'DeepLabCut':
        subpixel = config.getboolean('DeepLabCut settings', 'subpixel')
        weights = config.get('DeepLabCut settings', 'weights')
        backbone = config.get('DeepLabCut settings', 'backbone')
        alpha = config.getfloat('DeepLabCut settings', 'alpha')
        model = DeepLabCut(train_generator, weights=weights, subpixel=subpixel, backbone=backbone, alpha=alpha, pretrained=True)
    if NN_architecture == 'LEAP':
        filters = config.getint('LEAP settings', 'filters')
        upsampling_layers = config.getboolean('LEAP settings', 'upsampling_layers')
        batchnorm = config.getboolean('LEAP settings', 'batchnorm')
        pooling = config.get('LEAP settings', 'pooling')
        interpolation = config.get('LEAP settings', 'interpolation')
        subpixel = config.getboolean('LEAP settings', 'subpixel')
        initializer = config.get('LEAP settings', 'initializer')
        if downsampleFactor != 0:
            train_generator = TrainingGenerator(generator=data_generator, downsample_factor=downsampleFactor , augmenter=augmenter, sigma=sigma, validation_split=validation_split, use_graph=True, random_seed=1, graph_scale=graph_scale)
            train_generator.get_config()
        train_generator.get_config()
        model = LEAP(train_generator, filters=filters, upsampling_layers=upsampling_layers, batchnorm=batchnorm, pooling=pooling, interpolation=interpolation, subpixel=subpixel, initializer=initializer)
    callbacks = []
    if earlyStopCheck == True:
        early_stop = EarlyStopping(monitor="val_loss", min_delta=0.001, patience=100, verbose=1)
        callbacks.append(early_stop)
    if reducelrCheck  == True:
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=reduce_lr_factor, verbose=1, patience=20)
        callbacks.append(reduce_lr)
    if modelcheckPointCheck == True:
        model_checkpoint = ModelCheckpoint(model_output_path, monitor="val_loss", verbose=1, save_best_only=True)
        callbacks.append(model_checkpoint)
    if loggerCheck == True:
        logger = Logger(validation_batch_size=logger_validation_batch_size)
        callbacks.append(model_checkpoint)
    model.fit(batch_size=model_batch_size, validation_batch_size=validation_batch_size, callbacks=callbacks, epochs=epochs, n_workers=-1, steps_per_epoch=None)