from __future__ import print_function, absolute_import, division
import argparse
import os
import numpy as np
import glob
import cv2
import random
import itertools
import shutil
import matplotlib.pyplot as plt
from collections import namedtuple
import time


#--------------------------------------------------------------------------------
# Definitions
#--------------------------------------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


#--------------------------------------------------------------------------------
# A list of all labels
#--------------------------------------------------------------------------------

# Please adapt the train IDs as appropriate for your approach.
# Note that you might want to ignore labels with ID 255 during training.
# Further note that the current train IDs are only a suggestion. You can use whatever you like.
# Make sure to provide your results using the original IDs and not the training IDs.
# Note that many IDs are ignored in evaluation and thus you never need to predict these!

labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , False        , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , False        , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , False        , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

#--------------------------------------------------------------------------------
# Create dictionaries for a fast lookup
#--------------------------------------------------------------------------------


def create_label_dicts():
    class2color, class2name = {}, {}
    for label in labels:
        k = label.id
        name = label.name
        if label.id < 5:
            k = 0
            name = 'unlabeled'
        else:
            k -= 4

        class2color[k] = label.color
        class2name[k] = name

    return class2color, class2name

#--------------------------------------------------------------------------------
# Prepare Images and Segmentations
#--------------------------------------------------------------------------------


def getSegmentationArr(path, nClasses, width, height, class2color):
    '''
    inputs are TARGET width, target height
    each class corresponds to an (r, g, b) value
    for each class pixels are labeled 1 wherever there is a segmentation for that class
    Returns:
        h*w x nClasses tensor
    '''
    seg_labels = np.zeros((height, width, nClasses))
    try:
        img = cv2.imread(path, 1)
        img = cv2.resize(img, (width, height))

        seg_labels = np.zeros((height, width, nClasses))

        for c in range(nClasses):
            r,g,b = class2color[c]
            seg_labels[:, :, c] = ((img[:, :, 0] == r) & (img[:, :, 1] == g) & (img[:, :, 2] == b)).astype(int)

    except Exception as e:
        print(e)

    seg_labels = np.reshape(seg_labels, (width * height, nClasses))
    return seg_labels


def getImageArr(path, width, height, imgNorm="sub_mean", ordering='channels_first'):
    try:
        img = cv2.imread(path, 1)

        if imgNorm == "sub_and_divide":
            img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
        elif imgNorm == "sub_mean":
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img[:, :, 0] -= 103.939
            img[:, :, 1] -= 116.779
            img[:, :, 2] -= 123.68
        elif imgNorm == "divide":
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float32)
            img = img / 255.0

        if ordering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img
    except Exception as e:
        print(path, e)
        img = np.zeros((height, width, 3))
        if ordering == 'channels_first':
            img = np.rollaxis(img, 2, 0)
        return img

def resize_img(path, target_width, target_height, target_path):
    '''
    '''
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (target_width, target_height))
    cv2.imwrite(target_path, img)

def DataGenerator(img_path seg_path, split='small_train',
                  n_classes=30,
                  input_width=224, input_height=224,
                  output_width=112, output_height=112,
                  imgNorm='sub_mean',
                  ordering='channels_first',
                  batch_size=16
                  ):

    img_dir = os.path.join(img_path, split)
    seg_dir = os.path.join(seg_path, split)

    images = os.listdir(img_dir)
    images.sort()
    segmentations = os.listdir(seg_dir)
    segmentations.sort()

    assert len(images) == len(segmentations)

    zipped = itertools.cycle(zip(images, segmentations))
    class2color, _ = create_label_dicts()

    while True:
        X, Y = [], []
        for _ in range(batch_size):
            img, seg = next(zipped)
            assert ('_'.join(img.split('_')[0:3]) == '_'.join(seg.split('_')[0:3]))
            img, seg = os.path.join(img_dir, img), os.path.join(seg_dir, seg)

            img = getImageArr(img, input_width, input_height, imgNorm, ordering)
            seg = getSegmentationArr(seg, n_classes, output_width, output_height, class2color)

            X.append(img)
            Y.append(seg)

        yield np.array(X), np.array(Y)

def SampleDataVisualizer(img_path=IMG_PATH, seg_path=SEG_PATH, split='TRAIN',
                        num_imgs=10):
    img_dir = os.path.join(img_path, split)
    seg_dir = os.path.join(seg_path, split)

    images = os.listdir(img_dir)
    images.sort()
    segmentations = os.listdir(seg_dir)
    segmentations.sort()

    assert len(images) == len(segmentations)
    zipped = itertools.cycle(zip(images, segmentations))
    for _ in range(num_imgs):
        img, seg = next(zipped)
        img, seg = os.path.join(img_dir, img), os.path.join(seg_dir, seg)
        img, seg = cv2.imread(img), cv2.imread(seg)
        fig, axarr = plt.subplots(1,2, figsize=(15,15))
        axarr[0].imshow(img)
        axarr[0].axis('off')
        axarr[1].imshow(seg)
        axarr[1].axis('off')


def SegmentationDataset(img_path, seg_path, split,
                  n_classes=30,
                  input_width=224, input_height=224,
                  output_width=112, output_height=112,
                  imgNorm='sub_mean',
                  ordering='channels_first'
                  ):

    img_dir = os.path.join(img_path, split)
    seg_dir = os.path.join(seg_path, split)
    print('Looking for images in {}'.format(img_dir))
    print('Looking for segmentations in {}'.format(seg_dir))

    images = os.listdir(img_dir)
    images.sort()
    segmentations = os.listdir(seg_dir)
    segmentations.sort()

    assert len(images) == len(segmentations)
    print("Number of images match, creating dataset.")

    class2color, _ = create_label_dicts()

    X, Y = [], []
    n = 0
    start = time.time()
    for img, seg in zip(images, segmentations):
        assert ('_'.join(img.split('_')[0:3]) == '_'.join(seg.split('_')[0:3]))
        img, seg = os.path.join(img_dir, img), os.path.join(seg_dir, seg)

        img = getImageArr(img, input_width, input_height, imgNorm, ordering)
        seg = getSegmentationArr(seg, n_classes, output_width, output_height, class2color)

        X.append(img)
        Y.append(seg)
        n += 1
        if n % 100 == 0 :
            elapsed = time.time() - start
            print('{} images processed, Time elapsed: {}'.format(n, elapsed))

    return np.array(X), np.array(Y)

#--------------------------------------------------------------------------------
# Active Learning Splits
#--------------------------------------------------------------------------------

def train_test_split(X, y, percent):
    X_train = []
    y_train = []
    X_test = list(X)
    y_test = list(y)

    train_size = int(len(X_test) * percent)

    while len(X_train) < train_size:
        index = random.randrange(len(X_test))
        X_train.append(X_test.pop(index))
        y_train.append(y_test.pop(index))

    return np.array(X_test), np.array(y_test), np.array(X_train), np.array(y_train)

def split_dataset(X_train, X_test, y_train, y_test, initial_annotated_perc=0.1):
    X_pool, y_pool, X_initial, y_initial = train_test_split(X_train,
                                                            y_train,
                                                            initial_annotated_perc)
    return X_pool, y_pool, X_initial, y_initial, X_test, y_test


#--------------------------------------------------------------------------------
# Un-nest some folders and select desired annotations from CityScapes raw data
#--------------------------------------------------------------------------------

def prep_annotations():
    root = './CityScapes Dataset/'
    img_root = 'leftImg8bit'
    annotation_root = 'gtFine'

    annotation_dir = os.path.join(root, annotation_root)
    splits = ['train', 'val', 'test']

    dest_path = './CityScapes Dataset/segmentations'
    os.makedirs(dest_path, exist_ok=True)

    for split in splits:
        print(split)
        ann_split_path = os.path.join(annotation_dir, split)
        dest_split_path = os.path.join(dest_path, split)
        os.makedirs(dest_split_path, exist_ok=True)

        cities = os.listdir(ann_split_path)
        for city in cities:
            if city != '.DS_Store':
                full_origin_dir = os.path.join(ann_split_path, city)
            for seg in os.listdir(full_origin_dir):
                if seg.endswith('color.png'):
                    orig_seg = os.path.join(full_origin_dir, seg)
                    dest_seg = os.path.join(dest_split_path, seg)
                    shutil.copy(orig_seg, dest_seg)
        n_imgs = len(os.listdir(dest_split_path))
        print("Number of images copied: {}".format(n_imgs))

def prep_imgs():
    root = './CityScapes Dataset/'
    img_root = 'leftImg8bit'
    annotation_root = 'gtFine'
    splits = ['train', 'val', 'test']

    for type in [img_root, annotation_root]:
        print(type)
        if type == 'leftImg8bit':
            dest_path = os.path.join(root, 'rawimgs')
        else:
            dest_path = os.path.join(root, 'segmentations')

        base_dir = os.path.join(root, type)
        for spl in splits:
            og_spl_path = os.path.join(base_dir, spl)
            dest_split_path = os.path.join(dest_path, spl)
            os.makedirs(dest_split_path, exist_ok=True)
            cities = os.listdir(og_spl_path)
            for city in cities:
                if city != '.DS_Store':
                    full_origin_dir = os.path.join(og_spl_path, city)
                    print('copying file from: ', full_origin_dir)
                for im in os.listdir(full_origin_dir):
                    if type == 'gtFine':
                        if im.endswith('color.png'):
                            oi = os.path.join(full_origin_dir, im)
                            di = os.path.join(dest_split_path, im)
                    else:
                        oi = os.path.join(full_origin_dir, im)
                        di = os.path.join(dest_split_path, im)
                    shutil.copy(oi, di)
            n_imgs = len(os.listdir(dest_split_path))
            print("Number of images copied: {}".format(n_imgs))


def subset(num_images, split):
    '''
    '''
    img_origin_path = './CityScapes Dataset/rawimgs/{}'.format(split)
    seg_origin_path = './CityScapes Dataset/segmentations/{}'.format(split)

    img_dest_path = './CityScapes Dataset/rawimgs/small_{}'.format(split)
    seg_dest_path = './CityScapes Dataset/segmentations/small_{}'.format(split)

    os.makedirs(img_dest_path, exist_ok=True)
    os.makedirs(seg_dest_path, exist_ok=True)

    images = os.listdir(img_origin_path)
    segmentations = os.listdir(seg_origin_path)

    assert (len(images) == len(segmentations))
    images.sort()
    segmentations.sort()
    target_width, target_height = 512, 256
    for i in range(num_images):
        iop = os.path.join(img_origin_path, images[i])
        sop = os.path.join(seg_origin_path, segmentations[i])

        idp = os.path.join(img_dest_path, images[i])
        sdp = os.path.join(seg_dest_path, segmentations[i])

        resize_img(iop, target_width, target_height, idp)
        resize_img(sop, target_width, target_height, sdp)

        #shutil.copy(iop, idp)
        #shutil.copy(idp, sdp)


    print('Num images: {}'.format(len(os.listdir(img_dest_path))))
    print('Num segmentations: {}'.format(len(os.listdir(seg_dest_path))))


def data_generator():
    root = './CityScapes Dataset/'
    img_root = 'leftImg8bit'
    annotation_root = 'segmentations'
    splits = ['train', 'val', 'test']
    img_dir = os.path.join(root, img_root)
    img_dirs = [os.path.join(img_dir, spl) for spl in splits]
    annotation_dir = os.path.join(root, annotation_root)
    annotation_dirs = [os.path.join(annotation_dir, spl) for spl in splits]
    for (img, ann) in zip(img_dirs, annotation_dirs):
        yield(img, ann)
