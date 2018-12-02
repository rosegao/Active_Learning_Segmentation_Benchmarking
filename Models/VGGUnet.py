from keras.models import *
from keras.layers import *
from keras.applications import VGG16


def VGGUnet(n_classes, input_height, input_width, image_ordering='channels_first'):

    pretrained = VGG16(weights='imagenet')
	weights = pretrained.get_weights()
	img_input = Input(shape=(3, input_height, input_width))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=image_ordering)(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=image_ordering)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=image_ordering)(x)
    f1 = x

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=image_ordering)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=image_ordering)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=image_ordering)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', data_format=image_ordering)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', data_format=image_ordering)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', data_format=image_ordering)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', data_format=image_ordering)(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', data_format=image_ordering)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', data_format=image_ordering)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', data_format=image_ordering)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', data_format=image_ordering)(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', data_format=image_ordering)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', data_format=image_ordering)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', data_format=image_ordering)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool', data_format=image_ordering)(x)

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    vgg = Model(img_input, x)
    vgg.set_weights(weights)

    o = f4

    o = (ZeroPadding2D((1, 1), data_format=image_ordering))(o)
    o = (Conv2D(512, (3, 3), padding='valid', data_format=image_ordering))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=image_ordering))(o)
    o = (concatenate([o, f3], axis=1))
    o = (ZeroPadding2D((1, 1), data_format=image_ordering))(o)
    o = (Conv2D(256, (3, 3), padding='valid', data_format=image_ordering))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=image_ordering))(o)
    o = (concatenate([o, f2], axis=1))
    o = (ZeroPadding2D((1, 1), data_format=image_ordering))(o)
    o = (Conv2D(128, (3, 3), padding='valid', data_format=image_ordering))(o)
    o = (BatchNormalization())(o)

    o = (UpSampling2D((2, 2), data_format=image_ordering))(o)
    o = (concatenate([o, f1], axis=1))
    o = (ZeroPadding2D((1, 1), data_format=image_ordering))(o)
    o = (Conv2D(64, (3, 3), padding='valid', data_format=image_ordering))(o)
    o = (BatchNormalization())(o)

    o = Conv2D(n_classes, (3, 3), padding='same', data_format=image_ordering)(o)
    o_shape = Model(img_input, o).output_shape
    outputHeight = o_shape[2]
    outputWidth = o_shape[3]

    o = (Reshape((n_classes, outputHeight * outputWidth)))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model
