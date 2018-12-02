from keras.models import *
from keras.layers import *
from keras.applications import VGG16


# crop o1 with respect to o2
def crop(o1, o2, i, image_ordering='channels_first'):
    o_shape2 = Model(i, o2).output_shape
    outputHeight2 = o_shape2[2]
    outputWidth2 = o_shape2[3]

    o_shape1 = Model(i, o1).output_shape
    outputHeight1 = o_shape1[2]
    outputWidth1 = o_shape1[3]

    cx = abs(outputWidth1 - outputWidth2)
    cy = abs(outputHeight2 - outputHeight1)

    if outputWidth1 > outputWidth2:
        o1 = Cropping2D(cropping=((0, 0), (0, cx)), data_format=image_ordering)(o1)
    else:
        o2 = Cropping2D(cropping=((0, 0), (0, cx)), data_format=image_ordering)(o2)

    if outputHeight1 > outputHeight2:
        o1 = Cropping2D(cropping=((0, cy), (0, 0)), data_format=image_ordering)(o1)
    else:
        o2 = Cropping2D(cropping=((0, cy), (0, 0)), data_format=image_ordering)(o2)

    return o1, o2


def FCN8(nClasses, input_height, input_width, image_ordering='channels_first'):

	pretrained = VGG16(weights='imagenet')
	weights = pretrained.get_weights()

    img_input = Input(shape=(3, input_height, input_width))

    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', data_format=image_ordering)(
        img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', data_format=image_ordering)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool', data_format=image_ordering)(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', data_format=image_ordering)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', data_format=image_ordering)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool', data_format=image_ordering)(x)

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
    f5 = x

    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(1000, activation='softmax', name='predictions')(x)

    vgg = Model(img_input, x)
    vgg.set_weights(weights)

    o = f5
    o = (Conv2D(4096, (7, 7), activation='relu', padding='same', data_format=image_ordering))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(4096, (1, 1), activation='relu', padding='same', data_format=image_ordering))(o)
    o = Dropout(0.5)(o)
    o = (Conv2D(nClasses, (1, 1), kernel_initializer='he_normal', data_format=image_ordering))(o)
    o = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(2, 2), use_bias=False, data_format=image_ordering)(o)
    o2 = f4
    o2 = (Conv2D(nClasses, (1, 1), kernel_initializer='he_normal', data_format=image_ordering))(o2)

    o, o2 = crop(o, o2, img_input)

    o = Add()([o, o2])
    o = Conv2DTranspose(nClasses, kernel_size=(4, 4), strides=(2, 2), use_bias=False, data_format=image_ordering)(o)
    o2 = f3
    o2 = (Conv2D(nClasses, (1, 1), kernel_initializer='he_normal', data_format=image_ordering))(o2)
    o2, o = crop(o2, o, img_input)
    o = Add()([o2, o])

    o = Conv2DTranspose(nClasses, kernel_size=(16, 16), strides=(8, 8), use_bias=False, data_format=image_ordering)(o)

    o_shape = Model(img_input, o).output_shape

    outputHeight = o_shape[2]
    outputWidth = o_shape[3]

    o = (Reshape((-1, outputHeight * outputWidth)))(o)
    o = (Permute((2, 1)))(o)
    o = (Activation('softmax'))(o)
    model = Model(img_input, o)
    model.outputWidth = outputWidth
    model.outputHeight = outputHeight

    return model
