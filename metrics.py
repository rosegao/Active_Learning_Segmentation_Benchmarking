from keras import backend as K

SMOOTH_LOSS = 1e-12


def iou(y_true, y_pred):
    '''
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # References
    Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
    What is a good evaluation measure for semantic segmentation?.
    IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
    https://en.wikipedia.org/wiki/Jaccard_index
    '''

    y_true, y_pred = y_true[:,1:], y_pred[:,1:]  # remove background class for evaluation
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1) - intersection

    return (intersection + SMOOTH_LOSS) / (union + SMOOTH_LOSS)


def dice(y_true, y_pred):
    '''
    Dice = (2|X & Y|)/ (|X|+ |Y|)
            = 2(sum(|A*B|))/(sum(|A|)+sum(|B|)
    '''
    y_true, y_pred = y_true[:,1:], y_pred[:,1:]  # remove background class for evaluation
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        
    return (2 * intersection + SMOOTH_LOSS) / (sum_ + SMOOTH_LOSS)



def iou_flat(y_true, y_pred):
    '''
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # References
    Csurka, Gabriela & Larlus, Diane & Perronnin, Florent. (2013).
    What is a good evaluation measure for semantic segmentation?.
    IEEE Trans. Pattern Anal. Mach. Intell.. 26. . 10.5244/C.27.32.
    https://en.wikipedia.org/wiki/Jaccard_index
    '''

    y_true, y_pred = y_true[:,1:], y_pred[:,1:]  # remove background class for evaluation
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f) + K.sum(y_pred_f) - intersection
    return (intersection + SMOOTH_LOSS) / (union + SMOOTH_LOSS)


def dice_flat(y_true, y_pred):
    '''
    Dice = (2|X & Y|)/ (|X|+ |Y|)
            = 2(sum(|A*B|))/(sum(|A|)+sum(|B|)
    '''
    y_true, y_pred = y_true[:,1:], y_pred[:,1:]  # remove background class for evaluation
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + SMOOTH_LOSS) / (K.sum(y_true_f) + K.sum(y_pred_f) + SMOOTH_LOSS)


def dice_loss(y_true, y_pred):
    '''
    '''
    return 1-dice_flat(y_true, y_pred)
