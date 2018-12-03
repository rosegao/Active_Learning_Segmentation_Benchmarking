from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from Dataset import train_test_split, split_dataset, SegmentationDataset
from ActiveLearning import get_uncertain_samples_modified
import numpy as np
from keras.models import *
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.backend


def initialize_model(model, X_initial, y_initial, X_test, y_test,
                     n_classes, epochs, batch_size, verbose):

    model.fit(X_initial, y_initial, validation_data=(X_test, y_test),
              shuffle=True, batch_size=2, epochs=epochs, verbose=verbose)

    scores = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)
    print('Initial Test Loss: ', scores[0], ' Initial Test Accuracy: ', scores[1])
    return model


def run_ceal_modified(X_train, X_test, y_train, y_test,
    model, maximum_iterations, cost_effective, verbose,
    uncertain_samples_size, uncertain_criteria,
    delta, threshold_decay, fine_tuning_interval,
    epochs, batch_size):

    X_pool, y_pool, X_initial, y_initial, X_test, y_test = split_dataset(X_train, X_test, y_train, y_test, 0.1)

    model = initialize_model(model, X_initial, y_initial, X_test, y_test,
                             n_classes, epochs, batch_size, verbose)

    w, h, c = X_pool[-1,].shape

    # unlabeled samples
    DU = X_pool, y_pool

    # initially labeled samples
    DL = X_initial, y_initial

    # high confidence samples
    DH = np.empty((0, w, h, c)), np.empty((0, n_classes))

    for i in range(maximum_iterations):

        y_pred_prob = model.predict(DU[0], verbose=verbose)

        _, un_idx = get_uncertain_samples_modified(
            y_pred_prob, uncertain_samples_size,
            criteria=uncertain_criteria,
            labeled = DL[0], unlabeled = DU[0])

        DL = np.append(DL[0], np.take(DU[0], un_idx, axis=0), axis=0), \
             np.append(DL[1], np.take(DU[1], un_idx, axis=0), axis=0)

        if cost_effective:
            hc_idx, hc_labels = get_high_confidence_samples(y_pred_prob, delta)
            # remove samples also selected through uncertain
            hc = np.array([[i, l] for i, l in zip(hc_idx, hc_labels) if i not in un_idx])
            if hc.size != 0:
                DH = np.take(DU[0], hc[:, 0], axis=0), np_utils.to_categorical(hc[:, 1], n_classes)

        if i % fine_tuning_interval == 0:
            dtrain_x = np.concatenate((DL[0], DH[0])) if DH[0].size != 0 else DL[0]
            dtrain_y = np.concatenate((DL[1], DH[1])) if DH[1].size != 0 else DL[1]

            model.fit(dtrain_x, dtrain_y, validation_data=(X_test, y_test), batch_size=batch_size,
                      shuffle=True, epochs=epochs, verbose=verbose)
            delta -= (threshold_decay * fine_tuning_interval)

        DU = np.delete(DU[0], un_idx, axis=0), np.delete(DU[1], un_idx, axis=0)
        DH = np.empty((0, w, h, c)), np.empty((0, n_classes))

        _, acc = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=verbose)

        print(
            'Iteration: %d; High Confidence Samples: %d; Uncertain Samples: %d; Delta: %.5f; Labeled Dataset Size: %d; Accuracy: %.2f'
            % (i, len(DH[0]), len(DL[0]), delta, len(DL[0]), acc))
