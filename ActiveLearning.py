import numpy as np

def get_uncertain_samples_modified(y_pred_prob, n_samples, criteria, labeled, unlabeled):
    if criteria == 'lc':
        return least_confidence(y_pred_prob, n_samples)
    elif criteria == 'ms':
        return margin_sampling(y_pred_prob, n_samples)
    elif criteria == 'en':
        return entropy(y_pred_prob, n_samples)
    elif criteria == 'rs':
        return None, random_sampling(y_pred_prob, n_samples)
    elif criteria == 'gk':
        return None, greedy_k(labeled, unlabeled, n_samples)
    else:
        raise ValueError(
            'Unknown criteria value \'%s\', use one of [\'rs\',\'lc\',\'ms\',\'en\']' % criteria)


#--------------------------------------------------------------------------------
# Active Learning Sampling Methods
#--------------------------------------------------------------------------------


def random_sampling(y_pred_prob, n_samples):
    '''
    Random sampling
    '''
    return np.random.choice(range(len(y_pred_prob)), n_samples)


def least_confidence(y_pred_prob, n_samples):
    '''
    Rank all the unlabeled samples in an ascending order according to the least confidence
    '''
    origin_index = np.arange(0, len(y_pred_prob))
    max_prob = np.max(y_pred_prob, axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)

    lci = np.column_stack((origin_index,
                           max_prob,
                           pred_label))
    lci = lci[lci[:, 1].argsort()]
    return lci[:n_samples], lci[:, 0].astype(int)[:n_samples]


def margin_sampling(y_pred_prob, n_samples):
    '''
    Rank all the unlabeled samples in an ascending order according to the margin sampling
    '''

    origin_index = np.arange(0, len(y_pred_prob))
    margim_sampling = np.diff(-np.sort(y_pred_prob)[:, ::-1][:, :2])
    pred_label = np.argmax(y_pred_prob, axis=1)
    msi = np.column_stack((origin_index,
                           margim_sampling,
                           pred_label))
    msi = msi[msi[:, 1].argsort()]
    return msi[:n_samples], msi[:, 0].astype(int)[:n_samples]

def greedy_k(labeled, unlabeled, n_samples):
    '''
    Adapted from https://github.com/dsgissin/DiscriminativeActiveLearning/blob/master/query_methods.py
    '''

    flattened_labeled = np.array([i.flatten() for i in labeled])
    flattened_unlabeled = np.array([i.flatten() for i in unlabeled])

    greedy_indices = []

    # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
    min_dist = np.min(distance_matrix(flattened_labeled, flattened_unlabeled), axis=0)
    min_dist = min_dist.reshape((1, min_dist.shape[0]))
    for j in range(1, labeled.shape[0], 100):
        if j + 100 < labeled.shape[0]:
            dist = distance_matrix(flattened_labeled[j:j+100, :], flattened_unlabeled)
        else:
            dist = distance_matrix(flattened_labeled[j:, :], flattened_unlabeled)
        min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
        min_dist = np.min(min_dist, axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))

    # iteratively insert the farthest index and recalculate the minimum distances
    farthest = np.argmax(min_dist)
    greedy_indices.append(farthest)
    for i in range(n_samples-1):
        dist = distance_matrix(flattened_unlabeled[greedy_indices[-1], :].reshape((1, flattened_unlabeled.shape[1])), flattened_unlabeled)
        min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
        min_dist = np.min(min_dist, axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)

    indices = np.array(greedy_indices)
    return indices


def entropy(y_pred_prob, n_samples):
    '''
    Rank all the unlabeled samples in an descending order according to their entropy
    '''
    origin_index = np.arange(0, len(y_pred_prob))
    entropy = -np.nansum(np.multiply(y_pred_prob, np.log(y_pred_prob)), axis=1)
    pred_label = np.argmax(y_pred_prob, axis=1)
    eni = np.column_stack((origin_index,
                           entropy,
                           pred_label))

    eni = eni[(-eni[:, 1]).argsort()]
    return eni[:n_samples], eni[:, 0].astype(int)[:n_samples]

def get_high_confidence_samples(y_pred_prob, delta):
    eni, eni_idx = entropy(y_pred_prob, len(y_pred_prob))
    hcs = eni[eni[:, 1] < delta]
    return hcs[:, 0].astype(int), hcs[:, 2].astype(int)
