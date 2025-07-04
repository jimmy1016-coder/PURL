from operator import itemgetter

import numpy as np
from sklearn.preprocessing import normalize
from scipy.optimize import brentq
from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d


def cosine_similarity(x, y):
    """
    Returns a 1d array of cosine similarity
    x: shape (n_vecs, dim)
    y: shape (n_vecs, dim)
    """
    xn = normalize(x)
    yn = normalize(y)
    return (xn * yn).sum(1)


def compute_error_rates(scores, labels):
    """
    Taken from: https://github.com/JaesungHuh/VoxSRC2023/blob/main/compute_min_dcf.py
    """
    # Sort the scores from smallest to largest, and also get the corresponding
    # indexes of the sorted scores.  We will treat the sorted scores as the
    # thresholds at which the the error-rates are evaluated.
    sorted_indexes, thresholds = zip(
        *sorted(
            [(index, threshold) for index, threshold in enumerate(scores)],
            key=itemgetter(1),
        )
    )
    labels = [labels[i] for i in sorted_indexes]
    fnrs = []
    fprs = []

    # At the end of this loop, fnrs[i] is the number of errors made by
    # incorrectly rejecting scores less than thresholds[i]. And, fprs[i]
    # is the total number of times that we have correctly accepted scores
    # greater than thresholds[i].
    for i in range(0, len(labels)):
        if i == 0:
            fnrs.append(labels[i])
            fprs.append(1 - labels[i])
        else:
            fnrs.append(fnrs[i - 1] + labels[i])
            fprs.append(fprs[i - 1] + 1 - labels[i])
    fnrs_norm = sum(labels)
    fprs_norm = len(labels) - fnrs_norm

    # Now divide by the total number of false negative errors to
    # obtain the false positive rates across all thresholds
    fnrs = [x / float(fnrs_norm) for x in fnrs]

    # Divide by the total number of corret positives to get the
    # true positive rate.  Subtract these quantities from 1 to
    # get the false positive rates.
    fprs = [1 - x / float(fprs_norm) for x in fprs]
    return fnrs, fprs, thresholds


# Computes the minimum of the detection cost function.  The comments refer to
# equations in Section 3 of the NIST 2016 Speaker Recognition Evaluation Plan.
def compute_min_dcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa):
    """
    Taken from: https://github.com/JaesungHuh/VoxSRC2023/blob/main/compute_min_dcf.py
    """
    min_c_det = float("inf")
    min_c_det_threshold = thresholds[0]
    for i in range(0, len(fnrs)):
        # See Equation (2).  it is a weighted sum of false negative
        # and false positive errors.
        c_det = c_miss * fnrs[i] * p_target + c_fa * fprs[i] * (1 - p_target)
        if c_det < min_c_det:
            min_c_det = c_det
            min_c_det_threshold = thresholds[i]
    # See Equations (3) and (4).  Now we normalize the cost.
    c_def = min(c_miss * p_target, c_fa * (1 - p_target))
    min_dcf = min_c_det / c_def
    return min_dcf, min_c_det_threshold


def calculate_eer(ground_truth, prediction, pos_label=1):
    # ground_truth denotes groundtruth scores,
    # prediction denotes the prediction scores.
    fpr, tpr, thresholds = roc_curve(ground_truth, prediction, pos_label=pos_label)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)

    return eer, thresh


def calculate_min_dcf(ground_truth, prediction, c_miss=1, c_fa=1, p_target=0.05):
    fnrs, fprs, thresholds = compute_error_rates(prediction, ground_truth)
    mindcf, threshold = compute_min_dcf(fnrs, fprs, thresholds, p_target, c_miss, c_fa)

    return mindcf, threshold

def calculate_test_scores(emb_dict, files1, files2):
    embds_1 = np.vstack([emb_dict[p] for p in files1])
    embds_2 = np.vstack([emb_dict[p] for p in files2])

    scores = cosine_similarity(embds_1, embds_2)

    return scores

def calculate_eer_and_min_dcf(emb_dict, files1, files2, labels):
    scores = calculate_test_scores(emb_dict, files1, files2)

    eer, _ = calculate_eer(
            ground_truth=labels,
            prediction=scores,
        )

    min_dcf, _ = calculate_min_dcf(
        ground_truth=labels,
        prediction=scores,
    )

    return eer, min_dcf
