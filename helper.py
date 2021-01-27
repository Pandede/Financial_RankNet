import torch


def calc_accuracy(prediction, label, mode):
    """
    :param prediction: the predicted value of model
    :param label: the ground truth
    :param mode: the type of ground truth (label), 0 for binary, 1 for multi-class, 2 for multi-label
    :return: the mean of accuracy
    """
    if mode in [0, 2, 'binary', 'multilabel']:
        accuracy = (torch.sigmoid(prediction).round() == label)
    elif mode in [1, 'multiclass']:
        accuracy = (prediction.argmax(1) == label)
    else:
        raise ValueError('Invalid mode, expected 0 or binary, 1 or multi-class, 2 or multi-label')
    return torch.mean(accuracy.float())
