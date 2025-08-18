import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score


def evaluate(predictions, labels):
    predictions = np.array(predictions) > 0.5
    accuracy = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)

    return accuracy, recall, f1