import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class Metrics(object):
    def __init__(self):
        pass

    def get_scores(self, predicted, true):
        """ Compute the accuracies, precision, recall and F1 score for a batch of predictions and answers """
        predicted = torch.argmax(predicted, dim=-1).tolist()
        true = torch.argmax(true, dim=-1).tolist()
        
        acc = accuracy_score(true, predicted)
        pre = precision_score(true, predicted)
        recall = recall_score(true, predicted)
        f1 = f1_score(true, predicted)

        return {
            "accuracy": acc,
            "precision": pre,
            "recall": recall,
            "F1": f1
        }