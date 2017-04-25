import numpy as np
from sklearn import metrics as mets


def f1_score(y_true, y_pred):
    print(mets.f1_score(y_true, y_pred, average=None))

def confusion_matrix(y_true, y_pred):
    data = zip(y_true, y_pred)   # Merge the data
    tp = fp = tn = fn = 0.0      # Initialize value

    # Calculate the metrics
    for record in data:
        if record == (-1, -1):
            tn += 1
        elif record == (1, 1):
            tp += 1
        elif record == (1, -1):
            fn += 1
        else:
            fp += 1

    recall = tp / (tp + fn)
    precision = fp / (fp + tp)
    f1 = 2 * (precision * recall) / (precision + recall)
    accuracy = (tp + tn) * 100.0 / (tp + tn + fp + fn)
    
    # Display data
    print("Confusion matrix:\n")
    print("True positive:", tp)
    print("True negative:", tn)
    print("False positive:", fp)
    print("False negative:", fn)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)


# def print_positive_accuracy(y_true, y_pred):
#     assert()
#     len = len(y_true)
#     count = 0
#     for i in xrange(len):
#         if y_true



def main():
    # Demo values
    y_true = [-1, 1, 1, -1, 1, -1]
    y_pred = [-1, 1, 1, -1, -1, 1]

    f1_score(y_true, y_pred)
    confusion_matrix(y_true, y_pred)

if __name__ == '__main__':
    import sys
    sys.exit(int(main() or 0))
