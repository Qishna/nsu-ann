def binary_classification_metrics(true, predict):
    TP = ((predict == 1) & (true == 1)).sum()
    FP = ((predict == 1) & (true == 0)).sum()
    FN = ((predict == 0) & (true == 1)).sum()
    TN = ((predict == 0) & (true == 0)).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    accuracy = (predict == true).sum() / len(predict)
    f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1, accuracy


def multiclass_accuracy(true, predict):
    return (predict == true).sum() / len(predict)
