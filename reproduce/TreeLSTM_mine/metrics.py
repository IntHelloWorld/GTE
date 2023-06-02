from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# calculate metrics
def prec_recall_f1(predict_labels, expect_labels, average="binary"):
    precision, recall, f1, _ = precision_recall_fscore_support(expect_labels, predict_labels, average=average)
    return precision, recall, f1


def accuracy(predict_labels, expect_labels):
    acc = accuracy_score(expect_labels, predict_labels)
    return acc
