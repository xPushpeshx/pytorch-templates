from sklearn.metrics import *

def get_metrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    auc=roc_auc_score(y_test, y_pred, average='weighted')

    return acc, f1, precision, recall ,auc