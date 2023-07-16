from catalyst import metrics
import torch
from torcheval.metrics import MulticlassAUROC

def get_IOU(outputs,targets):
    metric = metrics.IOUMetric()
    metric.reset()
    metric.update_key_value(outputs, targets)
    metric.compute_key_value()
    return metric.compute_key_value()['iou'].reshape(1)

def get_dice(outputs,targets):
    metric = metrics.DiceMetric()
    metric.reset()
    metric.update_key_value(outputs, targets)
    metric.compute_key_value()
    return metric.compute_key_value()['dice'].reshape(1)

def get_tresky(outputs,targets,threshold=0.5):
    metric = metrics.trevsky(outputs, targets,alpha=0.2,threshold=threshold)
    return metric

def get_accuracy(outputs,targets):
    metric = metrics.accuracy(outputs,targets)
    return metric[0]

def get_auc_roc(outputs,targets):
    metric=MulticlassAUROC(num_classes=3)
    metric.update(outputs,targets)
    return torch.tensor([metric.compute()])

def get_clasification_metrics(outputs,targets):
    metric = metrics.precision_recall_fbeta_support(outputs,targets)
    precision=metric[0]
    recall=metric[1]
    fbeta=metric[2]
    support=metric[3]
    return precision,recall,fbeta,support
