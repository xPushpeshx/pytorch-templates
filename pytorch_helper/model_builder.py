import torch
from torch import nn
from torchinfo import summary
from torchvision.models import *
import segmentation_models_pytorch as smp
# create model builder class

def get_model(num_classes=1):
    weights = EfficientNet_B0_Weights.DEFAULT # .DEFAULT = best available weights 
    model = efficientnet_b0(weights=weights)
    for param in model.features.parameters():
        param.requires_grad = True
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=num_classes, # same number of output units as our number of classes
                        bias=True))
    return model



def get_summary(model, IMG_SIZE):
    return summary(model, input_size=(1, 3, IMG_SIZE, IMG_SIZE), col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"], depth=4)