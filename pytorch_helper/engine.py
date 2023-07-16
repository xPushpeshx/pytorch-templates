import torch
from torch import nn
from tqdm.auto import tqdm
from .metrics import *
from .utils import *

def train_step(model,
                dataloader,
                class_loss,
                seg_loss,
                optimizer,
                device):
    
    model.train()
    train_loss=torch.tensor([]).to(device)
    accuracy=torch.tensor([]).to(device)

    for batch , (X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)
        X=X.float()
        y=y.float()
        y_pred=model(X)
        #y_pred=y_pred.squeeze(1)
        loss_c=class_loss(y_pred,y)
        #loss_s=seg_loss(y_pred,y).reshape(1)
        train_loss=torch.cat((train_loss,loss_c.reshape(1)))

        optimizer.zero_grad()

        loss_c.backward()

        optimizer.step()
        
        accuracy=torch.cat((accuracy,get_accuracy(y_pred,y)))

    return train_loss.mean.item()  ,accuracy.mean().item()

def test_step(
        model,
        dataloader,
        class_loss,
        seg_loss,
        device,
):
    acc=torch.tensor([]).to(device)
    precision=torch.tensor([]).to(device)
    recall=torch.tensor([]).to(device)
    fbeta=torch.tensor([]).to(device)
    support=torch.tensor([]).to(device)
    iou=torch.tensor([]).to(device)
    dice=torch.tensor([]).to(device)
    auc_roc=torch.tensor([]).to(device)
    loss_com=torch.tensor([]).to(device)

    model.eval()

    with torch.inference_mode():
        for btach, (X,y) in enumerate(dataloader):
            X,y=X.to(device),y.to(device)
            X=X.float()
            y=y.float()
            outputs=model(X)
            loss=class_loss(outputs,y)
            loss_com =torch.cat((loss_com,loss.reshape(1)))
            #outputs = torch.where(outputs > 0.5, torch.tensor([1.0]).to(device), torch.tensor([0.0]).to(device))
            
            precision_,recall_,fbeta_,support_=get_clasification_metrics(outputs,y)
            acc=torch.cat((acc,get_accuracy(outputs,y)))
            auc_roc_=get_auc_roc(outputs,y).to(device)
            precision=torch.cat((precision,precision_))
            recall=torch.cat((recall,recall_))
            fbeta=torch.cat((fbeta,fbeta_))
            iou_=get_IOU(outputs,y).to(device)
            dice_=get_dice(outputs,y).to(device)
            iou=torch.cat((iou,iou_))
            dice=torch.cat((dice,dice_))
            auc_roc=torch.cat((auc_roc,auc_roc_))

    return loss_com.mean().item() ,acc.mean().item(),precision.mean().item(),recall.mean().item(),fbeta.mean().item(),iou.mean().item(),dice.mean().item(),auc_roc.mean().item()


def train_model(
        model,
        train_dataloader,
        test_dataloader,
        class_loss,
        seg_loss,
        optimizer,
        device,
        epochs,
        writer=None
):
    results={
        'train_loss':[],
        'train_acc':[],
        'test_loss':[],
        'test_acc':[],
        'test_precision':[],
        'test_recall':[],
        'test_fbeta':[],
        'test_support':[],
        'AUC_ROC':[],
        'IOU':[],
        'DICE':[],

    }

    model.to(device)
    max_acc=0
    max_dice=0
    for epoch in tqdm(range(epochs)):
        train_loss,train_acc=train_step(
            model,
            train_dataloader,
            class_loss,
            seg_loss,
            optimizer,
            device,
        )

        test_loss,test_acc,precision,recall,fbeta,iou,dice,auc_roc=test_step(
            model,
            test_dataloader,
            class_loss,
            seg_loss,
            device,
        )


        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        results['test_precision'].append(precision)
        results['test_recall'].append(recall)
        results['test_fbeta'].append(fbeta)
        results['iou'].append(iou)
        results['dice'].append(dice)
        results['AUC_ROC'].append(auc_roc)
      


        print(f'Epoch :{epoch+1} | train_loss:{train_loss:.4f} | train_acc:{train_acc:.4f} | test_loss:{test_loss:.4f} | test_acc:{test_acc:.4f} | test_precision:{precision:.4f} | test_recall:{recall:.4f} | test_f1Score:{fbeta:.4f} | AUC_ROC:{auc_roc:.4f} | iou:{iou:.4f} | dice:{dice:.4f}')
        if test_acc>max_acc and dice>max_dice:
            max_acc=test_acc
            max_dice=dice
            print("result improved--> saving model")
            save_model(model,'../models/resnet','Unet+_res50.pth',class_loss,seg_loss,optimizer,EPOCH=epoch+1)

        if writer is not None:
            writer.add_scalar(main_tag="Loss",
                            tag_scalar_dict={
                                "train":train_loss,
                                "test":test_loss,
                            },global_step=epoch)
            writer.add_scalar(main_tag="Accuracy",
                                tag_scalar_dict={
                                    "train":train_acc,
                                    "test":test_acc,
                                },global_step=epoch)
            writer.add_scalar(main_tag='other_metrics',
                            tag_scalar_dict={
                                "test_precision":precision,
                                "test_recall":recall,
                                "test_f1Score":fbeta,
                            },global_step=epoch)
            writer.close()
    
    return results