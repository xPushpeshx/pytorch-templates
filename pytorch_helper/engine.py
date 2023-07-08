import torch
from torch import nn
from tqdm.auto import tqdm
from .metrics import *

def train_step(model,
                dataloader,
                loss_fn,
                optimizer,
                device):
    
    model.train()
    train_loss,train_acc=0.0,0.0

    for batch , (X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)
        X=X.float()
        y=y.float()
        y_pred=model(X)
        y_pred=y_pred.squeeze(1)
        loss=loss_fn(y_pred,y)
        train_loss+=loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        y_pred=y_pred.unsqueeze(1)
        y_pred_class=torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        train_acc+=(y_pred_class==y).sum().item()/len(y_pred)

    train_loss=train_loss / len(dataloader)
    train_acc=train_acc / len(dataloader)

    return train_loss,train_acc

def test_step(
        model,
        dataloader,
        loss_fn,
        device,
):
    acc=torch.tensor([]).to(device)
    precision=torch.tensor([]).to(device)
    recall=torch.tensor([]).to(device)
    fbeta=torch.tensor([]).to(device)
    support=torch.tensor([]).to(device)

    model.eval()

    loss_com = 0.0
    with torch.inference_mode():
        for btach, (X,y) in enumerate(dataloader):
            X,y=X.to(device),y.to(device)
            X=X.float()
            outputs=model(X)
            loss=loss_fn(outputs,y)
            loss_com =loss_com+loss.item()
            outputs = torch.where(outputs > 0.5, torch.tensor([1.0]).to(device), torch.tensor([0.0]).to(device))
            
            precision_,recall_,fbeta_,support_=get_clasification_metrics(outputs,y)
            acc=torch.cat((acc,get_accuracy(outputs,y)))
            precision=torch.cat((precision,precision_))
            recall=torch.cat((recall,recall_))
            fbeta=torch.cat((fbeta,fbeta_))
            support=torch.cat((support,support_))

    return loss_com/len(dataloader),acc.mean().item(),precision.mean().item(),recall.mean().item(),fbeta.mean().item(),support.mean().item()


def train_model(
        model,
        train_dataloader,
        test_dataloader,
        loss_fn,
        optimizer,
        device,
        epochs,
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

    }

    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss,train_acc=train_step(
            model,
            train_dataloader,
            loss_fn,
            optimizer,
            device,
        )

        test_loss,test_acc,precision,recall,fbeta,suppport=test_step(
            model,
            test_dataloader,
            loss_fn,
            device,
        )


        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        results['test_precision'].append(precision)
        results['test_recall'].append(recall)
        results['test_fbeta'].append(fbeta)
        results['test_support'].append(suppport)
      


        print(f'Epoch :{epoch+1}| train_loss:{train_loss:.4f} | train_acc:{train_acc:.4f} | test_loss:{test_loss:.4f} | test_acc:{test_acc:.4f},test_precision:{precision:.4f} | test_recall:{recall:.4f} | test_f1Score:{fbeta:.4f}')

    
    return results