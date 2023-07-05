import torch
from torch import nn
from tqdm.auto import tqdm
from .metrics import get_metrics

def train_step(model,
                dataloader,
                loss_fn,
                optimizer,
                device):
    
    model.train()
    train_loss,train_acc=0.0,0.0

    for batch , (X,y) in enumerate(dataloader):
        X,y=X.to(device),y.to(device)

        y_pred=model(X)

        loss=loss_fn(y_pred,y)
        train_loss+=loss.item()

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

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
    y_true=[]
    y_pred=[]

    model.eval()
    test_loss,test_acc=0.0,0.0

    with torch.inference_model():
        for btach, (X,y) in enumerate(dataloader):
            X,y=X.to(device),y.to(device)

            y_pred=model(X)

            loss=loss_fn(y_pred,y)
            test_loss+=loss.item()

            test_pred_label=y_pred.argmax(dim=1)
            test_acc+=(test_pred_label==y).sum().item()/len(y_pred)

            y_true.extend(y.cpu().numpy())
            y_pred.extend(test_pred_label.cpu().numpy())
        
    test_loss=test_loss / len(dataloader)
    test_acc=test_acc / len(dataloader)

    more_mertics=get_metrics(y_true,y_pred)

    

    return test_loss,test_acc,more_mertics

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
        'accuracy_score':[],
        'test_loss':[],
        'test_acc':[],
        'test_precision':[],
        'test_recall':[],
        'test_f1':[],
        'test_auc':[],
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

        test_loss,test_acc,more_metric=test_step(
            model,
            test_dataloader,
            loss_fn,
            device,
        )

        acc,f1,precision,recall,auc=more_metric

        results['train_loss'].append(train_loss)
        results['train_acc'].append(train_acc)
        results['test_loss'].append(test_loss)
        results['test_acc'].append(test_acc)
        results['accuracy_score'].append(acc)
        results['test_f1'].append(f1)
        results['test_precision'].append(precision)
        results['test_recall'].append(recall)
        results['test_auc'].append(auc)


        print(f'Epoch :{epoch+1},train_loss:{train_loss:.4f},train_acc:{train_acc:.4f},test_loss:{test_loss:.4f},test_acc:{test_acc:.4f}accuracy_score:{acc:.4f},test_f1:{f1:.4f},test_precision:{precision:.4f},test_recall:{recall:.4f},test_auc:{auc:.4f} ')

    
    return results