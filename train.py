#wandb.watch(model)
import torch
import wandb
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)
    
def train_model(model, train_iter, epoch, loss_fn, batch_size=32):
    total_epoch_loss = 0
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        input = batch[0]
        target = batch[1]
        if len(input)!=batch_size:
            continue
        #target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
        optim.zero_grad()
        prediction = model(input)
        loss = loss_fn(prediction, target)
        wandb.log({"Training Loss": loss.item()})
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        
        if steps % 100 == 0:
            print (f'Epoch: {epoch+1}, Idx: {idx+1}, Training Loss: {loss.item():.4f}')
        
        total_epoch_loss += loss.item()
        
    return total_epoch_loss/len(train_iter)

def eval_model(model, val_iter, loss_fn, batch_size=32):
    total_epoch_loss = 0
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            input = batch[0]
            if len(input)!=batch_size:
                continue
            target = batch[1]
            #target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            prediction = model(input)
            loss = loss_fn(prediction, target)
            wandb.log({"Evaluation Loss": loss.item()})
            total_epoch_loss += loss.item()

    return total_epoch_loss/len(val_iter)

def test_model(model, test_iter, loss_fn, config, scaler):
    total_epoch_loss = 0
    model.eval()
    config.results_root=f"./results/{config.exp_name}"
    os.makedirs(config.results_root, exist_ok=True)
    with torch.no_grad():
        for idx, batch in enumerate(test_iter):
            input = batch[0]
            target = batch[1]
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
            prediction = model(input)
            loss = loss_fn(prediction, target)
            wandb.log({"Test Loss": loss.item()})
            total_epoch_loss += loss.item()
            if config.draw:
                draw_comparison(model, input, target, scaler.sc, config)

    return total_epoch_loss/len(test_iter)

def draw_comparison(model, input, target, sc, config):
    model.eval()
    train_predict = model(input)
    data_predict = train_predict.cpu().data.numpy()
    dataY_plot = target.cpu().data.numpy()
    data_predict = sc.inverse_transform(data_predict)
    dataY_plot = sc.inverse_transform(dataY_plot)
    #sample_size=len(input)
    sample_size=100
    start=0
    end=start+sample_size
    plt.figure(figsize=(20,10))
    plt.plot(dataY_plot[start:end])
    plt.plot(data_predict[start:end],"o", alpha=0.5 )
    plt.suptitle('Engine Time-Series Prediction')
    plt.legend(['real-data', 'prediction'])
    #plt.show()
    plt.savefig(f"{config.results_root}/results.png")
    df=pd.DataFrame(np.concatenate((dataY_plot, data_predict), axis=1), columns=["GT", "Prediction"])
    df.to_csv(f'{config.results_root}/record.csv')

