#wandb.watch(model)
import torch
import wandb


def train_model(model, train_iter, epoch):
    # todos: 
    # define a loss function
    # load the output as target
    # load the data as data or input
    # deal with the wandb 
    model.train()
    for idx, batch in enumerate(train_iter):
	
        prediction = model(data)
        loss = loss_fn(prediction, target)
        wandb.log({"Training Loss": loss.item()})
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects/len(batch)
        wandb.log({"Training Accuracy": acc.item()})
        ...
