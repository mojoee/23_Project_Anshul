import torch as T
import os
import wandb
import torch.nn.functional as F
from Models import LSTM, DNN
from train import train_model, eval_model, test_model
from engine_dataset import load_dataset

def create_config(experiment_name):
    # Feel free to change these and experiment !!
    config = wandb.config
    config.learning_rate = 2e-3
    config.batch_size = 32
    config.input_size = 1
    config.output_size = 1
    config.hidden_size = 32
    config.mode = "train"
    config.epochs = 10
    config.seq_length = 16
    config.num_classes = 1
    config.layers=1
    config.draw=True
    config.save_results=True
    config.exp_name=experiment_name
    config.data_stepsize=1
    return config

def main():

    os.environ["WANDB_API_KEY"] = '549ecdb2f42df12b07c5e06178473a51f5796f4c'
    # Create a wandb run to log all your metrics
    run=wandb.init(project="anshul", entity="mojoee", reinit=True)
    config=create_config(run.name)
    train_iter, valid_iter, test_iter, scaler = load_dataset(config)

    if config.mode=="train":
        #model = LSTM(config.num_classes, config.input_size, config.hidden_size,\
        #      config.layers, config.seq_length)
        model = DNN(config.seq_length)
        loss_fn = F.l1_loss
        #loss_fn = F.mse_loss

        wandb.watch(model)
        if T.cuda.is_available():
            model.cuda()

        for epoch in range(config.epochs):
            train_loss = train_model(model, train_iter, epoch, loss_fn)
            wandb.log({"Training Loss": train_loss})
            val_loss = eval_model(model, valid_iter, loss_fn)
            wandb.log({"Validation Loss": val_loss})
            if epoch==config.epochs-1:

                test_loss = test_model(model, test_iter, loss_fn, config, scaler)
                wandb.log({"Test Loss": test_loss})
            
            print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Val. Loss: {val_loss:3f}')

    run.finish()

if __name__=="__main__":
    main()