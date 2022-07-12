import torch as T
import os
from engine_dataset import EngineDataset
import torch.nn.functional as F
from LSTM_engine import LSTMRegressor
from train import train_model,eval_model
from engine_dataset import load_dataset

def main():
    import os
    import wandb

    # Paste your api key here
    wandb.init() 
    os.environ["WANDB_API_KEY"] = '549ecdb2f42df12b07c5e06178473a51f5796f4c'

    # Feel free to change these and experiment !!
    config = wandb.config
    config.learning_rate = 2e-5
    config.batch_size = 32
    config.input_size = 1
    config.output_size = 2
    config.hidden_size = 256
    config.embedding_length = 300
    config.epochs = 10
    mode="train"


    train_iter, valid_iter, test_iter = load_dataset()

    if mode=="train":
        model = LSTMRegressor(config.batch_size, config.input_size, config.output_size, config.hidden_size)
        loss_fn = F.cross_entropy

        # Create a wandb run to log all your metrics
        run = wandb.init(project='...', entity='...', reinit=True)

        wandb.watch(model)

        for epoch in range(config.epochs):
            train_loss, train_acc = train_model(model, train_iter, epoch)
            val_loss, val_acc = eval_model(model, valid_iter)
            
            print(f'Epoch: {epoch+1:02}, Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, Val. Loss: {val_loss:3f}, Val. Acc: {val_acc:.2f}%')

    run.finish()

if __name__=="__main__":
    main()