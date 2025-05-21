import yaml
import pickle
import os

import torch

from src.factories.dataset_factory import get_dataloader
from src.factories.model_factory import get_model
from src.trainer.vae_trainer import VAETrainer
from src.utils.evaluation_utils import visualize_images_grid

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    config_path = './config/config.yaml'
    config = load_config(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset_args = config['dataset']['args']
    train_dataloader = get_dataloader(dataset_name=config['dataset']['name'], train=True, **dataset_args)._get_dataset()
    val_dataloader = get_dataloader(dataset_name=config['dataset']['name'], train=False)._get_dataset()
    
    encoder = get_model(category='encoder', name=config['model']['encoder'], in_channels=3, out_channels=4)
    decoder = get_model(category='decoder', name=config['model']['decoder'], in_channels=4, out_channels=3)
    vae = get_model(category='vae', name=config['model']['vae'], encoder=encoder, decoder=decoder).to(device)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=config['trainer']['args']['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)
    
    save_dir = f"./results/{config['model']['name']}/img_size_{config['dataset']['args']['image_size']}"
    
    if not os.path.exists(os.path.abspath(save_dir)):
        train = True
    else:
        train = False

    vae_trainer = VAETrainer(model=vae, 
                train_dataloader=train_dataloader,
                val_dataloader = val_dataloader, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                epochs=config['trainer']['args']['epochs'], 
                save_dir=save_dir,
                device=device
                )

    if train:
        history = vae_trainer.train_model()
        pkl_name = 'history.pkl'
        pkl_path = os.path.join(save_dir, pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(history, f)
    else:
        vae.load_state_dict(torch.load('/home/19x3039_kimishima/multiscale-vae/results/img_size_32/model_epoch_200.pt'))
        vae.eval()
        for name, param in vae.named_parameters():
            print(f"名前: {name}")
            print(f"重みの形状: {param.shape}")
            print(f"値:\n{param.data}\n")
            input()
        z = torch.randn(64, 4, 64, 64).to(device)
        x_hat = vae.decoder(z)
        visualize_images_grid(x_hat, save_path='plot.jpg')
        

if __name__ == "__main__":
    main()