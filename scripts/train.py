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
    
    encoder = get_model(category='encoder', name=config['model']['encoder'])
    decoder = get_model(category='decoder', name=config['model']['decoder'])
    vae = get_model(category='vae', name=config['model']['vae'], encoder=encoder, decoder=decoder).to(device)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=config['trainer']['args']['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)
    
    # if os.path.exists(config['trainer']['args']['save_dir']):
    #     train = False
    # else:
    #     train = True
    vae_trainer = VAETrainer(model=vae, 
                train_dataloader=train_dataloader,
                val_dataloader = val_dataloader, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                epochs=config['trainer']['args']['epochs'], 
                save_dir=config['trainer']['args']['save_dir'],
                device=device
                )
    # if train:
    history = vae_trainer.train_model()
    with open(f"history_img_size_{config['dataset']['args']['image_size']}.pkl", 'wb') as f:
        pickle.dump(history, f)
    # else:
    #     vae.load_state_dict(torch.load('/home/19x3039_kimishima/multiscale-vae/results/img_size_32/model_epoch_50.pt'))
    #     vae.eval()
    #     z = torch.randn(64, 4, 32, 32).to(device)
    #     x_hat = vae.decoder(z)
    #     visualize_images_grid(x_hat, save_path='plot.jpg')
        

if __name__ == "__main__":
    main()