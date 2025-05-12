import yaml
import pickle

import torch

from src.factories.dataset_factory import get_dataloader
from src.factories.model_factory import get_model
from src.trainer.vae_trainer import VAETrainer

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
    
    vae_trainer = VAETrainer(model=vae, 
                train_dataloader=train_dataloader,
                val_dataloader = val_dataloader, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                epochs=config['trainer']['args']['epochs'], 
                save_dir=config['trainer']['args']['save_dir'],
                device=device
                )
    history = vae_trainer.train_model()
    with open('history.pkl', 'wb') as f:
        pickle.dump(history, f)

if __name__ == "__main__":
    main()