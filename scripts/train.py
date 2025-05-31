import yaml

import torch

from src.factories.dataset_factory import get_dataloader
from src.factories.model_factory import get_model
from src.runner.vae_runner import VAERunner

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
    
    encoder = get_model(category='encoder', name=config['model']['encoder'], in_channels=config['model']['args']['in_channels'], out_channels=config['model']['args']['latent_channels'])
    decoder = get_model(category='decoder', name=config['model']['decoder'], in_channels=config['model']['args']['latent_channels'], out_channels=config['model']['args']['out_channels'])
    vae = get_model(category='vae', name=config['model']['vae'], encoder=encoder, decoder=decoder).to(device)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=config['trainer']['args']['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[15], gamma=0.1)
    
    save_dir = f"./results/{config['model']['name']}/img_size_{config['dataset']['args']['image_size']}"
    pretrained_path = config['trainer']['args']['pretrained_path']
    
    vae_runner = VAERunner(model=vae, 
                train_dataloader=train_dataloader,
                val_dataloader = val_dataloader, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                epochs=config['trainer']['args']['epochs'], 
                save_dir=save_dir,
                pretrained_path=pretrained_path,
                device=device
                )
    
    if pretrained_path is None:
        vae_runner.train_model()
    else:
        vae_runner.visualize_reconstructed_image(file_name=f'reconstructed_img.jpg')
        vae_runner.check_vae_parameter(output_file='weights.txt')
if __name__ == "__main__":
    main()