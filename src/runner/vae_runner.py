import os
import pickle

import torch

from src.utils.evaluation_utils import visualize_images_grid, visualize_tensor_images
class VAERunner:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, save_dir, pretrained_path, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.save_dir = save_dir
        self.img_dir = os.path.join(self.save_dir, 'imgs')
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoints')
        self.log_dir = os.path.join(self.save_dir, 'history')
        self.pretrained_path = pretrained_path
        self.device = device
        self.history = {'train_loss': [], 'val_loss': []}
        
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.img_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        if self.pretrained_path is not None:
            self.model.load_state_dict(torch.load(self.pretrained_path))
        
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        for data, label in self.train_dataloader:
            data = data.to(self.device)
            loss, z, x_hat = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        total_loss /= len(self.train_dataloader)
        self.history['train_loss'].append(total_loss)
        return total_loss
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data, _ in self.val_dataloader:
                data = data.to(self.device)
                loss, z, x_hat = self.model(data)
                total_loss += loss.item()
        total_loss /= len(self.val_dataloader)
        self.history['val_loss'].append(total_loss)    
        return total_loss    
    
    def train_model(self):
        for epoch in range(self.epochs):
            print(f"\nEpoch [{epoch+1}/{self.epochs}]")
            train_loss = self.train_one_epoch()
            val_loss = self.validate()
            
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
            self.scheduler.step()
            
            if (epoch+1) % 100 == 0:
                save_path = os.path.join(self.save_dir, f'model_epoch_{epoch+1}.pt')
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved model to {save_path}")
        pkl_name = 'history.pkl'
        pkl_path = os.path.join(self.save_dir, pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.history, f)
        return self.history

    def visualize_reconstructed_image(self, file_name):
        file_path = os.path.join(self.img_dir, file_name)
        self.model.eval()
        for data, label in self.val_dataloader:
            data = data.to(self.device)
            loss, z, x_hat = self.model(data)
            break
        visualize_images_grid(x_hat, file_path)
        visualize_images_grid(data, os.path.join(self.img_dir, 'original_img.jpg'))
    
    def check_vae_parameter(self, file_name):
        file_path = os.path.join(self.log_dir, file_name)
        with open(file_path, 'w', encoding='utf-8') as f:            
            for name, param in self.model.named_parameters():
                f.write(f"Name: {name}\n")
                f.write(f"Shape of the weight: {param.shape}\n")
                f.write(f"Values:\n{param.data}\n\n")
    
    def visualize_sampled_image(self, shape, file_name):
        file_path = os.path.join(self.img_dir, file_name)
        self.model.eval()
        noise = torch.randn(shape).to(self.device)
        x_hat = self.model.sample(noise)
        visualize_images_grid(x_hat, file_path)
    
    def visualize_latent_image(self, file_name):
        file_path = os.path.join(self.img_dir, file_name)
        self.model.eval()
        for data, _ in self.val_dataloader:
            data = data.to(self.device)
            z = self.model(data, visualize_latent=True)
            break
        visualize_tensor_images(z, file_path)
    
    def visualize_noise(self, file_name):
        file_path = os.path.join(self.img_dir, file_name)
        self.model.eval()
        for data, _ in self.val_dataloader:
            data = data.to(self.device)
            z = self.model(data, visualize_noise=True)
            break
        visualize_tensor_images(z, file_path)