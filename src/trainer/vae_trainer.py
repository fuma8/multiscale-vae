import os

import torch
class VAETrainer:
    def __init__(self, model, train_dataloader, val_dataloader, optimizer, scheduler, epochs, save_dir, device):
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.save_dir = save_dir
        self.device = device
        self.history = {'train_loss': [], 'val_loss': [], 'z': [], 'label': []}
        
        os.makedirs(self.save_dir, exist_ok=True)
    
    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        for data, label in self.train_dataloader:
            data = data.to(self.device)
            loss, z = self.model(data)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            self.history['z'].append(z.detach().cpu())
            self.history['label'].append(label)
        total_loss /= len(self.train_dataloader)
        self.history['train_loss'].append(total_loss)
        return total_loss
    
    def validate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for data, _ in self.val_dataloader:
                data = data.to(self.device)
                loss, z = self.model(data)
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
            
            if (epoch+1) % 5 == 0:
                save_path = os.path.join(self.save_dir, f'model_epoch_{epoch+1}.pt')
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved model to {save_path}")
        return self.history