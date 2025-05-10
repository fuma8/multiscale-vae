import torch
import torch.optim as optim
from torch.utils.data import DataLoader


def vae_trainer(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            # 入力をGPUに転送
            batch = batch.to(device)
            
            # 順伝播
            recon_batch, mu, logvar = model(batch)
            
            # 損失計算
            loss = criterion(recon_batch, batch, mu, logvar)
            
            # バックプロパゲーション
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")
