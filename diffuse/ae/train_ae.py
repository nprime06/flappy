import torch    
import torch.nn as nn
from torch.optim import Adam
from dataclasses import dataclass

@dataclass
class TrainConfig:
    lr: float
    num_epochs: int
    batch_size: int
    device: str

def train_ae(model, dataloader, train_config):
    model.train()
    model.to(train_config.device)
    optimizer = AdamW(model.parameters(), lr=train_config.lr)
    criterion = nn.MSELoss()

    for epoch in range(CONFIG["num_epochs"]):
        for batch in dataloader:
            batch = batch.to(device)
            output = model(batch)
            loss = criterion(output, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()