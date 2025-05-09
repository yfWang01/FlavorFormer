import torch
import torch.nn as nn
from torch.nn import init
from torch.utils.data import Dataset, DataLoader
import pickle
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import time
from datetime import datetime
import math
import numpy as np
import torch
import random
import yaml


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False  

class NMRDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        self.R = data['R']
        self.S = data['S']
        self.y = data['y']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.R[idx], dtype=torch.float32),
            torch.tensor(self.S[idx], dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )


class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_dim, output_dim, channels=(16, 32), kernel_size=(16, 16), stride=(16, 16), pool_size=(2, 2), dropout=0.3):

        super(CNNFeatureExtractor, self).__init__()
        # assert len(channels) == num_layers"
        num_layers = len(channels)
        layers = []
        in_channels = 1

        for i in range(num_layers):
            layers.append(nn.Conv1d(in_channels, channels[i], kernel_size=kernel_size[i], stride=stride[i], padding='valid'))
            layers.append(nn.BatchNorm1d(channels[i])) 
            layers.append(nn.ReLU())
            layers.append(nn.MaxPool1d(pool_size[i]))
            layers.append(nn.Dropout(dropout)) 
            in_channels = channels[i]

        layers.append(nn.Conv1d(in_channels, output_dim, kernel_size=1))

        self.cnn = nn.Sequential(*layers)

        self.apply(init_weights)  

    def forward(self, x):
        x = x.unsqueeze(1)
        for i, layer in enumerate(self.cnn):
            x = layer(x)
        return x  # (batch_size, output_dim, seq_len)

def init_weights(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear): 
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  
        if m.bias is not None:
            m.bias.data.fill_(0.01)  


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term)  
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input `x` : (batch_size, seq_len, d_model)
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class CrossEncoder(nn.Module): 
    def __init__(self, input_dim, hidden_dim=128, num_heads=8, num_layers=4, num_classes=1, dropout=0.1, max_len=5000, seq_len=32):
        super(CrossEncoder, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim) 
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout, max_len)  
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)  
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))  
        self.sep_token = nn.Parameter(torch.randn(1, 1, hidden_dim)) 
        self.fc_combined = nn.Linear(hidden_dim * 2, num_classes)  
        self.hidden_dim = hidden_dim  

    def forward(self, standard_features, mixture_features):
        batch_size = standard_features.size(0)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_dim)
        sep_tokens = self.sep_token.expand(batch_size, -1, -1)  # (batch_size, 1, hidden_dim)   

        # ----------------- embedding -----------------
        standard_embed = self.embedding(standard_features)  # (batch_size, seq_len, hidden_dim)
        mixture_embed = self.embedding(mixture_features)  # (batch_size, seq_len, hidden_dim)

        # Concatenate tokens
        tokens = torch.cat(
            [cls_tokens, standard_embed, sep_tokens, mixture_embed, cls_tokens], dim=1)  # (batch_size, seq_len, hidden_dim)

        # Add positional encoding
        tokens = self.pos_encoder(tokens)

        # Transformer encoding
        encoded = self.transformer(tokens)  # (batch_size, seq_len, hidden_dim)

        # ----------------- fusion pooling strategy -----------------
        cls_token_output = encoded[:, 0]  
        mean_token_output = encoded.mean(dim=1)  
        combined_output = torch.cat([cls_token_output, mean_token_output], dim=-1)  

        cls_output = self.fc_combined(combined_output) 

        # return logits for BCEWithLogitsLoss
        return cls_output  # (batch_size, num_classes)


class NMRTrainer:
    def __init__(self, cnn_model, encoder_model, device, learning_rate=0.001, weight_decay=0.01, logger=None, early_stopping_patience=20, positive_weight = 3.0):
        self.cnn_model = cnn_model.to(device)
        self.encoder_model = encoder_model.to(device)
        self.device = device
        self.positive_weight = torch.tensor([positive_weight], device=self.device)  # Class Weight Adjustment
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=self.positive_weight)
        self.optimizer = torch.optim.Adam(
            list(cnn_model.parameters()) + list(encoder_model.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self.logger = logger
        self.early_stopping_patience = early_stopping_patience

    def calculate_accuracy(self, outputs, labels):
        probabilities = torch.sigmoid(outputs)
        preds = (probabilities.squeeze() > 0.5).float()
        return (preds == labels).float().mean().item()

    def train(self, train_loader, val_loader, num_epochs=10, save_path="model"):
        train_loss_history, val_loss_history = [], []
        train_acc_history, val_acc_history = [], []

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            self.cnn_model.train()
            self.encoder_model.train()
            epoch_loss, epoch_acc = 0.0, 0.0
            for standard, mixture, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                standard, mixture, labels = (
                    standard.to(self.device),
                    mixture.to(self.device),
                    labels.to(self.device),
                )

                # CNN + Transformer 
                standard_features = self.cnn_model(standard)  # (batch_size, input_dim, seq_len)
                standard_features = standard_features.permute(0, 2, 1)  # (batch_size, seq_len, input_dim)

                mixture_features = self.cnn_model(mixture)
                mixture_features = mixture_features.permute(0, 2, 1)

                outputs = self.encoder_model(standard_features, mixture_features)

                loss = self.criterion(outputs.squeeze(), labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.cnn_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.encoder_model.parameters(), max_norm=1.0)

                self.optimizer.step()
                self.optimizer.zero_grad()

                epoch_loss += loss.item()
                epoch_acc += self.calculate_accuracy(outputs, labels)

            # record training performance
            train_loss_history.append(epoch_loss / len(train_loader))
            train_acc_history.append(epoch_acc / len(train_loader))
            best_val_loss = float('inf')

            # validation
            self.cnn_model.eval()
            self.encoder_model.eval()
            val_loss, val_acc = 0.0, 0.0
            with torch.no_grad():
                for standard, mixture, labels in tqdm(val_loader, desc="Validation"):
                    standard, mixture, labels = (
                        standard.to(self.device),
                        mixture.to(self.device),
                        labels.to(self.device),
                    )
                    standard_features = self.cnn_model(standard)
                    standard_features = standard_features.permute(0, 2, 1)
                    mixture_features = self.cnn_model(mixture)
                    mixture_features = mixture_features.permute(0, 2, 1)

                    outputs = self.encoder_model(standard_features, mixture_features)

                    val_loss += self.criterion(outputs.squeeze(), labels).item()
                    val_acc += self.calculate_accuracy(outputs, labels)
            # print(outputs)
            # record validation performance
            val_loss /= len(val_loader)
            val_acc /= len(val_loader)
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)

            scheduler.step(val_loss)

            epoch_duration = time.time() - epoch_start_time
            if self.logger:
                self.logger.log(epoch, epoch_loss / len(train_loader), val_loss, epoch_acc / len(train_loader), val_acc)
            print(
                f"Epoch {epoch+1} - Time: {epoch_duration:.2f}s, "
                f"Train Loss: {train_loss_history[-1]:.4f}, Train Acc: {train_acc_history[-1]:.4f}, "
                f"Val Loss: {val_loss_history[-1]:.4f}, Val Acc: {val_acc_history[-1]:.4f}"
            )

        # save training information
        os.makedirs(save_path, exist_ok=True)
        training_info = {
            "train_loss": train_loss_history,
            "val_loss": val_loss_history,
            "train_acc": train_acc_history,
            "val_acc": val_acc_history,
        }
        with open(os.path.join(save_path, "training_info.pkl"), "wb") as f:
            pickle.dump(training_info, f)
        print(f"training information has been saved in {os.path.join(save_path, 'training_info.pkl')}")

        return train_loss_history, train_acc_history, val_loss_history, val_acc_history

    def test(self, test_loader):
        self.cnn_model.eval()
        self.encoder_model.eval()
        results = []
        with torch.no_grad():
            for standard, mixture, labels in tqdm(test_loader, desc="Testing"):
                standard, mixture = standard.to(self.device), mixture.to(self.device)
                standard_features = self.cnn_model(standard)
                mixture_features = self.cnn_model(mixture)
                outputs = self.encoder_model(standard_features, mixture_features)
                results.extend(outputs.squeeze().cpu().numpy())
        return results

    def save_model(self, cnn_model, encoder_model, cnn_params, encoder_params, dataset_name, save_path="model", 
                batch_size=None, learning_rate=None, weight_decay=None, num_epochs=None, cnn_output_shape=None, positive_weight=None):

        os.makedirs(save_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        cnn_filename = os.path.join(save_path, f"cnn_params.pth")
        encoder_filename = os.path.join(save_path, f"encoder_params.pth")
        cnn_model_filename = os.path.join(save_path, f"cnn_model.pt")
        encoder_model_filename = os.path.join(save_path, f"encoder_model.pt")

        torch.save(self.cnn_model, cnn_model_filename)
        torch.save(self.encoder_model, encoder_model_filename)

        cnn_structure = str(self.cnn_model)
        encoder_structure = str(self.encoder_model)
        cnn_info = str(summary(self.cnn_model, input_size=(batch_size, 39024)))
        encoder_info = str(summary(self.encoder_model, input_size=[encoder_params['input_shape']]*2))

        device = next(cnn_model.parameters()).device
        with torch.no_grad():
            x = torch.randn(1, cnn_params["input_dim"], device=device)  
            cnn_output_shape = cnn_model(x).shape

        model_info = {
            "cnn_params": cnn_params,
            "encoder_params": encoder_params,
            "training_config": {
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "num_epochs": num_epochs,
                "positive_weight":positive_weight,
            },
            "cnn_output_shape": cnn_output_shape,
            "dataset_name": dataset_name,
            "cnn_model_info": cnn_info,
            "encoder_model_info": encoder_info,
            "cnn_model_structure": cnn_structure,
            "transformer_model_structure": encoder_structure,
            "cnn_output_shape":cnn_output_shape,
        }

        # save as YAML 
        model_info_filename = os.path.join(save_path, f"model_info_{timestamp}.yaml")
        with open(model_info_filename, "w", encoding='utf-8') as f:
            yaml.dump(model_info, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

        print(f"save_path = {save_path}：")
        print(f" - CNN ：{cnn_model_filename}")
        print(f" - Transformer：{encoder_model_filename}")
        print(f" - YAML：{model_info_filename}")

    @staticmethod
    def plot_training_info(training_info_path,save_path):
        with open(training_info_path, "rb") as f:
            training_info = pickle.load(f)

        train_loss = training_info["train_loss"]
        val_loss = training_info["val_loss"]
        train_acc = training_info["train_acc"]
        val_acc = training_info["val_acc"]

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, label="Train Loss")
        plt.plot(val_loss, label="Val Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(train_acc, label="Train Accuracy")
        plt.plot(val_acc, label="Val Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.title("Accuracy Curve")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{save_path}/training_curves.png")
        plt.show()
        # plt.savefig(f"{save_path}/training_curves.png")


def get_data_loader(dataset, batch_size, shuffle=True):
    generator = torch.Generator().manual_seed(42)  
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)

def load_model(model_path, cnn_model_path=None, encoder_model_path=None, yaml_path=None, device='cpu'):

    if yaml_path is not None:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("load yaml：")
        print(config)
    else:
        config = {}

    cnn_model, encoder_model = None, None

    print("load model...")
    cnn_model = torch.load(model_path['cnn'], map_location=device, weights_only=False)
    encoder_model = torch.load(model_path['transformer'], map_location=device, weights_only=False)
    cnn_model.eval()
    encoder_model.eval()


    return cnn_model, encoder_model, config

def main():
    BATCH_SIZE = 256

    train_data_pattern = 'data/augmented_data/50000_samples/train_dataset.pkl'
    valid_data_pattern = 'data/augmented_data/50000_samples/val_dataset.pkl'

    train_loader = get_data_loader(NMRDataset(train_data_pattern), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = get_data_loader(NMRDataset(valid_data_pattern), batch_size=BATCH_SIZE, shuffle=True)

    set_random_seed(42)
    # define CNN 
    cnn_params = { 
        "input_dim": 39042, 
        "output_dim": 128,
        "channels": [64, 64, 64], 
        "kernel_size": [5, 5, 5],    
        "stride": [3, 3, 3],         
        "pool_size": [2, 2, 2],      
        "dropout": 0.1
    }

    cnn_model = CNNFeatureExtractor(**cnn_params)

    # define Transformer 
    transformer_params = {
        "input_dim": cnn_params["output_dim"],
        "hidden_dim":64,
        "num_heads": 4,
        "num_layers": 4,
        "dropout": 0.1,
    }
    transformer_model = CrossEncoder(**transformer_params)
    transformer_params['input_shape'] = (BATCH_SIZE, cnn_output_shape[2], cnn_output_shape[1])
    learning_rate = 5e-5
    weight_decay = 1e-6
    num_epochs = 100
    positive_weight = 3.0
    save_path = f"model/epoch_{num_epochs}_lr_{learning_rate}_wd_{weight_decay}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer = NMRTrainer(cnn_model, transformer_model, device, learning_rate, weight_decay, tb_logger, positive_weight = positive_weight)

    trainer.train(train_loader, val_loader, num_epochs, save_path)

    # save model
    trainer.save_model(
    cnn_model=cnn_model,
    transformer_model=transformer_model,
    cnn_params=cnn_params,
    transformer_params=transformer_params,
    dataset_name=train_data_pattern,
    save_path=save_path,
    batch_size=BATCH_SIZE,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    positive_weight = positive_weight,
    num_epochs=num_epochs,
    cnn_output_shape=cnn_output_shape)


if __name__ == "__main__":
    main()
