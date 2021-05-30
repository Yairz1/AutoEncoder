import os
from typing import Any

import torch
from torch import optim
from tqdm import tqdm
from Utils.data_utils import DataUtils


class TrainingUtils:
    @staticmethod
    def test_accuracy(net, criterion, test_loader, device="cpu"):
        loss = 0
        steps = 0
        with torch.no_grad():
            for data in test_loader:
                if len(data) == 2:
                    data, labels = data
                data = data.to(device)
                data_output = net(data)
                loss += criterion(data, data_output).item()
                steps += 1
        print(f"Test loss = {loss / steps}")
        return loss / steps

    @staticmethod
    def init(auto_encoder_init,
             input_size,
             input_seq_size,
             batch_size,
             lr,
             optimizer,
             lstm_layers_size,
             hidden_size: int,
             checkpoint_dir: str,
             decoder_output_size: int,
             device: Any):
        auto_encoder = auto_encoder_init(input_size=input_size,
                                         input_seq_size=input_seq_size,
                                         hidden_size=hidden_size,
                                         num_layers=lstm_layers_size,
                                         batch_size=batch_size,
                                         decoder_output_size=decoder_output_size,
                                         device=device)
        if optimizer == "adam":
            optimizer = optim.Adam(auto_encoder.parameters(), lr=lr)
        else:
            optimizer = optim.SGD(auto_encoder.parameters(), lr=lr, momentum=0.9)

        if checkpoint_dir:
            model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
            auto_encoder.load_state_dict(model_state)
            auto_encoder = auto_encoder.to(device)
            optimizer.load_state_dict(optimizer_state)
        return auto_encoder, optimizer

    @staticmethod
    def train(config,
              auto_encoder_init,
              input_size,
              input_seq_size,
              dataset_name,
              batch_size,
              optimizer,
              criterion,
              lstm_layers_size,
              decoder_output_size,
              epochs,
              device,
              load_data,
              training_iteration,
              validation,
              checkpoint_dir=None,
              data_dir=None):
        auto_encoder, optimizer = TrainingUtils.init(auto_encoder_init,
                                                     input_size,
                                                     input_seq_size,
                                                     batch_size,
                                                     config['lr'],
                                                     optimizer,
                                                     lstm_layers_size,
                                                     config["hidden_size"],
                                                     checkpoint_dir,
                                                     decoder_output_size,
                                                     device)
        auto_encoder.to(device)
        test_loader, train_loader, val_loader = DataUtils.data_factory(dataset_name, data_dir, batch_size, load_data)

        training_info = []
        val_info = []
        for _ in tqdm(range(epochs), desc="Training progress"):  # epochs loop
            average_loss = training_iteration(auto_encoder, config, criterion, device, optimizer, train_loader)
            training_info.append(average_loss)
            # Validation loss
            validation_average_loss = validation(auto_encoder, criterion, device, val_loader)
            val_info.append(validation_average_loss)
        print("Finished Training")
        return auto_encoder, training_info, val_info

    @staticmethod
    def training_iteration(auto_encoder, config, criterion, device, optimizer, train_loader):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            if len(data) == 2:
                data, labels = data
            data = data.to(device)
            optimizer.zero_grad()
            outputs = auto_encoder(data)
            loss = criterion(outputs, data)
            loss.backward()
            if config['grad_clip']:
                torch.nn.utils.clip_grad_norm_(auto_encoder.parameters(), config['grad_clip'])
            optimizer.step()
            # statistics
            running_loss += loss.item()
            epoch_steps += 1
        return running_loss / epoch_steps

    @staticmethod
    def validation(auto_encoder, criterion, device, val_loader):
        val_loss = 0.0
        val_steps = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                if len(data) == 2:
                    data, labels = data
                data = data.to(device)
                outputs = auto_encoder(data)
                loss = criterion(outputs, data)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        return val_loss / val_steps
