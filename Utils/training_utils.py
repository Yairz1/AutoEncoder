import os
from typing import Any

import torch
from torch import optim
from tqdm import tqdm
from Architectures.lstm_autoencoder import AutoEncoder
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
    def init(input_size,
             input_seq_size,
             batch_size,
             lr,
             optimizer,
             lstm_layers_size,
             hidden_size: int,
             checkpoint_dir: str,
             decoder_output_size: int,
             device: Any):
        auto_encoder = AutoEncoder(input_size=input_size,
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
    def train(config, input_size, input_seq_size, dataset_name, batch_size, optimizer, criterion, lstm_layers_size,
              decoder_output_size, epochs, device, load_data,
              checkpoint_dir=None,
              data_dir=None):
        auto_encoder, optimizer = TrainingUtils.init(input_size,
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
        test_loader, train_loader, val_loader = DataUtils.data_loader_factory(dataset_name,
                                                                              data_dir,
                                                                              batch_size,
                                                                              load_data)

        training_info = []
        val_info = []
        for epoch in tqdm(range(epochs)):  # loop over the dataset multiple times
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

                # print statistics
                running_loss += loss.item()
                epoch_steps += 1
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1,
                                                    running_loss / epoch_steps))
                    running_loss = 0.0
            training_info.append(loss.item())

            # Validation loss
            val_loss = 0.0
            val_steps = 0
            for i, data in enumerate(val_loader, 0):
                if len(data) == 2:
                    data, labels = data
                with torch.no_grad():
                    data = data.to(device)
                    outputs = auto_encoder(data)
                    loss = criterion(outputs, data)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            val_info.append(val_loss / val_steps)
        print("Finished Training")
        return auto_encoder, training_info, val_info
