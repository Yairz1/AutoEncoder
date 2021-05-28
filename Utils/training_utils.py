import os
from typing import Any

import torch
from torch import nn, optim

from Architectures.lstm_autoencoder import AutoEncoder
from Utils.data_utils import DataUtils


class TrainingUtils:
    @staticmethod
    def test_accuracy(net, criterion, test_loader, device="cpu"):
        with torch.no_grad():
            test_input = next(iter(test_loader))
            test_input = test_input.to(device)
            test_output = net(test_input)
        loss = criterion(test_input, test_output).item()
        print(f"Test MSE loss = {loss}")
        return loss

    @staticmethod
    def init(batch_size, lstm_layers_size, hidden_size: int, path: str, checkpoint_dir: str, device: Any):
        test_loader, train_loader, val_loader = DataUtils.load_synthetic_data(path, batch_size)

        auto_encoder = AutoEncoder(input_size=1,
                                   input_seq_size=50,
                                   hidden_size=hidden_size,
                                   num_layers=lstm_layers_size,
                                   device=device)
        optimizer = optim.SGD(auto_encoder.parameters(), lr=0.001, momentum=0.9)
        if checkpoint_dir:
            model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
            auto_encoder.load_state_dict(model_state)
            auto_encoder = auto_encoder.to(device)
            optimizer.load_state_dict(optimizer_state)
        return auto_encoder, train_loader, val_loader, test_loader,  optimizer

    @staticmethod
    def train_synthetic(config, batch_size, criterion, lstm_layers_size, epochs, device, checkpoint_dir=None, data_dir=None):
        auto_encoder, train_loader, val_loader, test_loader,  optimizer = TrainingUtils.init(
            batch_size,
            lstm_layers_size,
            config["hidden_size"],
            data_dir,
            checkpoint_dir,
            device)
        auto_encoder.to(device)
        training_info = []
        val_info = []
        for epoch in range(epochs):  # loop over the dataset multiple times
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(train_loader, 0):
                data = data.to(device)
                optimizer.zero_grad()
                outputs = auto_encoder(data)
                loss = criterion(outputs, data)
                loss.backward()
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
                with torch.no_grad():
                    data = data.to(device)
                    outputs = auto_encoder(data)
                    loss = criterion(outputs, data)
                    val_loss += loss.cpu().numpy()
                    val_steps += 1

            val_info.append(val_loss / val_steps)
        print("Finished Training")
        return auto_encoder, training_info, val_info
