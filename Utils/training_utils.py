import os
from typing import Any

import torch
from torch import optim
from tqdm import tqdm
from Utils.data_utils import DataUtils
from collections import defaultdict
from torch.utils.data import random_split


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
        return {str(criterion): loss / steps}

    @staticmethod
    def classification_test_accuracy(net, criterion, test_loader, device, ce_criterion):
        val_loss_mse = 0.0
        val_loss_ce = 0.0
        loss = 0
        steps = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                data, labels = data
                data, labels = data.to(device), labels.to(device)
                reconstruct, predictions = net(data)
                loss_mse = criterion(reconstruct, data).item()
                loss_ce = ce_criterion(predictions, labels).item()
                loss += loss_mse + loss_ce
                val_loss_mse += loss_mse
                val_loss_ce += loss_ce
                steps += 1
                predictions = torch.argmax(predictions, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()
        print(f"Test loss = {loss / steps}")

        loss_mse_avg = val_loss_mse / steps
        loss_ce_avg = val_loss_ce / steps
        accuracy = 100 * correct / total

        return {str(criterion): loss_mse_avg, str(ce_criterion): loss_ce_avg, "accuracy": accuracy}

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
              data_dir=None,
              ):
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

        def def_value():
            return []

        train_info_dic = defaultdict(def_value)
        val_info_dic = defaultdict(def_value)
        for _ in tqdm(range(epochs), desc="Training progress"):  # epochs loop

            train_info = training_iteration(auto_encoder, config, criterion, device, optimizer, train_loader)
            val_info = validation(auto_encoder, criterion, device, val_loader)

            for key, value in train_info.items():
                train_info_dic[key].append(value)
            for key, value in val_info.items():
                val_info_dic[key].append(value)

        print("Finished Training")
        return auto_encoder, train_info_dic, val_info_dic

    @staticmethod
    def kfold_train(train_loader,
                    val_loader,
                    auto_encoder_init,
                    lr,
                    hidden_size,
                    input_size,
                    input_seq_size,
                    batch_size,
                    optimizer,
                    criterion,
                    lstm_layers_size,
                    decoder_output_size,
                    epochs,
                    device,
                    training_iteration,
                    validation,
                    checkpoint_dir=None):
        auto_encoder, optimizer = TrainingUtils.init(auto_encoder_init,
                                                     input_size,
                                                     input_seq_size,
                                                     batch_size,
                                                     lr,
                                                     optimizer,
                                                     lstm_layers_size,
                                                     hidden_size,
                                                     checkpoint_dir,
                                                     decoder_output_size,
                                                     device)
        auto_encoder.to(device)
        training_info = []
        val_info = []
        # train_loader = DataUtils.create_data_loader(data_tensor[tr_ind, :], batch_size)
        # val_loader = DataUtils.create_data_loader(data_tensor[val_ind, :], batch_size)
        for _ in tqdm(range(epochs), desc="Training progress"):  # epochs loop
            average_loss = training_iteration(auto_encoder, criterion, device, optimizer, train_loader)
            training_info.append(average_loss)
            # Validation loss
            validation_average_loss = validation(auto_encoder, criterion, device, val_loader)
            val_info.append(validation_average_loss)

        print("Finished Training")
        return auto_encoder, training_info, val_info

    @staticmethod
    def training_iteration(auto_encoder, mse_criterion, device, optimizer, train_loader):
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(train_loader, 0):
            if len(data) == 2:
                data, labels = data
            data = data.to(device)
            optimizer.zero_grad()
            outputs = auto_encoder(data)
            loss = mse_criterion(outputs, data)
            loss.backward()
            optimizer.step()
            # statistics
            running_loss += loss.item()
            epoch_steps += 1
        return {str(mse_criterion): running_loss / epoch_steps}

    @staticmethod
    def classification_training_iteration(auto_encoder,
                                          config,
                                          mse_criterion,
                                          device,
                                          optimizer,
                                          train_loader,
                                          ce_criterion):
        running_loss_mse = 0.0
        running_loss_ce = 0.0
        epoch_steps = 0
        correct = 0
        total = 0
        for i, data in enumerate(train_loader, 0):
            data, labels = data
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            reconstruct, predictions = auto_encoder(data)
            loss_mse = mse_criterion(reconstruct, data)
            loss_ce = ce_criterion(predictions, labels)
            loss = loss_mse + loss_ce
            loss.backward()
            if config['grad_clip']:
                torch.nn.utils.clip_grad_norm_(auto_encoder.parameters(), config['grad_clip'])
            optimizer.step()
            # statistics
            running_loss_mse += loss_mse.item()
            running_loss_ce += loss_ce.item()
            epoch_steps += 1

            predictions = torch.argmax(predictions, 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

        loss_mse_avg = running_loss_mse / epoch_steps
        loss_ce_avg = running_loss_ce / epoch_steps
        accuracy = 100 * correct / total

        return {str(mse_criterion): loss_mse_avg, str(ce_criterion): loss_ce_avg, "accuracy": accuracy}

    @staticmethod
    def validation(auto_encoder, criterion, device, val_loader):
        val_loss = 0.0
        val_steps = 1
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                if len(data) == 2:
                    data, labels = data
                data = data.to(device)
                outputs = auto_encoder(data)
                loss = criterion(outputs, data)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        return {str(criterion): val_loss / val_steps}

    @staticmethod
    def classification_validation(auto_encoder, criterion, device, val_loader, ce_criterion):
        val_loss_mse = 0.0
        val_loss_ce = 0.0
        val_steps = 0
        correct = 0
        total = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                data, labels = data
                data, labels = data.to(device), labels.to(device)
                reconstruct, predictions = auto_encoder(data)
                loss_mse = criterion(reconstruct, data)
                loss_ce = ce_criterion(predictions, labels)
                val_loss_mse += loss_mse.cpu().numpy()
                val_loss_ce += loss_ce.cpu().numpy()
                val_steps += 1
                predictions = torch.argmax(predictions, 1)
                total += labels.size(0)
                correct += (predictions == labels).sum().item()

        loss_mse_avg = val_loss_mse / val_steps
        loss_ce_avg = val_loss_ce / val_steps
        accuracy = 100 * correct / total

        return {str(criterion): loss_mse_avg, str(ce_criterion): loss_ce_avg, "accuracy": accuracy}
