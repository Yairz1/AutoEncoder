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
        return loss / steps, 0, 0

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
        return val_loss_mse / steps, val_loss_ce / steps, 100 * correct / total

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
        training_info_reconstructing = []
        training_info_classifying = []

        val_info_reconstructing = []
        val_info_classifying = []
        val_info = []
        accuracy_training_info = []
        accuracy_val_info = []
        for _ in tqdm(range(epochs), desc="Training progress"):  # epochs loop
            loss_train_reconstructing, loss_train_classifying, accuracy_train = training_iteration(auto_encoder,
                                                                                                   config,
                                                                                                   criterion,
                                                                                                   device,
                                                                                                   optimizer,
                                                                                                   train_loader)
            training_info_reconstructing.append(loss_train_reconstructing)
            training_info_classifying.append(loss_train_classifying)
            accuracy_training_info.append(accuracy_train)
            # Validation loss
            val_loss_reconstructing, val_loss_classifying, validation_accuracy = validation(auto_encoder,
                                                                                            criterion,
                                                                                            device,
                                                                                            val_loader)
            val_info_reconstructing.append(val_loss_reconstructing)
            val_info_classifying.append(val_loss_classifying)
            accuracy_val_info.append(validation_accuracy)
        print("Finished Training")
        return auto_encoder, training_info_reconstructing, training_info_classifying, val_info_reconstructing, \
               val_info_classifying, accuracy_training_info, accuracy_val_info

    @staticmethod
    def training_iteration(auto_encoder, config, mse_criterion, device, optimizer, train_loader):
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
            if config['grad_clip']:
                torch.nn.utils.clip_grad_norm_(auto_encoder.parameters(), config['grad_clip'])
            optimizer.step()
            # statistics
            running_loss += loss.item()
            epoch_steps += 1
        return running_loss / epoch_steps, 0, 0

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

        return running_loss_mse / epoch_steps, running_loss_ce / epoch_steps, 100 * correct / total

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
        return val_loss / val_steps, 0, 0

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
        return val_loss_mse / val_steps, val_loss_ce / val_steps, 100 * correct / total
