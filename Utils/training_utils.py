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
        steps = 1
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
    def prediction_test(net, criterion, test_loader, device):
        running_loss_reconstruct = 0.0
        running_loss_prediction = 0.0
        loss = 0
        steps = 0

        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                b, r, c = data.shape
                data = data.view(b, int(r / 2), 2)
                reconstruct, predict = net(data)
                loss_rec = criterion(reconstruct, data[:, :, 0]).item()
                loss_pre = criterion(predict, data[:, :, 1]).item()

                loss += loss_rec + loss_pre
                running_loss_reconstruct += loss_rec
                running_loss_prediction += loss_pre
                steps += 1

        print(f"Test loss = {loss / steps}")

        loss_reconstruct_avg = running_loss_reconstruct / steps
        loss_prediction_avg = running_loss_prediction / steps

        return {"mse_reconstruct": loss_reconstruct_avg, "mse_prediction": loss_prediction_avg}

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

        train_info_dic = defaultdict(list)
        val_info_dic = defaultdict(list)
        for _ in tqdm(range(epochs), desc="Training progress"):  # epochs loop

            train_info = training_iteration(auto_encoder, criterion, device, optimizer, train_loader)
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
        train_info_dic = defaultdict(list)
        val_info_dic = defaultdict(list)
        for _ in tqdm(range(epochs), desc="Training progress"):  # epochs loop
            train_info = training_iteration(auto_encoder, criterion, device, optimizer, train_loader)
            val_info = validation(auto_encoder, criterion, device, val_loader)

            for key, value in train_info.items():
                train_info_dic[key].append(value)
            for key, value in val_info.items():
                val_info_dic[key].append(value)

        print("Finished Training")
        return auto_encoder, train_info_dic, val_info_dic

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
    def prediction_training_iteration(auto_encoder,
                                      mse_criterion,
                                      device,
                                      optimizer,
                                      train_loader):
        running_loss_reconstruct = 0.0
        running_loss_prediction = 0.0
        epoch_steps = 0

        for i, data in enumerate(train_loader, 0):
            data = data.to(device)
            optimizer.zero_grad()
            b, r, c = data.shape
            data = data.view(b, int(r/2), 2)
            reconstruct, predict = auto_encoder(data)
            loss_rec = mse_criterion(reconstruct, data[:, :, 0])
            loss_pre = mse_criterion(predict, data[:, :, 1])
            loss = loss_rec + loss_pre
            loss.backward()
            optimizer.step()
            # statistics
            running_loss_reconstruct += loss_rec.item()
            running_loss_prediction += loss_pre.item()
            epoch_steps += 1

        loss_reconstruct_avg = running_loss_reconstruct / epoch_steps
        loss_prediction_avg = running_loss_prediction / epoch_steps

        return {"mse_reconstruct": loss_reconstruct_avg, "mse_prediction": loss_prediction_avg}

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

    @staticmethod
    def prediction_validation(auto_encoder, criterion, device, val_loader):
        val_steps = 0
        running_loss_reconstruct = 0.0
        running_loss_prediction = 0.0

        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                data = data.to(device)
                b, r, c = data.shape
                data = data.view(b, int(r / 2), 2)
                reconstruct, predict = auto_encoder(data)
                loss_rec = criterion(reconstruct, data[:, :, 0])
                loss_pre = criterion(predict, data[:, :, 1])
                running_loss_reconstruct += loss_rec.cpu().numpy()
                running_loss_prediction += loss_pre.cpu().numpy()
                val_steps += 1

        loss_reconstruct_avg = running_loss_reconstruct / val_steps
        loss_prediction_avg = running_loss_prediction / val_steps

        return {"mse_reconstruct": loss_reconstruct_avg, "mse_prediction": loss_prediction_avg}