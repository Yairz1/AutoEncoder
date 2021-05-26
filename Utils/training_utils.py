import os
from functools import partial

import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

import torch
from torch.utils.data import DataLoader

from Utils.data_utils import DataUtils

"""Based on https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html"""


class TrainingUtils:
    @staticmethod
    def test_accuracy(net, test_data, device="cpu"):
        testloader = DataLoader(test_data, batch_size=4, shuffle=False)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        return correct / total



# def synthetic_train(net: nn.Module,
#                     epochs: int,
#                     train: Any,
#                     val: Any,
#                     test: Any,
#                     criterion: nn.Module,
#                     optimizer: Optimizer):
#     for epoch in range(epochs):
#         running_loss = 0.0
#         for i, inputs in enumerate(train, 0):
#             optimizer.zero_grad()
#             outputs = net(inputs)
#             loss = criterion(inputs, outputs)
#             writer.add_scalar("Loss/train", loss, epoch)
#             loss.backward()
#             torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip)
#             optimizer.step()
#
#             running_loss += loss.item()
#             if i % 2000 == 1999:  # print every 2000 mini-batches
#                 print('[%d, %5d] loss: %.3f' %
#                       (epoch + 1, i + 1, running_loss / 2000))
#                 running_loss = 0.0
#     test_input = next(iter(test))
#     test_output = net(test_input)
#     loss = criterion(test_input, test_output)
#     writer.add_scalar(f"Loss/test ", loss)
#     print('Finished Training')
#     writer.close()
