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

    @staticmethod
    def hyper_param_search_train(net_init, train_method, num_samples=10, max_num_epochs=10, gpus_per_trial=2):
        data_dir = os.path.join("data", "synthetic_data")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        data = DataUtils.create_synthetic_data(size=10000, sample_size=50, device_type=device, path=data_dir)
        config = {
            "hidden_size": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            "lr": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
            "grad_clip": tune.loguniform(1e-4, 1e-1),
        }
        scheduler = ASHAScheduler(metric="loss", mode="min", max_t=max_num_epochs, grace_period=1, reduction_factor=2)
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])

        result = tune.run(partial(train_method, data_dir=data_dir),
                          resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
                          config=config,
                          num_samples=num_samples,
                          scheduler=scheduler,
                          progress_reporter=reporter)

        best_trial = result.get_best_trial("loss", "min", "last")
        print("Best trial config: {}".format(best_trial.config))
        print("Best trial final validation loss: {}".format(
            best_trial.last_result["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_trial.last_result["accuracy"]))
        best_trained_model = net_init(input_size=1, hidden_size=best_trial.config["hidden_size"], num_layers=2)
        best_trained_model.to(device)

        best_checkpoint_dir = best_trial.checkpoint.value
        model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
        best_trained_model.load_state_dict(model_state)

        test_acc = TrainingUtils.test_accuracy(best_trained_model, device)
        print("Best trial test set accuracy: {}".format(test_acc))
