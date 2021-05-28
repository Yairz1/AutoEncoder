from typing import Any

from torch import nn
import torch.nn.functional as F
import torch


class EncoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, device: Any):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.device = device

    def forward(self, input: torch.tensor) -> torch.tensor:
        h_0 = torch.randn(self.num_layers, input.shape[0], self.hidden_size, requires_grad=True, device=self.device)
        c_0 = torch.randn(self.num_layers, input.shape[0], self.hidden_size, requires_grad=True, device=self.device)
        output, (h_n, c_n) = self.lstm(input, (h_0, c_0))
        return F.relu(h_n[-1])  # hidden state from the last layer.


class DecoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, input_seq_size: int, device: Any):
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.device = device
        self.input_seq_size = input_seq_size

    def forward(self, z: torch.tensor) -> torch.tensor:
        h_0 = torch.randn(self.num_layers, z.shape[0], self.hidden_size, requires_grad=True, device=self.device)
        c_0 = torch.randn(self.num_layers, z.shape[0], self.hidden_size, requires_grad=True, device=self.device)
        z = z.unsqueeze(1)
        z = z.repeat(1, self.input_seq_size, 1)
        output, (h_n, c_n) = self.lstm(z, (h_0, c_0))
        return torch.relu(output)


class AutoEncoder(nn.Module):
    def __init__(self, input_size: int, input_seq_size: int, hidden_size: int, num_layers: int, device: Any):
        """
        :param input_size: Encoder input size and decoder output size
        :param hidden_size: Encoder hidden size and decoder input size
        :param num_layers: lstm num layers
        """
        super(AutoEncoder, self).__init__()
        # the output of the encoder is h_t and that's why we init the decoder with input_size = hidden_size
        # the output of the decoder should be in the same size of the origin input and that's why the
        # hidden size = input_size
        self.encoder = EncoderLSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   device=device)
        self.decoder = DecoderLSTM(input_size=hidden_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   input_seq_size=input_seq_size,
                                   device=device)
        self.fc = nn.Linear(hidden_size, input_size)

    def forward(self, x: torch.tensor) -> torch.tensor:
        z = self.encoder(x)
        x_gal = self.decoder(z)
        x_gal = F.relu(self.fc(x_gal))
        return x_gal
