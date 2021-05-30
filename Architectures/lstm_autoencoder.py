from typing import Any
from torch import nn
import torch


class EncoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, batch_size: int, device: Any):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.device = device
        self.h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, requires_grad=True, device=self.device)
        self.c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, requires_grad=True, device=self.device)

    def forward(self, input: torch.tensor) -> torch.tensor:
        output, (h_n, c_n) = self.lstm(input, (self.h_0, self.c_0))
        return torch.relu(h_n[-1])  # hidden state from the last layer.


class DecoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, input_seq_size: int, batch_size: int,
                 device: Any):
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.device = device
        self.input_seq_size = input_seq_size
        self.h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, requires_grad=True, device=self.device)
        self.c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size, requires_grad=True, device=self.device)

    def forward(self, z: torch.tensor) -> torch.tensor:
        z = z.unsqueeze(1)
        z = z.repeat(1, self.input_seq_size, 1)
        output, (h_n, c_n) = self.lstm(z, (self.h_0, self.c_0))
        return torch.relu(output)


class AutoEncoder(nn.Module):
    def __init__(self, input_size: int, input_seq_size: int, hidden_size: int, num_layers: int, batch_size: int,
                 decoder_output_size: int, device: Any):
        """

        :param input_size: Encoder input size and decoder output size
        :param hidden_size: Encoder hidden size and decoder input size
        :param num_layers: lstm num layers
        :param input_seq_size: length of the input seq
        :param batch_size: 
        :param decoder_output_size: size of the decoder output 
        :param device: 
        """
        super(AutoEncoder, self).__init__()
        # the output of the encoder is h_t and that's why we init the decoder with input_size = hidden_size
        # the output of the decoder should be in the same size of the origin input and that's why the
        # hidden size = input_size
        self.encoder = EncoderLSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   batch_size=batch_size,
                                   device=device)
        self.decoder = DecoderLSTM(input_size=hidden_size,
                                   hidden_size=decoder_output_size,
                                   num_layers=num_layers,
                                   input_seq_size=input_seq_size,
                                   batch_size=batch_size,
                                   device=device)
        self.fc = nn.Linear(decoder_output_size * input_seq_size, decoder_output_size * input_seq_size)
        self.batch_size = batch_size

    def forward(self, x: torch.tensor) -> torch.tensor:
        z = self.encoder(x)
        decoded = self.decoder(z)
        b, w, h = decoded.shape
        decoded = decoded.reshape(b, w * h)
        decoded = torch.relu(self.fc(decoded))
        decoded = decoded.reshape(b, w, h)
        return decoded


class AutoEncoderClassifier(nn.Module):
    def __init__(self, input_size: int, input_seq_size: int, hidden_size: int, num_layers: int, batch_size: int,
                 decoder_output_size: int, classes: int, device: Any):
        """

        :param input_size: Encoder input size and decoder output size
        :param hidden_size: Encoder hidden size and decoder input size
        :param num_layers: lstm num layers
        :param input_seq_size: length of the input seq
        :param batch_size:
        :param decoder_output_size: size of the decoder output
        :param device:
        :param classes: The size of the last layer output
        """
        super(AutoEncoderClassifier, self).__init__()
        # the output of the encoder is h_t and that's why we init the decoder with input_size = hidden_size
        # the output of the decoder should be in the same size of the origin input and that's why the
        # hidden size = input_size
        self.encoder = EncoderLSTM(input_size=input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   batch_size=batch_size,
                                   device=device)
        self.decoder = DecoderLSTM(input_size=hidden_size,
                                   hidden_size=decoder_output_size,
                                   num_layers=num_layers,
                                   input_seq_size=input_seq_size,
                                   batch_size=batch_size,
                                   device=device)
        self.fc = nn.Linear(decoder_output_size, input_size)
        self.classifier = nn.Linear(decoder_output_size, classes)

    def forward(self, input: torch.tensor) -> torch.tensor:
        context_vector = self.encoder(input)
        decoded = self.decoder(context_vector)
        reconstruct = torch.relu(self.fc(decoded))
        predictions = torch.softmax(self.classifier(decoded), dim=1)
        return reconstruct, predictions
