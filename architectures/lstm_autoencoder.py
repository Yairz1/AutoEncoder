import numpy as np
from torch import nn
import torch


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM

    def forward(self, input):
        h0 = torch.randn(2, 3, 20)
        c0 = torch.randn(2, 3, 20)


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

    def forward(self, x):
        pass


class LSTMAUTOENCODER(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAUTOENCODER, self).__init__()
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers)
        self.decoder = DecoderLSTM(hidden_size, input_size, num_layers)

    def forward(self, input):
        enc_x = self.encoder(input)
        dec_x = self.decoder(enc_x)
        return dec_x
