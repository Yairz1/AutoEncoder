from torch import nn
import torch


class EncoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(EncoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, input: torch.tensor) -> torch.tensor:
        h_0 = torch.randn(self.num_layers, input.shape[0], self.hidden_size, requires_grad=True, device="cuda:0")
        c_0 = torch.randn(self.num_layers, input.shape[0], self.hidden_size, requires_grad=True, device="cuda:0")
        output, (h_n, c_n) = self.lstm(input, (h_0, c_0))
        return output


class DecoderLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super(DecoderLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, enc_x: torch.tensor) -> torch.tensor:
        h_0 = torch.randn(self.num_layers, enc_x.shape[0], self.hidden_size, requires_grad=True, device="cuda:0")
        c_0 = torch.randn(self.num_layers, enc_x.shape[0], self.hidden_size, requires_grad=True, device="cuda:0")
        dec_x, (h_n, c_n) = self.lstm(enc_x, (h_0, c_0))
        return dec_x


class AutoEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        """
        :param input_size: Encoder input size and decoder output size
        :param hidden_size: Encoder hidden size and decoder input size
        :param num_layers: lstm num layers
        """
        super(AutoEncoder, self).__init__()
        # the output of the encoder is h_t and that's why we init the decoder with input_size = hidden_size
        # the output of the decoder should be in the same size of the origin input and that's why the
        # hidden size = input_size
        self.encoder = EncoderLSTM(input_size, hidden_size, num_layers)
        self.decoder = DecoderLSTM(hidden_size, input_size, num_layers)

    def forward(self, x: torch.tensor) -> torch.tensor:
        enc_x = self.encoder(x)
        dec_x = self.decoder(enc_x)
        return dec_x
