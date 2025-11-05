import torch
from torch import Tensor

from torch.nn import Module, LSTM, Linear, ReLU, Dropout, LayerNorm, BatchNorm1d, Sequential
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMClassifier(Module):
    def __init__(self, input_size, hidden_size, fc1_size, output_size, dropout_rate):
        super(LSTMClassifier, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc1_size = fc1_size
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # defining modules - two layer bidirectional LSTM with layer norm in between
        self.rnn1 = LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.rnn2 = LSTM(2 * hidden_size, hidden_size, bidirectional=True, batch_first=True)
        self.fc1 = Linear(hidden_size * 4, fc1_size)
        self.fc2 = Linear(fc1_size, output_size)
        self.relu = ReLU()
        self.dropout = Dropout(dropout_rate)
        self.layer_norm = LayerNorm((hidden_size * 2,))
        self.bn = BatchNorm1d(hidden_size * 4)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths, batch_first=True, enforce_sorted=False)
        packed_h1, (final_h1, _) = rnn1(packed_sequence)
        padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths, batch_first=True, enforce_sorted=False)
        _, (final_h2, _) = rnn2(packed_normed_h1)
        return final_h1, final_h2

    def rnn_flow(self, x, lengths):
        batch_size = lengths.size(0)
        h1, h2 = self.extract_features(x, lengths, self.rnn1, self.rnn2, self.layer_norm)
        h = torch.cat((h1, h2), dim=2).permute(1, 0, 2).contiguous().view(batch_size, -1)
        return self.bn(h)

    def mask2length(self, mask):
        """mask [batch_size, seq_length, feat_size]"""
        _mask = torch.mean(mask, dim=-1).long()  # [batch_size, seq_len]
        length = torch.sum(_mask, dim=-1)  # [batch_size,]
        return length

    def forward(self, x, mask):
        lengths = self.mask2length(mask)
        h = self.rnn_flow(x, lengths)
        h = self.fc1(h)
        h = self.dropout(h)
        h = self.relu(h)
        o = self.fc2(h)
        return o, h


class SimpleClassifier(Module):
    """Linear classifier, use embedding as input
    Linear approximation, should append with softmax
    """

    def __init__(self, embd_size, output_dim, dropout):
        super(SimpleClassifier, self).__init__()
        self.dropout = dropout
        self.C = Linear(embd_size, output_dim)
        self.dropout_op = Dropout(dropout)

    def forward(self, x):
        if self.dropout > 0:
            x = self.dropout_op(x)
        return self.C(x)


class Identity(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x) -> Tensor:
        return x


class FcClassifier(Module):
    def __init__(
        self, input_dim: int, layers: list[int], output_dim: int, *, dropout: float = 0.3, use_bn: bool = False
    ) -> None:
        """Fully Connect classifier
        Parameters:
        --------------------------
        input_dim: input feature dim
        layers: [x1, x2, x3] will create 3 layers with x1, x2, x3 hidden nodes respectively.
        output_dim: output feature dim
        activation: activation function
        dropout: dropout rate
        """
        super().__init__()
        self.all_layers = []
        for i in range(0, len(layers)):
            self.all_layers.append(Linear(input_dim, layers[i]))
            self.all_layers.append(ReLU())
            if use_bn:
                self.all_layers.append(BatchNorm1d(layers[i]))
            if dropout > 0:
                self.all_layers.append(Dropout(dropout))
            input_dim = layers[i]

        if len(layers) == 0:
            layers.append(input_dim)
            self.all_layers.append(Identity())
        self.module = Sequential(*self.all_layers)

        self.fc_out = Linear(layers[-1], output_dim)

    def forward(self, x) -> Tensor:
        if x.shape[0] == 1:
            # If batch size is 1, we need to duplicate the input to avoid issues with batch normalization
            x = x.repeat(2, 1)
        feat = self.module(x)
        out = self.fc_out(feat)
        return out


class EF_model_AL(Module):
    def __init__(
        self,
        fc_classifier,
        lstm_classifier,
        out_dim_a,
        out_dim_v,
        fusion_size,
        num_class,
        dropout,
    ):
        """Early fusion model classifier
        Parameters:
        --------------------------
        fc_classifier: acoustic classifier
        lstm_classifier: lexical classifier
        out_dim_a: fc_classifier output dim
        out_dim_v: lstm_classifier output dim
        fusion_size: output_size for fusion model
        num_class: class number
        dropout: dropout rate
        """
        super(EF_model_AL, self).__init__()
        self.fc_classifier = fc_classifier
        self.lstm_classifier = lstm_classifier
        self.out_dim = out_dim_a + out_dim_v
        self.dropout = Dropout(dropout)
        self.num_class = num_class
        self.fusion_size = fusion_size
        # self.out = Sequential(
        #     Linear(self.out_dim, self.fusion_size),
        #     ReLU(),
        #     Linear(self.fusion_size, self.num_class),
        # )
        self.out1 = Linear(self.out_dim, self.fusion_size)
        self.relu = ReLU()
        self.out2 = Linear(self.fusion_size, self.num_class)

    def forward(self, A_feat, L_feat, L_mask):
        _, A_out = self.fc_classifier(A_feat)
        _, L_out = self.lstm_classifier(L_feat, L_mask)
        feat = torch.cat([A_out, L_out], dim=-1)
        feat = self.dropout(feat)
        feat = self.relu(self.out1(feat))
        out = self.out2(self.dropout(feat))
        return out, feat


class MaxPoolFc(Module):
    def __init__(self, hidden_size, num_class=4):
        super(MaxPoolFc, self).__init__()
        self.hidden_size = hidden_size
        self.fc = Sequential(Linear(hidden_size, num_class), ReLU())

    def forward(self, x):
        """x shape => [batch_size, seq_len, hidden_size]"""
        batch_size, seq_len, hidden_size = x.size()
        x = x.view(batch_size, hidden_size, seq_len)
        # print(x.size())
        out = torch.max_pool1d(x, kernel_size=seq_len)
        out = out.squeeze()
        out = self.fc(out)

        return out


if __name__ == "__main__":
    a = FcClassifier(256, [128], 4)
    print(a)
