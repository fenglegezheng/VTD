import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TransformerEncoderModule(nn.Module):
    def __init__(self, input_size, hidden_size, nhead, num_layers):
        super(TransformerEncoderModule, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead),
            num_layers=num_layers
        )

    def forward(self, x):
        x = self.embedding(x)
        return self.transformer_encoder(x)

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size1, embed_size, num_layers):
        super(Encoder, self).__init__()
        nhead = 4
        self.transformer_encoder = TransformerEncoderModule(input_size, hidden_size1, nhead, num_layers)
        self.fc = nn.Linear(hidden_size1, embed_size)

    def forward(self, x):
        out = self.transformer_encoder(x)
        out = self.fc(out)
        return out

class TransformerDecoderModule(nn.Module):
    def __init__(self, embed_size, hidden_size, nhead, num_layers):
        super(TransformerDecoderModule, self).__init__()
        self.embedding = nn.Linear(embed_size, hidden_size)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead),
            num_layers=num_layers
        )

    def forward(self, tgt, memory):
        tgt = self.embedding(tgt)
        return self.transformer_decoder(tgt, memory)

class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size2, x_size, a_size, y_size, num_layers):
        super(Decoder, self).__init__()
        nhead = 4
        self.transformer_decoder = TransformerDecoderModule(embed_size, hidden_size2, nhead, num_layers)
        self.fcx = nn.Linear(hidden_size2, x_size)
        self.fcy = nn.Linear(hidden_size2 + a_size, y_size)

    def forward(self, x, a):
        out = self.transformer_decoder(x, x)
        x_out = self.fcx(out)
        out = torch.cat((out, a), dim=2)
        y_out = self.fcy(out)
        return x_out, y_out

class VTD(nn.Module):
    def __init__(self, input_size, hidden_size1, embed_size, hidden_size2, x_size, a_size, y_size, num_layers):
        super(VTD, self).__init__()
        self.encoder = Encoder(input_size, hidden_size1, embed_size, num_layers).to(device)
        self.decoder = Decoder(embed_size, hidden_size2, x_size, a_size, y_size, num_layers).to(device)

    def forward(self, x, a, a_po):
        conf = self.encoder(x)
        x_out, y_out = self.decoder(conf, a)
        _, y_po_out = self.decoder(conf, a_po)
        return x_out, y_out, y_po_out, conf
