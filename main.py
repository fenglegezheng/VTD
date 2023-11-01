import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

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

device = torch.device('cuda')
rank = [0.75*p, 0.5*p, 0.25*p, 0.1*p]
drop = [0, 0.1, 0.2, 0.3]
n_layers = [1, 2, 3]
lr = [0.01, 0.005, 0.001]
alpha = [0, 0.5, 1, 2, 5]
theta = [0, 0.5, 1, 2, 5]
n_epochs = [100, 200]
b_size = [64, 128, 256]
rank = math.ceil(rank[2])
drop = drop[3]
n_layers = n_layers[1]
lr = lr[2]
alpha = alpha[1]
theta = theta[2]
n_epochs = n_epochs[0]
b_size = b_size[1]
train, test = train_test_split(data, test_size=0.1, shuffle=False)
train, test = torch.from_numpy(train.astype(np.float32)), torch.from_numpy(test.astype(np.float32))
train_loader = DataLoader(dataset=train, batch_size=b_size, shuffle=True)
a_po = np.empty(shape=(2**k,train.shape[0],T,k))

def all_comb(length):
    return np.array(np.meshgrid(*[[0,1]]*length, indexing='ij')).reshape((length,-1)).transpose()
x = all_comb(k)
for i in range(2**k):
    a_po[i] = (a == x[i]).all(2)[:,:,np.newaxis]*1
a_po = torch.tensor(a_po, dtype=torch.float32)
scaler = StandardScaler()
a_po = torch.tensor(scaler.fit_transform(a_po.reshape(-1, k)).reshape((2**k,train.shape[0],T,k)), dtype=torch.float32)
a_po_test = np.empty(shape=(2**k,test.shape[0],T,k))
for i in range(2**k):
    a_po_test[i] = (a_test == x[i]).all(2)[:,:,np.newaxis]*1
a_po_test = torch.tensor(a_po_test, dtype=torch.float32)
a_po_test = torch.tensor(scaler.transform(a_po_test.reshape(-1, k)).reshape((2**k,test.shape[0],T,k)), dtype=torch.float32)
vtd = VTD(x_size*3, rank, rank, rank, x_size, a_size, y_size, n_layers)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(vtd.parameters(), lr=lr)
vtd.train()
min_loss = 1e10
for epoch in range(n_epochs):
    for idx, x_batch in enumerate(train_loader):
        optimizer.zero_grad()
        x_out, y_out, y_po_out, conf = vtd(x_batch[:, :x_size], x_batch[:, x_size:x_size+a_size][:,:,np.newaxis], a_po.permute(1,0,2,3))
        loss = criterion(x_out, x_batch[:, :x_size]) + alpha*criterion(y_out, x_batch[:, x_size+a_size:x_size+a_size+y_size]) + theta*criterion(y_po_out, x_batch[:, x_size+a_size:x_size+a_size+y_size].repeat((2**k,1,1,1)).permute(1,0,2,3))
        loss.backward()
        optimizer.step()
    vtd.eval()
    with torch.no_grad():
        x_out, y_out, y_po_out, conf = vtd(test[:, :x_size], test[:, x_size:x_size+a_size][:,:,np.newaxis], a_po_test.permute(1,0,2,3))
        loss = criterion(x_out, test[:, :x_size]) + alpha*criterion(y_out, test[:, x_size+a_size:x_size+a_size+y_size]) + theta*criterion(y_po_out, test[:, x_size+a_size:x_size+a_size+y_size].repeat((2**k,1,1,1)).permute(1,0,2,3))
    if loss.item() < min_loss:
        min_loss = loss.item()
        best_vtd = vtd
vtd = best_vtd
conf_train = vtd.encoder(train[:, :x_size])
conf_test = vtd.encoder(test[:, :x_size])
clf = LinearRegression().fit(conf_train.cpu().detach().numpy().reshape((-1, rank)), train[:, x_size:x_size+a_size].cpu().detach().numpy().reshape((-1, a_size)))
rmse = np.mean(clf.predict(conf_test.cpu().detach().numpy().reshape((-1, rank))) == test[:, x_size:x_size+a_size].cpu().detach().numpy().reshape((-1, a_size)))
print(f"Accuracy: {rmse:.4f}")
