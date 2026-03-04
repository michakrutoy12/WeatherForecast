from src.predict import *
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import pickle

with open('./weights/dataset/prepared.pkl', 'rb') as f:
    prep_data = pickle.load(f)

X = torch.from_numpy(prep_data['train'])
Y = torch.from_numpy(prep_data['targets'])

train_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
targets_idx = [4, 5, 7, 8, 9]

dataset = TensorDataset(X[:, :, train_idx], Y[:, :, targets_idx])

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

print(f'Train size: {train_size} | Test size: {len(dataset) - train_size}')

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

model = LSTMInformer(
    enc_in=len(train_idx),
    c_out=len(targets_idx),
    out_len=prep_data['targets'].shape[1],
    d_model=256,
    n_heads=16,
    n_layers=4,
    factor=5
).to(device)

criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=7e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.4)

epochs = 20
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        output = model(batch_x)

        loss = criterion(output, batch_y) + 0.5 * nn.functional.l1_loss(output[:, :, 0], batch_y[:, :, 0])

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            output = model(batch_x)
            val_loss += criterion(output, batch_y).item()

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss / len(train_loader):.4f} | Val Loss: {val_loss / len(val_loader):.4f}")

torch.save(model.state_dict(), "./weights/weights.pth")

# https://colab.research.google.com/drive/1YW2rN8Dq5x_NxgGBUxsEA6IsIk7BSdRG?usp=sharing