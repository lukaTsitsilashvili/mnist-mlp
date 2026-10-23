import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.001



transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

train_data = datasets.MNIST("data", train=True, download=True, transform=transform)
test_data = datasets.MNIST("data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000)





class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),                
            nn.Linear(784, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 10),           
        )

    def forward(self, x):
        return self.net(x)


model = MLP().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)             
        loss = loss_fn(outputs, labels)     
        optimizer.zero_grad()               
        loss.backward()                     
        optimizer.step()                    

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{EPOCHS}  average loss: {avg_loss:.4f}")



model.eval()
correct = 0

with torch.no_grad():  
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)  
        correct += (predictions == labels).sum().item()

accuracy = 100 * correct / len(test_data)
print(f"\nTest accuracy: {accuracy:.2f}%")


images, labels = next(iter(test_loader))
with torch.no_grad():
    predictions = model(images[:10].to(device)).argmax(dim=1).cpu()

print("\nFirst 10 test images")
print("True labels:", labels[:10].tolist())
print("Predictions:", predictions.tolist())


torch.save(model.state_dict(), "mnist_mlp.pt")
print("\nSaved weights to mnist_mlp.pt")