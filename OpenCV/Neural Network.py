from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.nn.functional as F
import cv2



transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("dataset_aug", transform=transform)

train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

class FingerCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(32 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        x = x.view(-1, 32 * 14 * 14)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x

model = FingerCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def print_mae_r2(model, data_loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in data_loader:
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.float())
            all_targets.append(labels.float())

    y_pred = torch.cat(all_preds)
    y_true = torch.cat(all_targets)

    mae = torch.mean(torch.abs(y_true - y_pred)).item()
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)
    r2 = (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0

    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")

for epoch in range(10):
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} done")

print(loss)
print_mae_r2(model, train_loader)

torch.save(model.state_dict(), "finger_model.pth")
model.load_state_dict(torch.load("finger_model.pth"))
model.eval()

labels = ["index", "middle", "pinky", "ring", "thumb"]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    hand = frame[h//4:3*h//4, w//4:3*w//4]

    # preprocess
    img = cv2.resize(hand, (64, 64))

    tensor = torch.tensor(img).float() / 255.0
    tensor = tensor.permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output).item()

    label = labels[pred]

    cv2.putText(frame, label, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Finger Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()