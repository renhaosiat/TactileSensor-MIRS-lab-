import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 2
hidden_size = 20
num_layers = 2
num_classes = 6  
labels = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5'] 
sequence_length = 12  

test_loader = torch.load('testData.pt', weights_only=False) #load raw test data

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.Bilstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0, bidirectional=True)
        self.lstm = nn.LSTM(hidden_size*2, hidden_size, 1, batch_first=True, dropout=0, bidirectional=False)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        h1 = torch.zeros(1, x.size(0), self.hidden_size).to(device)
        c1 = torch.zeros(1, x.size(0), self.hidden_size).to(device)

        out, _ = self.Bilstm(x, (h0, c0))
        out, _ = self.lstm(out, (h1, c1))
        out = self.fc(out[:, -1, :])
        out = self.softmax(out)
        return out

model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)
model.load_state_dict(torch.load('APTM.ckpt', map_location=device)) #load trainned APTM
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels_batch in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(torch.float32).to(device)
        labels_batch = labels_batch.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        all_labels.extend(labels_batch.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

conf_matrix = confusion_matrix(all_labels, all_preds)
accuracy = accuracy_score(all_labels, all_preds)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(8, 6))
plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Greys)
plt.title('Normalized Confusion Matrix')
plt.colorbar()

thresh = 0.8

for x in range(num_classes):
    for y in range(num_classes):
        value = conf_matrix[y, x]
        plt.text(x, y, '{:.2f}'.format(value),
                 verticalalignment='center',
                 horizontalalignment='center',
                 color="white" if value > thresh else "black")

plt.xticks(np.arange(num_classes), labels)
plt.yticks(np.arange(num_classes), labels)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout() 
plt.show()
plt.close()
