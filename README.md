# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
An automobile company has plans to enter new markets with their existing products. After intensive market research, they’ve decided that the behavior of the new market is similar to their existing market.

In their existing market, the sales team has classified all customers into 4 segments (A, B, C, D ). Then, they performed segmented outreach and communication for a different segment of customers. This strategy has work exceptionally well for them. They plan to use the same strategy for the new markets.

You are required to help the manager to predict the right group of the new customers.

## Neural Network Model
<img width="1027" height="695" alt="548853387-1440702b-e4af-497b-ae63-09ea9058293f" src="https://github.com/user-attachments/assets/a0dd31a4-f2f1-4bd9-a34d-16d6136134a3" />

## DESIGN STEPS


### STEP 1: 
Load the dataset, remove irrelevant columns (ID), handle missing values, encode categorical features using Label Encoding, and encode the target class (Segmentation).

### STEP 2: 
Split the dataset into training and testing sets, then normalize the input features using StandardScaler for better neural network performance.


### STEP 3: 
Convert the scaled training and testing data into PyTorch tensors and create DataLoader objects for batch-wise training and evaluation.


### STEP 4: 

Design a feedforward neural network with multiple fully connected layers and ReLU activation functions, ending with an output layer for multi-class classification.

### STEP 5: 

Train the model using CrossEntropyLoss and Adam optimizer by performing forward propagation, loss calculation, backpropagation, and weight updates over multiple epochs.


### STEP 6: 
Evaluate the trained model on test data using accuracy, confusion matrix, and classification report, and perform prediction on a sample input.





## PROGRAM

### Name:Mohamed Zabir Khan A

### Register Number: 212224230162

```python
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,32)
        self.fc4 = nn.Linear(32,4)



    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.3)

        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=0.3)

        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
        
# Initialize the Model, Loss Function, and Optimizer

def train_model(model, train_loader, criterion, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

```

### Dataset Information
<img width="1250" height="240" alt="image" src="https://github.com/user-attachments/assets/35e0d638-5403-4e36-a647-c10d22c24d4a" />


### OUTPUT

<img width="667" height="563" alt="image" src="https://github.com/user-attachments/assets/ae19e657-f858-4816-a9e7-a8093d1e6949" />


## Classification Report
<img width="578" height="422" alt="image" src="https://github.com/user-attachments/assets/2cac0716-9247-43cb-95d7-5b4fc55cae29" />


### New Sample Data Prediction
<img width="383" height="92" alt="image" src="https://github.com/user-attachments/assets/dc6d8592-baa4-4b1d-819e-8d6b5837bb80" />


## RESULT
Neural network classification model for the given dataset is successfully developed.
