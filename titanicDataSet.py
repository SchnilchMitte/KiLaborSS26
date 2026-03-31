import numpy as np
import pandas as pd
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

random_seed = 42
torch.manual_seed(random_seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TitanicDataSet(Dataset):

    def __init__(self, csv_file, train=True, test_size=0.2, seed=42):
        self.csv_file = csv_file
        self.train = train
        self.test_size = test_size
        self.seed = seed

        data = self._prepare_data(csv_file, test_size)

        if train:
            self.X = data["X_train"]
            self.y = data["y_train"]
        else:
            self.X = data["X_test"]
            self.y = data["y_test"]

        self.feature_names = data["feature_names"]


    def _show_info_about_data(self, df):
        print(df.describe())
        print("Data types:", df.dtypes)


    def _prepare_data(self, csv_file, test_size):
        df = pd.read_csv(csv_file)
        print("test")
        self._show_info_about_data(df)

        # Nur die Features behalten die in den Folien stehen.
        # Errinerung: Wir sollen uns nochmal anschauen warum die anderen Features unpraktisch sind
        feature_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex", "Embarked"]
        target = "Survived"

        df = df[feature_cols + [target]].copy()
        # Shuffle
        df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

        split_idx = int(len(df) * (1 - test_size))
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        numeric_cols = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        # ergebnis von df.dtypes zeigt, dass Sex und Embarked str sind. -> One hot encoden
        categorical_cols = ["Sex", "Embarked"]

        #Missing values:
        # Können wir auch mit Median austauschen: train_df[numeric_cols].fillna(train_df[numeric_cols].mean())
        train_df = train_df.dropna()
        test_df = test_df.dropna()

        train_cat = pd.get_dummies(train_df[categorical_cols], drop_first=False)
        test_cat = pd.get_dummies(test_df[categorical_cols], drop_first=False)

        # Durch align wird sichergestellt, dass in trainings und testset nicht unterschiedliche Kategorien vorkommen.
        # Wenn im Testset z.B. Embarked_Q wäre aber in Train nicht, hätte train und test unterschiedliche anzahl Features
        # Axis= 1 -> Spalten werden verglichen, nicht Zeilen
        train_cat, test_cat = train_cat.align(test_cat, join="left", axis=1, fill_value=0)

        # Numerische Features standardisieren
        # Wichtig weil Age z.b. zwischen 5-90 liegt und Fare bis 500 oder so geht. Mit Standardisierung liegen die Features
        # in iener vergliechbaren Größenordnung
        means = train_df[numeric_cols].mean()
        stds = train_df[numeric_cols].std().replace(0, 1)

        train_num = (train_df[numeric_cols] - means) / stds
        test_num = (test_df[numeric_cols] - means) / stds

        # Alles zusammenfügen
        # durch axis = 1 -> features werden nebeneinander zusammengelegt, axis = 0 untereinander
        X_train_df = pd.concat([train_num, train_cat], axis=1)
        X_test_df = pd.concat([test_num, test_cat], axis=1)

        # umwandlung in NumPy array
        X_train = X_train_df.astype(np.float32).values
        X_test = X_test_df.astype(np.float32).values


        # wegen ValueError: Using a target size (torch.Size([16])) that is different to the input size (torch.Size([16, 1])) is deprecated. Please ensure they have the same size.

        y_train = train_df[target].astype(np.float32).values.reshape(-1, 1)
        y_test = test_df[target].astype(np.float32).values.reshape(-1, 1)

        return {
            "X_train": torch.tensor(X_train, dtype=torch.float32),
            "X_test": torch.tensor(X_test, dtype=torch.float32),
            "y_train": torch.tensor(y_train, dtype=torch.float32),
            "y_test": torch.tensor(y_test, dtype=torch.float32),
            "feature_names": list(X_train_df.columns)
        }

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TitanicNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.do = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(input_dim, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 32)
        self.fc6 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.do(x)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.relu(x)

        x = self.fc6(x)
        x = self.sigmoid(x)
        return x

def compute_metrics(y_true : Tensor, y_pred_probs):
    y_pred = (y_pred_probs >= 0.5).float()

    tp = ((y_pred == 1) & (y_true == 1)).sum().item()
    tn = ((y_pred == 0) & (y_true == 0)).sum().item()
    fp = ((y_pred == 1) & (y_true == 0)).sum().item()
    fn = ((y_pred == 0) & (y_true == 1)).sum().item()

    # Accuracy = (TP + TN) / alles
    if tp + tn + fp + fn == 0:
        accuracy = 0
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Precisision = TP / (TP + FP)
    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    # Recall = TP / (TP + FN)
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    return accuracy, precision, recall


def main():
    titanic_train = TitanicDataSet("data/titanic.csv", train=True)
    titanic_test = TitanicDataSet("data/titanic.csv", train=False)


    train_loader = DataLoader(titanic_train, batch_size=16, shuffle=True)
    test_loader = DataLoader(titanic_test, batch_size=16, shuffle=False)

    input_dim = titanic_train.X.shape[1]
    model = TitanicNet(input_dim)
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 100
    train_losses = []
    test_losses = []

    train_accuracies = []
    test_accuracies = []

    train_precisions = []
    test_precisions = []

    train_recalls = []
    test_recalls = []

    model.eval()
    for epoch in range(num_epochs):
        # ---- Training ----
        model.train()
        model = model.to(device)
        train_loss = 0.0
        train_outputs = []
        train_targets = []
        for X, y in train_loader:
            optimizer.zero_grad()
            y = y.to(device)
            X = X.to(device)
            outputs = model(X)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()
            train_outputs.append(outputs.detach())
            train_targets.append(y)
            # X.size(0) = Batch size (letzter Batch kann kleiner als 16 sein!)
            train_loss += loss.item() * X.size(0)

        train_loss /= len(train_loader.dataset)
        y_train_pred_probs = torch.cat(train_outputs, dim=0)
        y_train_true = torch.cat(train_targets, dim=0)

        train_accuracy, train_precision, train_recall = compute_metrics(
            y_train_true, y_train_pred_probs
        )

        # ---- Evaluation ----
        model = model.to('cpu')
        model.eval()
        test_outputs = []
        test_targets = []
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_loader:
                outputs = model(X)
                loss = criterion(outputs, y)

                test_loss += loss.item() * X.size(0)

                test_outputs.append(outputs)
                test_targets.append(y)

        test_loss /= len(test_loader.dataset)

        y_pred_probs = torch.cat(test_outputs, dim=0)
        y_true = torch.cat(test_targets, dim=0)

        y_test_pred_probs = torch.cat(test_outputs, dim=0)
        y_test_true = torch.cat(test_targets, dim=0)

        test_accuracy, test_precision, test_recall = compute_metrics(
            y_test_true, y_test_pred_probs
        )

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)

        train_precisions.append(train_precision)
        test_precisions.append(test_precision)

        train_recalls.append(train_recall)
        test_recalls.append(test_recall)
        print(
            f"Epoch {epoch + 1:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Train Acc: {train_accuracy:.4f} | "
            f"Test Acc: {test_accuracy:.4f} | "
            f"Test Prec: {test_precision:.4f} | "
            f"Test Recall: {test_recall:.4f}"
        )

    import matplotlib.pyplot as plt

    plt.figure()
    epochs = range(1, num_epochs + 1)
    # Loss
    plt.subplot(2, 1, 1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, test_losses, label="Test Loss")
    plt.legend()

    # Metrics
    ax = plt.subplot(2, 1, 2)
    ax.set_ylim([.5, 1])
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.plot(epochs, train_accuracies, label="Train_Accuracy")
    plt.plot(epochs, test_accuracies, label="Test_Accuracy")
    plt.plot(epochs, train_recalls, label="Train_Recall")
    plt.plot(epochs, test_recalls, label="Test_Recall")
    plt.plot(epochs, train_precisions, label="Train_Precision")
    plt.plot(epochs, test_precisions, label="Test_Precision")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    main()