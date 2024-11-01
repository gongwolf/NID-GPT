
import argparse
import pandas as pd 
from sklearn.metrics import classification_report, accuracy_score,f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import torch 
from torch import nn 
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from eval.models import MLP_Mult

def encode_labels(y_train, y_test):
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
    y_test_encoded = encoder.transform(y_test)

    y_train_encoded_df = pd.DataFrame(y_train_encoded, columns=["label"])
    y_test_encoded_df = pd.DataFrame(y_test_encoded, columns=["label"])
    
    label_mapping = {index: label for index, label in enumerate(encoder.classes_)}
    print("Label Mapping:", label_mapping)  # Display label-to-code mapping

    return y_train_encoded_df, y_test_encoded_df, label_mapping


def load_data(train_file, test_file):
    # Load the datasets
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, -1]
    
    label_mapping = None

    if train_df.iloc[:, -1].dtype=="object":
        y_train, y_test, label_mapping= encode_labels(y_train, y_test)

    return X_train, X_test, y_train, y_test, label_mapping

def train_mlp(model, X_train, y_train, X_test, y_test, device, epochs=200, batch_size=64, lr=0.001, verbose = False):
    # Move data to tensors and device
    X_train = torch.tensor(X_train.to_numpy() if isinstance(X_train, pd.DataFrame) else X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train.to_numpy() if isinstance(y_train, pd.DataFrame) else y_train, dtype=torch.long).to(device)
    X_test = torch.tensor(X_test.to_numpy() if isinstance(X_test, pd.DataFrame) else X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test.to_numpy() if isinstance(y_test, pd.DataFrame) else y_test, dtype=torch.long).to(device)

    # Loss function and optimizer with learning rate from Optuna
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Data loader for training
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in tqdm(range(epochs)):
        model.train()
        start_time = time.time()  # Start time of the epoch
        model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.squeeze(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if verbose:
            avg_loss = epoch_loss / len(train_loader)
            model.eval()
            with torch.no_grad():
                y_pred = model(X_test).argmax(dim=1).cpu().numpy()
                f1 = f1_score(y_test.cpu().numpy(), y_pred, average="weighted")  # Weighted F1 score
            if epoch%20==0 and epoch!=0:
                exec_time = time.time() - start_time
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - F1 Score: {f1:.4f} - Time: {exec_time:.4f} sec")

    # Final evaluation after training
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).argmax(dim=1).cpu().numpy()
        f1 = f1_score(y_test.cpu().numpy(), y_pred, average="weighted")
        final_report = classification_report(y_test.cpu().numpy(), y_pred)

    return f1, final_report


def train_and_evaluate_model(model_name, X_train, y_train, X_test, y_test):
    print(f"\nEvaluating model: {model_name}")


    model = init_model(model_name, input_shape=X_train.shape[1], num_classes=len(y_train.iloc[:, -1].unique()))

    if model_name == 'mlp':
        model = model.to(device)
        f1, report = train_mlp(model, X_train, y_train, X_test, y_test, device=device, verbose=True)
        print(report)
    else:
        # Fit the model on the training data
        model.fit(X_train, y_train)
        # Predict on the test data
        y_pred = model.predict(X_test)
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        # Print classification report
        report = classification_report(y_test, y_pred)
        # Output results
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:")
        print(report)

def init_model(model_name, input_shape=76, num_classes = 10):
    if model_name == 'mlp':
        return MLP_Mult(input_shape=input_shape, d_layers=[64,64,128], num_classes=num_classes)
    elif model_name == 'decision_tree':
        return DecisionTreeClassifier(max_depth=28)


def main():
    parser = argparse.ArgumentParser(description="Evaluate different ML models")
    parser.add_argument('--train', type=str, required=True, help='Path to the training CSV file')
    parser.add_argument('--test', type=str, required=True, help='Path to the testing CSV file')
    parser.add_argument('--model', type=str, required=True, choices=['logistic_regression', 'decision_tree', 'random_forest', 'svc','mlp'],
                        help='Name of the model to evaluate')
    args = parser.parse_args()

    X_train, X_test, y_train, y_test, label_mapping = load_data(args.train, args.test)

    print("training X shape:", X_train.shape)
    print("training Y shape:", y_train.shape)
    print("test X shape:", X_test.shape)    
    print("test Y shape:", y_test.shape)

    train_and_evaluate_model(args.model, X_train, y_train, X_test, y_test)

    if label_mapping!=None:
        print(label_mapping)

if __name__ == "__main__":
    main()