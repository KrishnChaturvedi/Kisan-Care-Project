import torch

def accuracy(y_true, y_pred):
    correct = (y_true == y_pred).sum().item()
    acc = correct / len(y_true) * 100
    return acc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_train_time(start: float, end: float, device: torch.device = device):
    total_time = end - start
    print(f"Time taken by the device {device} is {total_time:.2f} seconds.")
    return total_time

def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy,
               device: torch.device = device):
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        # forward
        y_pred = model(X)
        loss = loss_fn(y_pred, y)

        # accumulate metrics
        train_loss += loss.item()
        train_acc += accuracy(y_true=y, y_pred=y_pred.argmax(dim=1))

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # average metrics
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)

    print(f"Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.2f}%")
    return train_loss, train_acc   # 🔥 FIXED

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              accuracy,
              device: torch.device = device):
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            test_loss += loss_fn(y_pred, y).item()
            test_acc += accuracy(y_true=y, y_pred=y_pred.argmax(dim=1))

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)

    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.2f}%")
    return test_loss, test_acc   # 🔥 FIXED

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy,
               device: torch.device = device):
    model.eval()
    loss, acc = 0, 0

    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)

            loss += loss_fn(y_pred, y).item()
            acc += accuracy(y_true=y, y_pred=y_pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    print(f"Model Name: {model.__class__.__name__} \nLoss: {loss:.4f}  \nAccuracy: {acc:.2f}")
    return loss, acc   # 🔥 FIXED
