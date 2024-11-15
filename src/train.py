import torch

def train_epoch(model, dataloader, optimizer, loss_function, device):
    model.train()
    total_loss = 0
    train_correct = 0
    for data, target in dataloader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = train_correct / len(dataloader.dataset)
    return model, total_loss, accuracy


def test_model(model, dataloader, loss_function, device):
    model.eval()
    total_loss = 0
    test_correct = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += loss_function(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
    accuracy = test_correct / len(dataloader.dataset)
    return total_loss, accuracy


def train(model, train_loader, test_loader, optimizer, loss_function, device, num_epochs):
    for epoch in range(num_epochs):
        model, train_loss, train_accuracy = train_epoch(model, train_loader, optimizer, loss_function, device)
        if test_loader:
            test_loss, test_accuracy = test_model(model, test_loader, loss_function, device)
            # print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f} - Test Loss: {test_loss:.4f} - Test Accuracy: {test_accuracy:.4f}')
        else:
            # print(f'Epoch {epoch + 1}/{num_epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}')
            test_accuracy = 'nan'
            test_loss = 'nan'
    return model, train_loss, train_accuracy, test_loss, test_accuracy