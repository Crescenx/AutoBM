import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torch.nn.functional as F
import json
import pandas as pd

def evaluate_model(model, data_loader, device):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for human_hist, opponent_hist, timestep, labels in data_loader:
            inputs = [t.to(device) for t in (human_hist, opponent_hist, timestep)]
            labels = labels.to(device)

            probs = model(*inputs)
            loss = F.cross_entropy(probs, labels)
            test_loss += loss.item()
    return test_loss / len(data_loader)

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs=200, patience=15):
    from tqdm import tqdm
    import time
    import torch  
    import pandas as pd

    best_val_loss = float('inf')
    best_val_acc = 0.0 
    best_checkpoint = None
    train_losses = []
    val_losses = []
    val_accs = []  
    no_improvement_count = 0

    start_time = time.time()

    pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in pbar:

        model.train()
        running_loss = 0.0
        for human_hist, opponent_hist, timestep, labels in train_loader:
            inputs = [t.to(device) for t in (human_hist, opponent_hist, timestep)]
            labels = labels.to(device)

            optimizer.zero_grad()
            probs = model(*inputs)
            loss = F.cross_entropy(probs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        avg_val_loss = evaluate_model(model, val_loader, device)
        val_losses.append(avg_val_loss)

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for human_hist, opponent_hist, timestep, labels in val_loader:
                inputs = [t.to(device) for t in (human_hist, opponent_hist, timestep)]
                labels = labels.to(device)

                probs = model(*inputs)
                _, predicted = torch.max(probs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_acc = 100 * correct / total
        val_accs.append(val_acc)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_val_acc = val_acc  
            best_checkpoint = model.state_dict()
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        total_time = time.time() - start_time
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        time_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        pbar.set_postfix({
            'epoch': f'{epoch+1}/{num_epochs}',
            'train_loss': f'{avg_train_loss:.4f}',
            'val_loss': f'{avg_val_loss:.4f}',
            'val_acc': f'{val_acc:.2f}%',  
            'best': f'{best_val_loss:.4f}',
            'time': time_str
        })

        if no_improvement_count >= patience:
            pbar.write(f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss for {patience} epochs.")
            break

    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Total training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")

    losses_df = pd.DataFrame({
        'epoch': range(1, len(train_losses) + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_acc': val_accs  
    })

    return {
        'best_checkpoint': best_checkpoint,
        'losses_df': losses_df,
        'best_val_loss': best_val_loss,
        'best_val_acc': best_val_acc  
    }

def test_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    with torch.no_grad():
        for human_hist, opponent_hist, timestep, labels in test_loader:
            inputs = [t.to(device) for t in (human_hist, opponent_hist, timestep)]
            labels = labels.to(device)

            probs = model(*inputs)
            loss = F.cross_entropy(probs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(probs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = total_loss/len(test_loader)
    return avg_loss, accuracy

def test_model_integrity(model, train_loader, device):
    model.train()

    human_hist, opponent_hist, timestep, labels = next(iter(train_loader))
    inputs = [t.to(device) for t in (human_hist, opponent_hist, timestep)]
    labels = labels.to(device)

    try:
        probs = model(*inputs)
    except Exception as e:
        return False, f"Error during forward pass: {e}"

    if torch.isnan(probs).any() or torch.isinf(probs).any():
        return False, "Error: Output contains NaN or inf values."


    try:
        loss = F.cross_entropy(probs, labels)
    except Exception as e:
        return False, f"Error calculating loss: {e}"

    if torch.isnan(loss).any() or torch.isinf(loss).any():
        return False, "Error: Loss is NaN or inf."

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    try:
        optimizer.zero_grad()
        loss.backward()
    except Exception as e:
        return False, f"Error during backward pass: {e}"

    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                return False, f"Error: Gradient for {name} contains NaN or inf values."

    try:
        optimizer.step()
    except Exception as e:
        return False, f"Error during parameter update: {e}"

    return True, None
