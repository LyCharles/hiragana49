import optuna
import torch
import torch.optim as optim
from torch import nn
from data_processing import get_data_loaders
from model import SpinalVGG
def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpinalVGG(num_classes=49).to(device)
    train_loader, val_loader = get_data_loaders()

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3)

    num_epochs = 25
    best_acc = 0.0
    for epoch in range(num_epochs):
        # Training loop here (refer to your code)
        pass  # Simplified

    return best_acc  # Return accuracy as metric for optimization

def run_optuna():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    return study.best_params
