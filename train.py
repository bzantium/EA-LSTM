import math
import torch
import torch.nn as nn
import torch.optim as optim
from utils import make_cuda
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train(args, model, data_loader):
    MSELoss = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.num_epochs):
        loss = 0
        for step, (features, targets) in enumerate(data_loader):
            features = make_cuda(features)
            targets = make_cuda(targets)

            optimizer.zero_grad()

            preds = model(features)
            mse_loss = MSELoss(preds, targets)
            loss += math.sqrt(mse_loss.item())
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            # print step info
            if (step + 1) % args.log_step == 0:
                print("Epoch [%.2d/%.2d] Step [%.3d/%.3d]: RMSE_loss=%.4f"
                      % (epoch + 1,
                         args.num_epochs,
                         step + 1,
                         len(data_loader),
                         loss / args.log_step))
                loss = 0
    return model


def evaluate(args, model, data_loader):
    model.eval()
    model.lstm.flatten_parameters()
    all_preds = []
    all_targets = []

    for features, targets in data_loader:
        features = make_cuda(features)

        with torch.no_grad():
            preds = model(features)
        all_preds.append(preds)
        all_targets.append(targets)

    all_preds = torch.cat(all_preds, dim=0).unsqueeze(1).cpu().numpy()
    all_targets = torch.cat(all_targets, dim=0).unsqueeze(1).cpu().numpy()
    mse = mean_squared_error(all_targets, all_preds)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_preds)
    print("RMSE = %.4f, MAE = %.4f\n" % (rmse, mae))
    return rmse, mae
