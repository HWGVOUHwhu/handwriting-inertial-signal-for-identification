import torch
import os


def train_loop(device, optimizer, scheduler, loss_fn, epochs, train_loader, test_loader, model, model_path):
    train_losses, test_losses = [], []
    train_acc, test_acc = [], []
    best_acc = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for _, (_, statistic_data, gray_latent, tch_latent, label) in enumerate(train_loader):
            statistic_data = statistic_data.to(device).float()
            gray_latent = gray_latent.to(device).float().squeeze(1)
            tch_latent = tch_latent.to(device).float().squeeze(1)
            label = label.to(device)
            # print(label)

            statistic_flat = statistic_data.float().view(statistic_data.size(0), -1)

            features = torch.cat((statistic_flat, gray_latent, tch_latent), dim=1)

            logits = model(features)

            loss = loss_fn(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()/len(train_loader)

            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == label).sum().item()
            train_total += label.size(0)

        train_losses.append(train_loss)
        train_acc.append(train_correct / train_total)

        scheduler.step()

        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for _, (_, statistic_data, gray_latent, tch_latent, label) in enumerate(test_loader):
                statistic_data = statistic_data.to(device)
                gray_latent = gray_latent.to(device).squeeze(1)
                tch_latent = tch_latent.to(device).squeeze(1)
                label = label.to(device)
                # print(label)

                statistic_flat = statistic_data.float().view(statistic_data.size(0), -1)

                features = torch.cat((statistic_flat, gray_latent, tch_latent), dim=1)

                logits = model(features)
                loss = loss_fn(logits, label)

                test_loss += loss.item()/len(test_loader)

                preds = torch.argmax(logits, dim=1)
                test_correct += (preds == label).sum().item()
                test_total += label.size(0)

        test_losses.append(test_loss)
        test_acc.append(test_correct / test_total)

        if test_acc[-1] > best_acc:
            if not os.path.exists(os.path.dirname(model_path)):
                os.makedirs(os.path.dirname(model_path))
            torch.save(model.state_dict(), model_path)
            best_acc = test_acc[-1]
            print('Saved best model!')

        print(f'Epoch-{epoch+1}: train_loss-{train_loss} train_acc-{train_acc[-1]} test_loss-{test_loss} test_acc-{test_acc[-1]}')

    return train_losses, test_losses, train_acc, test_acc

