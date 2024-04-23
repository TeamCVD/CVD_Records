import torch
import numpy as np
from torchinfo import summary
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

import dataloader
import model
import cnn_model
import numpy as np
import pandas

from resnet1d import ResNet1D
from sklearn.cluster import MeanShift

if __name__ == "__main__":
    csv_file = "Smoothened_Data/smoothened.csv"
    # csv_file = "Tokenized_Outputs/Encoded_code.csv"
    batch_size = 256
    split_size = 0.2

    train_loader, val_loader, test_loader = dataloader.CustomDataloader(csv_file, batch_size)

    device = ("cuda" if torch.cuda.is_available() else "cpu")
    # device = ('cpu')
    print("Using: ", device)

    # model = ResNet1D (
    #     kernel_size= 16,
    #     in_channels = 1,
    #     base_filters = 64,
    #     stride=2,
    #     groups=32,
    #     n_block=48,
    #     n_classes=1,
    #     downsample_gap=6,
    #     increasefilter_gap=12,
    #     use_do=True
    # )

    # model.to(device)

    model = cnn_model.BinaryClassifierCNN().to(device)
    # model = model.BinaryClassifier().to(device)


    summary(model)

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, betas=(0.99, 0.999))


    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()

        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            x = torch.unsqueeze(x, 1)
            pred_nn = model(x)
            clf = RandomForestModel(pred_nn, y)
            pred = clf.predict(pred_nn.cpu().detach().numpy())
            pred = torch.tensor(pred, requires_grad=True).to(device)
            loss = loss_fn(pred.float(), y.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 2 == 0:
                loss, current = loss.item(), (batch + 1) * len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


    def validate(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct, true_correct = 0, 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                X = torch.unsqueeze(X, 1)
                pred_nn = model(X)
                # print(pred_nn)
                # clf = RandomForestModel(pred_nn, y)
                pred = clf.predict(pred_nn.cpu().detach().numpy())
                mean = get_mean(pred)
                pred = torch.tensor(pred, requires_grad=True).to(device)
                # print(pred)
                # mean = 0.5
                binary_pred = (pred > mean).float()
                # print(binary_pred)
                test_loss += loss_fn(pred.unsqueeze(1).float(), y.unsqueeze(1).float()).item()
                batch_correct = 0
                for x in range(len(binary_pred)):
                    if binary_pred[x] == y[x]:
                        batch_correct += 1
                # print("Batch_Correct: ", batch_correct)
                correct += batch_correct
                print("Predicted Correct values: ", correct)
                print()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


    from sklearn.cluster import MeanShift
    import numpy as np


    def get_mean(array,):
        array = array.reshape(-1, 1)

        # Setting bandwidth to a small value to encourage more clusters
        bandwidth = 0.1

        mean_shift = MeanShift(bandwidth=bandwidth)
        mean_shift.fit(array)

        cluster_centers = mean_shift.cluster_centers_
        cluster_labels = mean_shift.labels_

        # Count the number of data points in each cluster
        cluster_counts = np.bincount(cluster_labels)

        # Sort cluster centers to ensure first two are the most significant
        # cluster_centers_sorted = np.sort(cluster_centers.flatten())

        # Calculate the mean of the first two cluster centers
        # mean_value = np.mean(cluster_centers_sorted[:2])

        total = 0

        for center,count in zip(cluster_centers,cluster_counts):
            total += center * count

        mean_value = total/np.sum(cluster_counts)

        print("Cluster Centers and Counts:")
        for center, count in zip(cluster_centers, cluster_counts):
            print(f"Center: {center}, Count: {count}")

        print("Mean Value:", mean_value)
        # print()

        return mean_value[0]
    def RandomForestModel(X,y):
        X = X.squeeze()
        y = y.squeeze()
        X_array = X.detach().cpu().numpy()
        y_array = y.detach().cpu().numpy()
        clf = RandomForestClassifier(max_depth=100, random_state=42)
        clf.fit(X_array, y_array.astype(float))
        return clf

    def test(dataloader, model, loss_fn):
        model.eval()
        y_test = []
        predicted = []
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                X = torch.unsqueeze(X, 1)
                pred_nn = model(X)
                # print(pred_nn)
                clf = RandomForestModel(pred_nn, y)
                pred = clf.predict(pred_nn.cpu().detach().numpy())
                mean = get_mean(pred)
                pred = torch.tensor(pred, requires_grad=True).to(device)
                # print(pred)
                # mean = 0.5
                binary_pred = (pred > mean).float()
                predicted.append(binary_pred.cpu().numpy())
                y_test.append(y.cpu().numpy())

        # np.ndarray(y_test)
        # np.ndarray(predicted)
        # print(y_test)
        p = []
        y = []
        for x in predicted:
            for pr in x:
                p.append(pr)

        for x in y_test:
            for yr in x:
                y.append(yr)
        accuracy = accuracy_score(y, p)
        recall = recall_score(y, p)
        precision = precision_score(y, p)
        f1 = f1_score(y, p)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")






    epochs = 8
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        print("----------------Batch Training Started---------------")
        train(train_loader, model, loss_fn, optimizer)
        print("----------------Batch Validation Started---------------")
        validate(val_loader, model, loss_fn)
    print("Training Done!")

    print("---------------------MODEL PREDICTION STARTED--------------")

    test(test_loader, model, loss_fn)
    print("---------------------MODEL PREDICTION FINISHED--------------")
