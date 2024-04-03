import torch
import numpy as np
from torchinfo import summary
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import dataloader
import model
import numpy as np
from sklearn.cluster import MeanShift

if __name__ == "__main__":
    csv_file = "Tokenized_Outputs/my-new-tokenizer_tokenized.csv"
    batch_size = 2048
    split_size = 0.2

    train_loader,val_loader,test_loader = dataloader.CustomDataloader(csv_file,batch_size)


    device = ("cuda" if torch.cuda.is_available() else "cpu")
    # device = ('cpu')
    print("Using: ", device)

    model = model.BinaryClassifier().to(device)
    summary(model)

    

    loss_fn= nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00004, betas=(0.95, 0.999))


    def train (dataloader,model,loss_fn,optimizer):
        size = len(dataloader.dataset)
        model.train()

        for batch, (x,y) in enumerate(dataloader):
            x,y = x.to(device),y.to(device)

            pred = model(x)
            loss = loss_fn(pred,y.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 2 == 0:
                loss,current = loss.item(), (batch + 1)*len(x)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
    def validate(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct, true_correct = 0, 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                mean = get_mean(pred.cpu().numpy())
                binary_pred = (pred > mean).float() 
                test_loss += loss_fn(pred, y.unsqueeze(1).float()).item()
                for x in range(len(binary_pred)):
                    if binary_pred[x] == y [x]:
                        correct+=1
                print("Predicted Correct values: ",correct)
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



    def get_mean(array):
        
        array = array.reshape(-1, 1)
        
        mean_shift = MeanShift()
        mean_shift.fit(array)
        
        cluster_centers = mean_shift.cluster_centers_
        
        mean_value = np.mean(cluster_centers)
        print(cluster_centers)
        print(mean_value)
        
        return mean_value


    def test(dataloader, model, loss_fn):
        model.eval()
        y_test = []
        predicted= [] 
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                mean = get_mean(pred.cpu().numpy())
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
        accuracy = accuracy_score(y,p)
        recall = recall_score(y ,p)
        precision = precision_score(y,p)
        f1 = f1_score(y,p)

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        print("----------------Batch Training Started---------------")
        train(train_loader, model, loss_fn, optimizer)
        print("----------------Batch Validation Started---------------")
        validate(val_loader, model, loss_fn)
    print("Training Done!")



    print("---------------------MODEL PREDICTION STARTED--------------")

    test(test_loader,model,loss_fn)






