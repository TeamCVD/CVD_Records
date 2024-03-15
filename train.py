import torch
from torchinfo import summary
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import dataloader
import model

if __name__ == "__main__":
    csv_file = "Tokenized_Outputs/Bert_tokenized.csv"
    batch_size = 1024
    split_size = 0.2

    train_loader,val_loader,test_loader = dataloader.CustomDataloader(csv_file,batch_size)


    device = ("cuda" if torch.cuda.is_available() else "cpu")
    # device = ('cpu')
    print("Using: ", device)

    model = model.BinaryClassifier().to(device)
    summary(model)

    

    loss_fn= nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)


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

            if batch % 3 == 0:
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
                binary_pred = (pred > 0.5).float() 
                test_loss += loss_fn(pred, y.unsqueeze(1).float()).item()
                for x in range(len(binary_pred)):
                    if binary_pred[x] == y [x]:
                        correct+=1
                print("Predicted Correct values: ",correct)
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def test(dataloader, model, loss_fn):
        model.eval()
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                binary_pred = (pred > 0.5).float() 
                
                accuracy = accuracy_score(y.cpu(),binary_pred.cpu())
                recall = recall_score(y.cpu(),binary_pred.cpu())
                precision = precision_score(y.cpu(),binary_pred.cpu())
                f1 = f1_score(y.cpu(),binary_pred.cpu())

        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    epochs = 10
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        print("----------------Batch Training Started---------------")
        train(train_loader, model, loss_fn, optimizer)
        print("----------------Batch Validation Started---------------")
        validate(val_loader, model, loss_fn)
    print("Training Done!")



    print("---------------------MODEL PREDICTION STARTED--------------")

    test(test_loader,model,loss_fn)






