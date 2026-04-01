from .pipeline import build_pipeline, preprocess_and_save
from .dataset import churn_dataset
from .model import bank_churn_model

import os
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def training_stages(path):
    
    # read data from csv
    df = pd.read_csv(path)

    # data pipeline from here ---
    pipeline = build_pipeline()

    X,y = preprocess_and_save(df,pipeline)

    # tarin test split
    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=42,test_size= .2)

    # data to dataloader

    train_dataset = churn_dataset(X_train,y_train)
    test_dataset = churn_dataset(X_test,y_test)

    train_dataloader = DataLoader(train_dataset,batch_size=32,shuffle=True)
    test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=False)

    # Neural Network

    epochs = 50
    input_dim = X_train.shape[1]
    output_dim = 1
    num_of_layer = 2
    num_of_nurone_perlayer = 99
    dropout = 0.309
    learning_rate = 0.0059
    momentum = 0.65


    #initialize the model
    model = bank_churn_model(input_dim,output_dim,num_of_layer,num_of_nurone_perlayer,dropout)
    # loss function
    criterion = nn.BCELoss()
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate,momentum= momentum)

    # training loop

    for epech in range(epochs):
        model.train() # set the model to training mode
        total_loss = 0
        for features,labels in train_dataloader:
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output,labels.view(-1,1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        
        print(f"Epoch {epech+1}/{epochs}, Loss: {total_loss/len(train_dataloader)}")
        print('-' * 50)

    torch.save(model.state_dict(), os.path.join("Bank Customer Churn", "models", "model.pt"))
    

    # Evaluate on test set
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for features,labels in test_dataloader:
            output = model(features)
            predicted = (output.squeeze() > 0.5).float()
            all_predictions.append(predicted)
            all_labels.append(labels.float())
    
    # Calculate overall accuracy
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    overall_accuracy = (all_predictions == all_labels).float().mean()
    print(f"Overall Test Accuracy: {overall_accuracy.item()*100:.2f}%")
    



    


if __name__ == "__main__":
    path = 'Bank Customer Churn/customer-churn-data/Bank Customer Churn Prediction.csv'
    training_stages(path)