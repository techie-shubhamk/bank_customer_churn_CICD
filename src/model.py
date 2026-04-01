import torch
import torch.nn as nn



class bank_churn_model(nn.Module):
    def __init__(self,input_dim,output_dim,num_of_layer,num_nurone_perlayer,dropout):
        super().__init__()

        nn_stages = []
        for i in range(num_of_layer):
            nn_stages.append(nn.Linear(input_dim,num_nurone_perlayer))
            nn_stages.append(nn.BatchNorm1d(num_nurone_perlayer))
            nn_stages.append(nn.ReLU())
            nn_stages.append(nn.Dropout(dropout))

            input_dim = num_nurone_perlayer

        nn_stages.append(nn.Linear(num_nurone_perlayer,output_dim))
        nn_stages.append(nn.Sigmoid())

        self.model = nn.Sequential(*nn_stages)


    def forward(self,features):
        return self.model(features)

