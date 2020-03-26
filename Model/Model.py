import torch.nn as nn



class Model(nn.Module):
    def __init__(self,inputneurons):
        super(Model,self).__init__()
        layers=[nn.Linear(inputneurons,20),nn.Linear(20,20),nn.Linear(20,9),nn.Linear(9,2)]
        self.net=nn.Sequential(*layers)

    def forward(self,x):
        return self.net(x)
