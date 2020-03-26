from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
from Dataset.Dataset import *
from Model.Model import *
import torch
import matplotlib.pyplot as plt

traindataset=MyDataset('../Dataset/all_cvda.csv','../Dataset/CVDA/All/1/trainidx.csv')
traindataloader=DataLoader(traindataset,batch_size=len(traindataset),shuffle=True,num_workers=2)

validationdataset=MyDataset('../Dataset/all_cvda.csv','../Dataset/CVDA/All/1/testidx.csv')
validationdataloader=DataLoader(validationdataset,batch_size=len(validationdataset),shuffle=True,num_workers=2)

mymodel=Model(inputneurons=9)
optimizer=optim.SGD(mymodel.parameters(),lr=1e-5,momentum=0.9)
criterion=nn.CrossEntropyLoss()

totalepochs=400

trainloss=0.0
trainlosses=[]
validationinglosses=[]
for epoch in tqdm(range(totalepochs)):
    for i,batch in enumerate(traindataloader):
        x,y=batch
        optimizer.zero_grad()
        out=mymodel(x.float())
        lossval=criterion(out,y.long())
        trainloss+=lossval
        lossval.backward()
        optimizer.step()

    trainlosses.append(trainloss/(epoch+1))

    validationloss=0.0
    with torch.no_grad():
    	for i,batch in enumerate(validationdataloader):
    		x,y=batch
    		out=mymodel(x.float())
    		lossval=criterion(out,y.long())
    		validationloss+=lossval
    	validationinglosses.append(validationloss)
    
    if((epoch+1)%20==0):
        print(trainloss/epoch+1)

print('Average Train Loss:',trainloss/totalepochs)

validationloss=0.0
with torch.no_grad():
	for i,batch in enumerate(validationdataloader):
	    x,y=batch
	    out=mymodel(x.float())
	    lossval=criterion(out,y.long())
	    validationloss+=lossval

	print('validation loss:',validationloss)
	    

fig=plt.figure()
plt.title('All CVD - Male & Female')
plt.plot(trainlosses,label='Train Loss',c='blue')
plt.plot(validationinglosses,label='Validation Loss',c='red')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss')
fig.savefig('losses.png',dpi=600,bbox_inches = "tight")

	    