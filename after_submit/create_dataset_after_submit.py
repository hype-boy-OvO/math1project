import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from model_train import set
import math


compute = set.Compute_fucs()
model = set.Model()
trainer = set.Modeltrainer(model=model)
checkpoint = torch.load('./after_submit/model_after_submit.pth')
model.load_state_dict(checkpoint['model_state_dict'],strict=True)
model.double()
model.eval()
model.to('cuda')


datas = torch.linspace(0, 90, steps=9001)
datax,datay = compute.pre_process(data=datas,shuffle=False)
dataloader = DataLoader(datax, batch_size=8, shuffle=False)
data = torch.tensor([],dtype=torch.double)
for i in dataloader:

    output = model(i.double().to('cuda'))
    data = torch.cat([data,output.squeeze(1).to('cpu')])

iter = torch.linspace(0, 90, steps=9001)
coses = []
closses = []
sins = []
slosses = []
tans = []
tlosses = []
for daty, angle in zip(data,iter):
    angle = angle.item()*torch.pi/180
    cospred = daty.item()
    cos = math.cos(angle)
    closs = abs(cospred - cos)
    coses.append(cospred)
    closses.append(closs)

    val = 1 - cospred**2
    sinpred = max(val,0)
    sin = math.sin(angle)
    sloss = abs(sinpred - sin)
    sins.append(sin)
    slosses.append(sloss)

    if cospred >= math.sqrt(2)/2 :
        tanpred = sinpred/cospred
        tan = math.tan(angle)
        tloss = abs(tanpred - tan)
    else:
        tanpred = 10
        tloss = 10

    tans.append(tanpred)
    tlosses.append(tloss)





datas = torch.linspace(0, 90, steps=9001)
datas = datas*torch.pi/180
print(datas.shape)
print(len(closses))
print(len(coses))
print(len(slosses))
print(len(sins))
print(len(tlosses))
print(len(tans))
print(data.shape)

df = pd.DataFrame({'x':datas.detach().cpu().numpy(),
                   'cos_loss':closses,
                   'cos':coses,
                   'sin_loss':slosses,
                   'sin':sins,
                   'tan_loss':tlosses,
                   'tan':tans,
                   'output':data.detach().cpu().numpy()
                   })
df.to_csv("./after_submit/data_after_submit.csv", index=False)
