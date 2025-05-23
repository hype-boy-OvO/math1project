import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from model_train import set
import math



model = set.Model()
trainer = set.Modeltrainer(model=model)
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'],strict=True)
model.double()
model.eval()
model.to('cuda')

trainer.max, trainer.min = model(torch.tensor([[0], [90]], dtype=torch.float64).to('cuda'))
trainer.max = trainer.max.item()
trainer.min = trainer.min.item()

datas = torch.linspace(0, 90, steps=901)
dataloader = DataLoader(datas, batch_size=4, shuffle=False)
data = torch.tensor([])
for i in dataloader:
    output = model(i.unsqueeze(1).double().to('cuda'))
    output = trainer.min_max_normalize(y=output)
    data = torch.cat([data,output.squeeze(1).to('cpu')])

iter = torch.linspace(0, 90, steps=901)
coses = []
closses = []
sins = []
slosses = []
tans = []
tlosses = []
for data, angle in zip(data,iter):
    angle = angle.item()*torch.pi/180
    cospred = data.item()
    cos = math.cos(angle)
    closs = abs(cospred - cos)
    coses.append(cospred)
    closses.append(closs)

    sinpred = math.sqrt(1 - data.item()**2)
    sin = math.sin(angle)
    sloss = abs(sinpred - sin)
    sins.append(sin)
    slosses.append(sloss)

    if cos >= math.sqrt(2)/2 :
        tanpred = sin/cos
        tan = math.tan(angle)
        tloss = abs(tanpred - tan)
    else:
        tanpred = 10
        tloss = 10

    tans.append(tanpred)
    tlosses.append(tloss)






datas = datas*torch.pi/180
df = pd.DataFrame({'x':datas.detach().numpy(),
                   'cos_loss':closses,
                   'cos':coses,
                   'sin_loss':slosses,
                   'sin':sins,
                   'tan_loss':tlosses,
                   'tan':tans
                   })
df.to_csv("data.csv", index=False)
