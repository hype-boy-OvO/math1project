import torch
import torch.nn as nn
import set


data = torch.linspace(0, 90, steps=901, dtype=torch.float64,requires_grad=True)
data = data[:,None]
model = set.Model().double()
opt = torch.optim.Adam(model.parameters(), lr=0.001)

checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'],strict=True)
model.double()
opt.load_state_dict(checkpoint['optimizer_state_dict'])

for state in opt.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to('cuda')

best_loss = checkpoint['best_loss']

trainer = set.Modeltrainer(model=model,opt=opt,best_loss=best_loss)


try:
    trainer.train(data,data,num_epochs=10**5)
except KeyboardInterrupt:
    torch.save( {'model_state_dict': trainer.best_state,
                 'optimizer_state_dict': trainer.opt_best_state,
                 'best_loss': trainer.best_loss}, './model.pth')
    
   # torch.save( {'model_state_dict': model.state_dict()}, './modelf.pth')

torch.save( {'model_state_dict': trainer.model.state_dict()}, './model.pth')
