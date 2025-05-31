import torch
import torch.nn as nn
import set_after_submit

compute = set_after_submit.Compute_fucs()
data = torch.linspace(0, 90, steps=9001, dtype=torch.float64)
datax, datay = compute.pre_process(data=data,shuffle=True)
datax = datax.requires_grad_(True)

model = set_after_submit.Model().double()
opt = torch.optim.Adam(model.parameters(), lr=0.1)
'''
checkpoint = torch.load('model.pth')
model.load_state_dict(checkpoint['model_state_dict'],strict=True)
model.double()
opt.load_state_dict(checkpoint['optimizer_state_dict'])

for state in opt.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.to('cuda')

best_loss = checkpoint['best_loss']
'''
trainer = set_after_submit.Modeltrainer(model=model,opt=opt)#best_loss=best_loss


try:
    trainer.train(datax,datay,num_epochs=10**5)
finally:
    torch.save( {'model_state_dict': trainer.best_state,
                 'optimizer_state_dict': trainer.opt_best_state,
                 'best_loss': trainer.best_loss}, './after_submit/model_after_submit.pth')
    

