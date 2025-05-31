import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import copy
import torch.nn.init as init
import statistics as stats

class ResidualBlock(nn.Module):
    def __init__(self,in_features=None,out_features=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.SiLU(),
            )
        self.downsample = nn.Linear(in_features, out_features) if in_features != out_features else None

    def forward(self, x):
        ide = x
        out = self.block(x)

        if self.downsample is not None:
            ide = self.downsample(ide)

        out += ide
        return out

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            ResidualBlock(1, 256),
            ResidualBlock(256,512),
            ResidualBlock(512,256),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    
    def forward(self, x):
        out = self.model(x)
        return out

class Compute_fucs():
    def __init__(self):
        self.grid = torch.linspace(0, 1, steps=400, dtype=torch.float64)

    def pre_process(self,data,shuffle=None):
        data = data[:,None]
        datay = data
        datax = data
        datax = (datax - datax.mean()) / datax.std()
        if shuffle == True :
            perm = torch.randperm(datax.size(0))
            datax = datax[perm][:-200]
            datay = datay[perm][:-200]


        return datax, datay

    def compute_circle(self, cos):
        cos = cos.to('cuda')
        starts = cos
        xs = starts + (self.grid.to('cuda') * (1 - starts))
        length = (self.grid[1]-self.grid[0])*(1 - starts)
        length = length[0].to('cuda')
        xs += length
        xs = xs[:-1]
        ys = torch.sqrt(torch.clamp(1 - xs**2, min=1e-8))
        result = length*ys
        return result.sum(dim=0)
    
    def compute_triangle(self, cos):
        cos = cos.to('cuda')
        return cos*torch.sqrt(torch.clamp(1 - cos**2, min=1e-8))

    def loss_fn(self, output, target,panerty):
        cos = output

        out1 = self.compute_circle(cos)
        out2 = self.compute_triangle(cos)


        output = 2*out1 + out2
        target = target*(torch.pi / 180)

        loss = torch.mean((target - output)**2) 

        return loss + panerty/40

class Modeltrainer():
    def __init__(self,model = None, opt = None, best_loss = None):
        self.model = copy.deepcopy(model)
        self.model.apply(self._init_weights)
        self.model = self.model.to('cuda')
        self.opt = opt if opt is not None else torch.optim.Adam(model.parameters(), lr=0.01)
        self.compute = Compute_fucs()
        self.loss_fn = self.compute.loss_fn
        self.best_loss = best_loss if best_loss is not None else float('inf')
        self.best_state = None
        self.opt_best_state = None
        self.scheduler = ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=20)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            if m.out_features != 1:
                init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif m.out_features == 1:
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('sigmoid'))

            if m.bias is not None:
                init.zeros_(m.bias)

    def check_best_model(self,train_loader):
        loop = tqdm(train_loader, desc=f"eval",leave=False)
        self.model.eval()
        mean = []
        for x, y in loop:
            x = x.to('cuda')
            y = y.to('cuda')
            output = self.model(x)
            panerty = self.is_decrease(output,x,create_graph1=True,create_graph2=False)
            with torch.no_grad():
                loss = self.loss_fn(output,y,panerty)

                loop.set_postfix(loss=loss.item())
                mean.append(loss.item())

        meanloss = sum(mean)/len(mean)
        print(f'std:{stats.pstdev(mean)}')
        self.model.train() 
        if meanloss < self.best_loss:
            self.best_loss = meanloss
            self.best_state = copy.deepcopy(self.model.state_dict()) 
            self.opt_best_state = copy.deepcopy(self.opt.state_dict())  
        return meanloss 

    def is_decrease(self,y,x,create_graph1=None,create_graph2=None):                
        dy_dx = torch.autograd.grad(y.sum(), x,create_graph=create_graph1)[0]
        dec = torch.tensor(0.0, dtype=torch.float64, device='cuda')
        dec += torch.sum(dy_dx[dy_dx > 0])

        dy_dx2 = torch.autograd.grad(dy_dx.sum(), x, create_graph=create_graph2)[0]
        dec += torch.sum(dy_dx2[(dy_dx2 > 0) | (dy_dx2 < -1)])

        return dec



    
    def train(self,x,y,num_epochs):

        self.best_state = copy.deepcopy(self.model.state_dict()) 
        self.opt_best_state = copy.deepcopy(self.opt.state_dict()) 
        finished = False
        eval_loss = []
        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
        for epoch in range(num_epochs):

            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}",leave=False)
            for x, y in loop:
                x = x.to('cuda')
                y = y.to('cuda')

                self.opt.zero_grad()
                output = self.model(x)
                panerty = self.is_decrease(output,x,create_graph1=True,create_graph2=True)
                loss = self.loss_fn(output,y,panerty)
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=7.0)
                self.opt.step()

                loop.set_postfix(loss=loss.item())


            eval = self.check_best_model(train_loader)
            eval_loss.append(eval)
            self.scheduler.step(eval)


            if eval < 1e-3:
                finished = True
                print(f'finished! Epoch {epoch}, Loss: {loss.item()}')
                break

            if (epoch+1) % 10 == 0 and epoch != 0:
                mean_eval_loss = sum(eval_loss)/len(eval_loss)
                eval_loss = []

                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        print(f"{name} - grad mean: {param.grad.mean():.6f}, grad std: {param.grad.std():.6f}")
                print(f'Epoch {epoch+1}, best_loss: {self.best_loss}, mean_eval_loss: {mean_eval_loss}')



            if finished == True:
                torch.cuda.empty_cache()
                break
        
        