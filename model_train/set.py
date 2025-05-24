import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import CyclicLR
from tqdm import tqdm
import copy

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64),
            nn.LeakyReLU(negative_slope=0.01),
            nn.LayerNorm(64),

            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),

            nn.Linear(128, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),

            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.LayerNorm(64),

            nn.Linear(64, 1),

        )
    
    def forward(self, x):
        out = self.model(x)
        return out

class Compute_fucs():
    def __init__(self):
        self.grid = torch.linspace(0, 1, steps=10000000, dtype=torch.float64)

    def compute_circle(self, cos):
        cos = cos.to('cuda')
        starts = cos
        xs = starts + (self.grid.to('cuda') * (1 - starts))
        length = (self.grid[1]-self.grid[0])*(1 - starts)
        length = length.to('cuda')
        ys = torch.sqrt(torch.clamp(1 - xs**2, min=1e-8))
        result = length*ys
        return result.sum(dim=1)
    
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
        return loss + 10*panerty

class Modeltrainer():
    def __init__(self,model = None, opt = None, best_loss = None):
        self.model = model.to('cuda')
        self.opt = opt if opt is not None else optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = CyclicLR(self.opt,
                                  base_lr=0.0001,
                                  max_lr=0.006,
                                  step_size_up=113,
                                  mode='exp_range',
                                  gamma=0.99994
                                  )
        self.compute = Compute_fucs()
        self.loss_fn = self.compute.loss_fn
        self.best_loss = best_loss if best_loss is not None else 1e30
        self.best_state = None
        self.opt_best_state = None
        self.min = None
        self.max = None

    def check_best_model(self,train_loader):
        loop = tqdm(train_loader, desc=f"eval",leave=False)
        self.model.eval()
        mean = []
        for x, y in loop:
            x = x.to('cuda')
            y = y.to('cuda')

            output = self.model(x)
            panerty = self.is_decrease(output,x)
            with torch.no_grad():
                loss = self.loss_fn(output, y,panerty)

                loop.set_postfix(loss=loss.item())
                mean.append(loss.item())

        meanloss = sum(mean)/len(mean)
        self.model.train() 
        if meanloss < self.best_loss:
            self.best_loss = loss
            self.best_state = copy.deepcopy(self.model.state_dict()) 
            self.opt_best_state = copy.deepcopy(self.opt.state_dict())   

    def is_decrease(self,y,x):                
        dy_dx = torch.autograd.grad(y.sum(), x, create_graph=True)[0]
        if (dy_dx > 0).any().item() == True:
            dec1 = dy_dx[dy_dx > 0]
            dec1 = torch.sum(dec1)
        else:
            dec1 = 0

        if (dy_dx < -1).any().item() == True:
            dec2 = torch.sum(torch.abs(dy_dx[dy_dx < -1]))
        else:
            dec2 = 0

        return dec1 + dec2



    
    def train(self,x,y,num_epochs):

        self.best_state = copy.deepcopy(self.model.state_dict()) 
        self.opt_best_state = copy.deepcopy(self.opt.state_dict()) 
        finished = False
        dataset = TensorDataset(x, y)
        train_loader = DataLoader(dataset, batch_size=8)
        meanloss = []
        for epoch in range(num_epochs):
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}",leave=False)
            for x, y in loop:
                x = x.to('cuda')
                y = y.to('cuda')

                self.opt.zero_grad()
                output = self.model(x)

                panerty = self.is_decrease(output,x)
                loss = self.loss_fn(output, y, panerty)

                loss.backward()
                self.opt.step()

                loop.set_postfix(loss=loss.item())
                self.scheduler.step()
                meanloss.append(loss.item())

                if loss.item() < 1e-6:
                    finished = True
                    print(f'finished! Epoch {epoch}, Loss: {loss.item()}')
                    break

            self.check_best_model(train_loader)
            if epoch % 10 == 0 and epoch != 0:
                mean =sum(meanloss)/len(meanloss)
                self.check_best_model(train_loader)

                print(f'Epoch {epoch+1}, Loss: {loss.item()}, best_loss: {self.best_loss}, meanloss: {mean}')

                meanloss = []


            if finished == True:
                torch.cuda.empty_cache()
                break
        
        
