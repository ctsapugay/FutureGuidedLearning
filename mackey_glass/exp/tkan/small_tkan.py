import math
import torch
import torch.nn as nn
import tkan.nn as tnn 
from tkan.plotting import plot_kan

import matplotlib.pyplot as plt
torch.random.manual_seed(0)

def goal(x:torch.Tensor) -> torch.Tensor: # the goal function the kan should approx.
    return torch.exp(torch.sin(math.pi * x[:, 0]) + x[:, 1]**2)

with torch.no_grad(): # disable gradient to make faster
    x_train = torch.empty((128, 2)).uniform_(-1, 1) # input of training set
    y_train = goal(x_train) # labeled output of training set
    # validation set
    # create 64 x vals and 64 y vals and combine them to create (x, y) pairs
    xx, yy = torch.meshgrid(torch.linspace(-1, 1, 64), torch.linspace(-1, 1, 64), indexing='ij') 
    x_val = torch.stack([xx, yy], dim=-1).view(-1, 2) # stack pairs and reshape so that there are 2 columns
    y_val = goal(x_val) # calculate the labels for the output of the validation set

# visualize the validation inputs and results
# plt.scatter(x_val[:, 0], x_val[:, 1], c=y_val)
# plt.colorbar()
# plt.show()

model = nn.Sequential( # model configuration
    tnn.LegendreKan(2, 1, order=4),
    tnn.LegendreKan(1, 1, order=3) 
)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.01) # optmizer
criterion = nn.MSELoss() # loss function

epochs = 1000
for epoch in range(epochs):
   model.train() # set to training mode
   y_pred = model(x_train).squeeze(-1) # get prediction
   loss = criterion(y_pred, y_train) # calculate loss

   optimizer.zero_grad() # reset gradient
   loss.backward() # backprop
   optimizer.step() # update parameters

   # validation
   if epoch % 10 == 0 or epoch == epochs - 1:
      model.eval()
      with torch.no_grad():
         y_pred = model(x_val).squeeze(-1)
         val_loss = criterion(y_pred, y_val)
      print(f"\rEpoch {epoch+1:4d}: {loss.item():.3e} - {val_loss.item():.3e}", end="")
      if val_loss < 0.03: # stop at validation loss less than 3%
         break
print()

# plot results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
c = ax1.scatter(x_val[:, 0], x_val[:, 1], c=y_val)
ax1.scatter(x_train[:, 0], x_train[:, 1], color='white', marker='x', alpha=0.75)
fig.colorbar(c, ax=ax1)
ax1.set_title("Ground Truth")
c = ax2.scatter(x_val[:, 0], x_val[:, 1], c=y_pred)
fig.colorbar(c, ax=ax2)
ax2.set_title("KAN Prediction")
plt.show()

# plot kan model
fig = plt.figure(figsize=(6, 10))
plot_kan(model)
plt.show()