```python
!jupyter nbconvert --to markdown 8_8_6_Exercises.ipynb
```


```python
import sys
import torch.nn as nn
import torch
import warnings
sys.path.append('/home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/')
import d2l
from torchsummary import summary
warnings.filterwarnings("ignore")

class Data(d2l.DataModule):
    def __init__(self, batch_size=16, T=1000, num_train=600, tau=4, randn=0.2):
        self.save_hyperparameters()
        self.time = torch.range(1, T, dtype=torch.float32)
        self.x = torch.sin(0.01*self.time) + torch.randn(T)*randn
        
    def get_dataloader(self, train):
        features = [self.x[i:self.T-self.tau+i] for i in range(self.tau)]
        self.features = torch.stack(features, 1)
        self.labels = self.x[self.tau:].reshape((-1, 1))
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.features, self.labels], train, i)
    
class TauData(d2l.DataModule):
    def __init__(self, x, batch_size=16, T=1000, num_train=600, tau=4):
        self.save_hyperparameters()
        self.time = torch.range(1, T, dtype=torch.float32)
        
    def get_dataloader(self, train):
        features = [self.x[i:self.T-self.tau+i] for i in range(self.tau)]
        self.features = torch.stack(features, 1)
        self.labels = self.x[self.tau:].reshape((-1, 1))
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.features, self.labels], train, i)
```

# 1. Improve the model in the experiment of this section.

## 1.1 Incorporate more than the past four observations? How many do you really need?

past **64** observations will be enough.


```python
org_data = Data()
taus = [4, 8, 16, 32, 64, 128, 256, 512]
tau_loss = []
for t in taus:
    data = TauData(x=org_data.x, tau=t)
    model = d2l.LinearRegression(lr=0.01)
    trainer = d2l.Trainer(max_epochs=5)
    trainer.fit(model, data)
    onestep_preds = model(data.features[data.num_train:])
    tau_loss.append(model.loss(y_hat=onestep_preds, y=data.labels[data.num_train:]).item())
```


    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_5_0.svg)
    



    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_5_1.svg)
    



    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_5_2.svg)
    



    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_5_3.svg)
    



    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_5_4.svg)
    



    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_5_5.svg)
    



    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_5_6.svg)
    



    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_5_7.svg)
    



```python
d2l.plot(taus, tau_loss)
```


    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_6_0.svg)
    


## 1.2 How many past observations would you need if there was no noise? Hint: you can write $\sin$ and $\cos$ as a differential equation.



past 8/64 observations will be enough.


```python
org_data = Data(randn=0)
taus = [1, 2, 4, 8, 16, 32, 64, 128]
tau_loss = []
for t in taus:
    data = TauData(x=org_data.x, tau=t)
    model = d2l.LinearRegression(lr=0.01)
    trainer = d2l.Trainer(max_epochs=5)
    trainer.fit(model, data)
    onestep_preds = model(data.features[data.num_train:])
    tau_loss.append(model.loss(y_hat=onestep_preds, y=data.labels[data.num_train:]).item())
```


    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_9_0.svg)
    



    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_9_1.svg)
    



    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_9_2.svg)
    



    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_9_3.svg)
    



    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_9_4.svg)
    



    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_9_5.svg)
    



    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_9_6.svg)
    



    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_9_7.svg)
    



```python
d2l.plot(taus, tau_loss)
```


    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_10_0.svg)
    


## 1.3 Can you incorporate older observations while keeping the total number of features constant? Does this improve accuracy? Why?

We can use weighted average of older observations while keeping the total number of features constant, while this does not improve accuracy, because the weighted average might lose some information.


```python
class CorOldData(d2l.DataModule):
    def __init__(self, x, batch_size=16, T=1000, num_train=600, tau=4, randn=0.2):
        self.save_hyperparameters()
        self.time = torch.range(1, T, dtype=torch.float32)
        self.x = x
        
    def get_dataloader(self, train):
        features = [self.x[i:self.T-self.tau+i] for i in range(self.tau+1)]
        features = torch.stack(features, 1)
        features[:, -2] = (features[:, -2]+features[:, -1])/2
        self.features = features[:, :-1]
        self.labels = self.x[self.tau:].reshape((-1, 1))
        # print(self.features.shape,self.labels.shape)
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.features, self.labels], train, i)
```


```python
org_data = Data(randn=0)
model = d2l.LinearRegression(lr=0.01)
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, org_data)
onestep_preds = model(org_data.features[org_data.num_train:])
model.loss(y_hat=onestep_preds, y=data.labels[org_data.num_train:]).item()
```




    0.00035995926009491086




    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_14_1.svg)
    



```python
data = CorOldData(x=org_data.x)
model = d2l.LinearRegression(lr=0.01)
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)
onestep_preds = model(data.features[data.num_train:])
model.loss(y_hat=onestep_preds, y=data.labels[data.num_train:]).item()
```




    0.000580643187277019




    
![svg](9_1_6_Exercises_files/9_1_6_Exercises_15_1.svg)
    


## 1.4 Change the neural network architecture and evaluate the performance. You may train the new model with more epochs. What do you observe?


```python
model = d2l.MulMLP(lr=0.01,num_outputs=1, num_hiddens=[2,2])
trainer = d2l.Trainer(max_epochs=5)
trainer.fit(model, data)
onestep_preds = model(data.features[data.num_train:])
model.loss(y_hat=onestep_preds, y=data.labels[data.num_train:]).item()
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[64], line 3
          1 model = d2l.MulMLP(lr=0.01,num_outputs=1, num_hiddens=[2,2])
          2 trainer = d2l.Trainer(max_epochs=5)
    ----> 3 trainer.fit(model, data)
          4 onestep_preds = model(data.features[data.num_train:])
          5 model.loss(y_hat=onestep_preds, y=data.labels[data.num_train:]).item()


    File ~/work/d2l_solutions/notebooks/exercises/d2l_utils/d2l.py:206, in Trainer.fit(self, model, data)
        204 self.val_batch_idx = 0
        205 for i in range(self.max_epochs):
    --> 206     train_loss, valid_loss = self.fit_epoch()
        207     self.epoch += 1
        208 return train_loss, valid_loss


    File ~/work/d2l_solutions/notebooks/exercises/d2l_utils/d2l.py:222, in Trainer.fit_epoch(self)
        218 train_loss, valid_loss = 0, 0
        219 for batch in self.train_dataloader:
        220     # if len(batch[0]) != 32:
        221     #     print(len(batch[0]))
    --> 222     loss = self.model.training_step(self.prepare_batch(batch),
        223                                     plot_flag=self.plot_flag)
        224     # print(f'step train loss:{loss}, T:{self.model.T}')
        225     self.optim.zero_grad()


    File ~/work/d2l_solutions/notebooks/exercises/d2l_utils/d2l.py:331, in Classifier.training_step(self, batch, plot_flag)
        329 # auc = torch.tensor(roc_auc_score(batch[-1].detach().numpy() , y_hat[:,1].detach().numpy()))
        330 if plot_flag:
    --> 331     self.plot('loss', self.loss(y_hat, batch[-1]), train=True)
        332     # self.plot('auc', auc, train=True)
        333     self.plot('acc', self.accuracy(y_hat, batch[-1]), train=True)


    File ~/work/d2l_solutions/notebooks/exercises/d2l_utils/d2l.py:357, in Classifier.loss(self, y_hat, y, averaged)
        355 y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        356 y = y.reshape((-1,))
    --> 357 return F.cross_entropy(y_hat, y, reduction='mean' 
        358                        if averaged else 'none')


    File ~/.local/lib/python3.11/site-packages/torch/nn/functional.py:3029, in cross_entropy(input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing)
       3027 if size_average is not None or reduce is not None:
       3028     reduction = _Reduction.legacy_get_string(size_average, reduce)
    -> 3029 return torch._C._nn.cross_entropy_loss(input, target, weight, _Reduction.get_enum(reduction), ignore_index, label_smoothing)


    RuntimeError: expected scalar type Long but found Float


# 2. An investor wants to find a good security to buy. They look at past returns to decide which one is likely to do well. What could possibly go wrong with this strategy?



# 3. Does causality also apply to text? To which extent?



# 4. Give an example for when a latent autoregressive model might be needed to capture the dynamic of the data.




