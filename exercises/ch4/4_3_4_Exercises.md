```python
import torch
import warnings
import sys
sys.path.append('/home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/')
import d2l
warnings.filterwarnings('ignore')


class Classifier(d2l.Module):
    def validation_step(self, batch):
        y_hat = self(*batch[:-1])
        self.plot('loss', self.loss(y_hat, batch[-1]), train=False)
        self.plot('acc', self.accuracy(y_hat, batch[-1]), train=False)
        
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)
    
    def accuracy(self, y_hat, y, averaged=True):
        y_hat = y_hat.reshape((-1, y_hat.shape[-1]))
        preds = y_hat.argmax(axis=1).type(y.dtype)
        comp = (preds == y.reshape(-1)).type(torch.float32)
        return comp.mean if averaged else comp
```

    /home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/d2l.py:119: SyntaxWarning: assertion is always true, perhaps remove parentheses?
      assert(self, 'net'), 'Neural network is defined'
    /home/jovyan/work/d2l_solutions/notebooks/exercises/d2l_utils/d2l.py:123: SyntaxWarning: assertion is always true, perhaps remove parentheses?
      assert(self, 'trainer'), 'trainer is not inited'


# 4.3.4. Exercises

## 1. Denote by $L_v$ the validation loss, and let $L_v^q$ be its quick and dirty estimate computed by the loss function averaging in this section. Lastly, denote by $l_v^b$ the loss on the last minibatch. Express $L_v$ in terms of $L_v^q$, $l_v^b$, and the sample and minibatch sizes.

We assume that the validation dataset is split into $N$ samples, and each minibatch contains $M$ samples.
The quick and dirty estimate $L_v^q$ is computed by averaging the loss computed on each minibatch. Since there are $N$ samples in total, and each minibatch contains $M$ samples, there are $N/M$ minibatches in total.
Now, let's express $L_v$ in terms of $L_v^q$, $l_v^b$, $N$, and $M$:
$L_v$ is the true validation loss, and it can be considered as an average of the batch losses:
$$L_v = \frac{M}{N} \sum_{i=1}^{N/M}l_v^q$$

## 2. Show that the quick and dirty estimate $L_v^q$ is unbiased. That is, show that $E[L_v]=E[L_v^q]$. Why would you still want to use $L_v$ instead?

$$E[L_v] = E[\frac{M}{N} \sum_{i=1}^{N/M}l_v^q]==\frac{M}{N}\sum_{i=1}^{N/M}E[l_v^q]=E[l_v^q]$$

## 3. Given a multiclass classification loss, denoting by $l(y,y^\prime)$ the penalty of estimating $y^\prime$ when we see $y$ and given a probabilty $p(y|x)$, formulate the rule for an optimal selection of $y^\prime$.
Hint: express the expected loss, using $l$ and $p(y|x)$.

The optimal selection of $y^\prime$ in a multiclass classification scenario can be formulated using the concept of expected loss. Given a true class $y$ and a predicted class $y^\prime$, and assuming that $p(y|x)$ represents the probability of observing class $y$ given input $x$, the expected loss can be used to guide the decision-making process.
The expected loss $\mathbb{E}[l(y, y^\prime)]$ is the average loss that we expect to incur when predicting $y^\prime$ while the true class is $y$. To minimize the expected loss, we need to select the $y^\prime$ that minimizes this average.
The optimal selection of $y^\prime$ can be formulated as follows:
$$y^\prime = \arg\min_{\text{all possible } y^\prime} \sum_{y} p(y|x) \cdot l(y, y^\prime)$$
