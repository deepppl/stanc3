
import torch
from torch import nn
from torch.nn import functional as F
from torch import tensor
import pyro
from pyro import distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
from torch.autograd import Variable

import deepppl
import os

from .util import loadData


batch_size, nx, nh, ny = 128, 28 * 28, 1024, 10
def build_mlp():
    # Model

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.l1 = torch.nn.Linear(nx, nh)
            self.l2 = torch.nn.Linear(nh, ny)
            self.relu = torch.nn.ReLU()

        def forward(self, x):
            h = self.relu(self.l1(x.view((-1, nx))))
            yhat = self.l2(h)
            return F.log_softmax(yhat, dim=-1)

    return MLP()


def predict(data, posterior):
    predictions = [model(data) for model in posterior]
    prediction = torch.stack(predictions).mean(dim=0)
    return prediction.argmax(dim=-1)

def test_mlp_inference():
    mlp = build_mlp()
    train_loader, test_loader = loadData(batch_size)
    model = deepppl.PyroModel(model_file = 'deepppl/tests/good/mlp.stan', mlp=mlp)
    svi = model.svi(params = {'lr' : 0.01})

    for epoch in range(2):  # loop over the dataset multiple times
        for j, (imgs, lbls) in enumerate(train_loader, 0):
            # calculate the loss and take a gradient step
            loss = svi.step(batch_size, imgs, lbls)
            print('.', end="", flush=True)
    posterior = svi.posterior(30)

    for j, data in enumerate(test_loader):
        images, labels = data
        accuracy = (predict(images, posterior) == labels).type(torch.float).mean()
        assert accuracy > 0.6

    print('accuracy:', accuracy)

if __name__ == "__main__":
    test_mlp_inference()
