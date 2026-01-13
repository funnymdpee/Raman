from model.dncnn import DnCNN
from model.fbnet import BigNet,create_BigNet
import torch
from torchinfo import summary


model = create_BigNet()
summary(model, input_size=(2,1,10000))