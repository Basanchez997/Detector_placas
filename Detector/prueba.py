import torch
torch.cuda.is_available()
X_train = torch.FloatTensor([0., 1., 2.])
X_train.is_cuda
False
X_train = X_train.to(device)
X_train.is_cuda
True
model = MyModel(args)
model.to(device)