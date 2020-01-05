import time
import torch
import torchvision
from torch.nn import functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

class Config(object):

    def __init__(self):
        self.torch_seed = 666013

        self.save_path = './checkpoint_cnn' + '.pt'
        self.batch_size = 200
        self.eval_batch_size = 100
        self.num_workers = 4

        self.max_epoch = 17

        self.initial_lr = 0.01
        self.device='cpu'
        self.lambd=0.01


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def train(model, optimizer, loader,cfg):
    model.train()
    loss_sum = 0
    acc_sum = 0

    for idx, (data, target) in enumerate(loader):
        data = data.view(data.shape[0], -1)
        data, target = data.to(cfg.device), target.to(cfg.device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss_sum += loss.item()
        loss.backward()
        optimizer.step()

        predict = output.argmax(1)
        acc = (predict==target).sum().cpu().item()
        acc_sum += acc

    return loss_sum / len(loader), acc_sum / len(loader)/ float(cfg.batch_size)


def evaluate(model, loader, cfg):
    model.eval()
    loss_sum = 0
    acc_sum = 0
    for idx, (data, target) in enumerate(loader):
        data = data.view(data.shape[0], -1)
        data, target = data.to(cfg.device), target.to(cfg.device)

        output = model(data)
        loss = F.cross_entropy(output, target)
        loss_sum += loss.item()

        predict = output.argmax(1)
        acc = (predict==target).sum().cpu().item()
        acc_sum += acc
    return loss_sum / len(loader), acc_sum / len(loader)/ float(cfg.eval_batch_size)



if __name__ == '__main__':
    cfg = Config()
    torch.manual_seed(cfg.torch_seed)

    # Load MNIST data
    mnist_train = torchvision.datasets.MNIST('data',
                                             train=True,
                                             transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
                                             download=True)

    mnist_test = torchvision.datasets.MNIST('data', train=False,
                                            transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor()]),
                                            download=True)

    # Sample the training data to be the same size as logistic regression
    train_set_size = 5000
    test_set_size = 10000
    mnist_train, _ = torch.utils.data.random_split(mnist_train, [train_set_size, len(mnist_train) - train_set_size])

    assert len(mnist_train) == train_set_size and len(mnist_test) == test_set_size

    train_iter = DataLoader(mnist_train, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    test_iter = DataLoader(mnist_test, cfg.eval_batch_size, shuffle=False)
    model = Net().to(cfg.device)
    optimizer = optim.Adam(model.parameters(), cfg.initial_lr)

    # Train NN and report results
    total_time = time.time()
    for epoch in range(cfg.max_epoch):
        print('\nEpoch {:3d}/{:3d}'.format(epoch, cfg.max_epoch))
        start_time = time.time()

        train_loss, train_acc = train(model, optimizer, train_iter, cfg)
        test_loss, test_acc = evaluate(model, test_iter, cfg)
        print('\tTrain loss: {:.3f}, accuracy: {:.1f}%'.format(train_loss, round(train_acc * 100, 1)))
        print('\tTest loss:  {:.3f}, accuracy: {:.1f}%'.format(test_loss, round(test_acc * 100, 1)))

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(), cfg.save_path)
        end_time = time.time()
        print('\tTime: {:.2f}s'.format(end_time - start_time))

    total_time_end = time.time()
    test_loss, test_acc = evaluate(model, test_iter, cfg)
    print('\n\n----- Total train time = {:.2f}s. Final test loss:  {:.3f}, final test accuracy: {:.1f}%'
                                        .format(total_time_end - total_time, test_loss, round(test_acc * 100, 1)))
