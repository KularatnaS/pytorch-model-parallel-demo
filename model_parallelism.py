import torch
import torch.nn as nn
from dataset import train_loader, test_loader
from tqdm import tqdm


# Model
class SimpleNet(nn.Module):
    def __init__(self, hidden_size_1=500, hidden_size_2=1000):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(28*28, hidden_size_1).to('cuda:0')
        self.linear2 = nn.Linear(hidden_size_1, hidden_size_2).to('cuda:0')
        self.linear3 = nn.Linear(hidden_size_2, 10).to('cuda:1')
        self.relu = nn.ReLU()

    def forward(self, img):
        x = img.view(-1, 28*28).to('cuda:0')
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = x.to('cuda:1')
        x = self.linear3(x)
        return x


net = SimpleNet()


def train(train_loader, net, epochs=5, total_iterations_limit=None):
    cross_el = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

    total_iterations = 0

    for epoch in range(epochs):
        net.train()

        loss_sum = 0
        num_iterations = 0

        data_iterator = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        if total_iterations_limit is not None:
            data_iterator.total = total_iterations_limit
        for data in data_iterator:
            num_iterations += 1
            total_iterations += 1
            x, y = data
            y = y.to('cuda:1')
            optimizer.zero_grad()
            output = net(x.view(-1, 28*28))
            loss = cross_el(output, y)
            loss_sum += loss.item()
            avg_loss = loss_sum / num_iterations
            data_iterator.set_postfix(loss=avg_loss)
            loss.backward()
            optimizer.step()

            if total_iterations_limit is not None and total_iterations >= total_iterations_limit:
                return


def test():
    correct = 0
    total = 0

    wrong_counts = [0 for i in range(10)]

    with torch.no_grad():
        for data in tqdm(test_loader, desc='Testing'):
            x, y = data
            y = y.to('cuda:1')
            output = net(x.view(-1, 28*28))
            for idx, i in enumerate(output):
                if torch.argmax(i) == y[idx]:
                    correct += 1
                else:
                    wrong_counts[y[idx]] += 1
                total += 1
    print(f'Accuracy: {round(correct/total, 3)}')
    for i in range(len(wrong_counts)):
        print(f'wrong counts for the digit {i}: {wrong_counts[i]}')


if __name__ == '__main__':
    train(train_loader, net, epochs=5)
    test()