import torch
import time
# If a GPU is available, use it
if torch.cuda.is_available():
    device = torch.device("cuda")
    use_cuda = True
    print('Using cuda !')
else:
    device = torch.device("cpu")
    use_cuda = False
    print('GPU not available !')

import numpy as np
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
class CustomDataset(Dataset):

    def __init__(self, data, label, transform=None):
        self.data = data
        self.data_len = len(self.data)
        self.label_arr = label

    def __getitem__(self, index):
        label = int(self.label_arr[index])
       # print(f"Data shape before reshape: {self.data[index].shape}")

        sample = (self.data[index]).reshape(1, 20, 4001)
        sample = torch.from_numpy(sample)
        return sample, label

    def __len__(self):
        return self.data_len


def EEGdata_loader():
    x_train = np.load('x0.npy')
    x_test = np.load('x0_test.npy')
    x_valid = np.load('x0_test.npy')
    y_train = np.load("y1.npy")
    y_test = np.load("y1_test.npy")
    y_valid = np.load("y1_test.npy")

    train_data = CustomDataset(x_train, y_train)
    valid_data = CustomDataset(x_valid, y_valid)
    test_data = CustomDataset(x_test, y_test)

    train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=use_cuda, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=batch_size_eval, pin_memory=use_cuda)
    test_loader = DataLoader(test_data, batch_size=batch_size_eval, pin_memory=use_cuda)

    return train_loader, valid_loader, test_loader


from attention import inference


def train(model_global, model_local, model_top, optimizer, loss_fn_local_top, epoch, only_global_model):
    model_global.train()
    model_local.train()
    model_top.train()

    for batch_idx, (inputs, target) in enumerate(train_loader):
        inputs, target = inputs.to(device), target.to(device)
        inputs = inputs.float()
        target = target.long()
        optimizer.zero_grad()

        wpser = inputs[:, :, :, -1]  # WPSER corresponding to each channel
        inputs = inputs[:, :, :, 0:inputs.shape[3] - 1]  # raw EEG signal
        output_merged, hint_loss, channel_loss = inference(inputs, wpser, model_global, model_local, model_top,
                                                           n_slices, device,
                                                           only_global_model, is_training=True)

        loss_local_and_top = loss_fn_local_top(output_merged, target)
        loss_global_model = loss_local_and_top + hint_loss + channel_loss

        for param in model_local.parameters():
            param.requires_grad = False
        for param in model_top.parameters():
            param.requires_grad = False
        loss_global_model.backward(retain_graph=True)
        for param in model_local.parameters():
            param.requires_grad = True
        for param in model_top.parameters():
            param.requires_grad = True
        for param in model_global.parameters():
            param.requires_grad = False
        loss_local_and_top.backward()
        for param in model_global.parameters():
            param.requires_grad = True

        optimizer.step()

    if epoch % 10 == 0:
        print('\rTrain Epoch: {}'
              '  Total_Loss: {:.4f} (CrossEntropy: {:.2f} Hint: {:.2f} Ch: {:.2f})'
              ''.format(epoch, loss_local_and_top.item() + hint_loss.item(), loss_local_and_top.item(),
                        hint_loss.item(), channel_loss.item()),
              end='')

    return loss_local_and_top.item() + hint_loss.item() + channel_loss.item(), loss_local_and_top.item()


def test(model_global, model_local, model_top, test_loss_fn_local_top, epoch, loader, only_global_model):
    model_global.eval()
    model_local.eval()
    model_top.eval()

    avg_test_loss, avg_hint_loss, avg_channel_loss = 0, 0, 0
    correct = 0
    test_size = 0

    with torch.no_grad():
        for inputs, target in loader:
            inputs, target = inputs.to(device), target.to(device)

            inputs = inputs.float()
            target = target.long()

            wpser = inputs[:, :, :, -1]
            inputs = inputs[:, :, :, 0:inputs.shape[3] - 1]

            output_merged, hint_loss, channel_loss = inference(inputs, wpser, model_global, model_local, model_top,
                                                               n_slices, device,
                                                               only_global_model, is_training=False)

            test_size += len(inputs)
            avg_test_loss += test_loss_fn_local_top(output_merged, target).item()
            avg_hint_loss += len(inputs) * hint_loss.item()
            avg_channel_loss += len(inputs) * channel_loss.item()
            pred = output_merged.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_test_loss /= test_size
    avg_hint_loss /= test_size
    avg_channel_loss /= test_size
    accuracy = correct / test_size

    if epoch % 10 == 0:
        print('\nTest set: Avg_Total_Loss: {:.4f} (CrossEntropy: {:.2f} Hint: {:.2f} Ch: {:.2f})'
              '  Accuracy: {}/{} ({:.0f}%)\n'
              .format(avg_test_loss + avg_hint_loss + avg_channel_loss, avg_test_loss, avg_hint_loss, avg_channel_loss,
                      correct, test_size, 100. * accuracy))

    return avg_test_loss + avg_hint_loss + avg_channel_loss, avg_test_loss, accuracy


import torch.nn as nn
import torch.optim as optim
from network import globalnetwork, localnetwork, topnetwork

n_slices = 1  # number of time slices

n_epochs = 50
loss_fn_local_top = nn.NLLLoss()
test_loss_fn_local_top = nn.NLLLoss(reduction='sum')
learning_rate = 0.0625 * 0.01
batch_size = 8
batch_size_eval = 1

train_loader, valid_loader, test_loader = EEGdata_loader()
model_global = globalnetwork().to(device)
model_local = localnetwork().to(device)
model_top = topnetwork().to(device)

only_global_model = True  # only use global model

if only_global_model:
    optimizer = optim.Adam(list(model_global.parameters())
                           + list(model_top.parameters()), lr=learning_rate)
else:
    optimizer = optim.Adam(list(model_global.parameters())
                           + list(model_local.parameters())
                           + list(model_top.parameters()), lr=learning_rate)

min_cross_entropy = 100000

for ep in range(n_epochs):
    train_total_loss, train_cross_entropy = train(model_global, model_local, model_top, optimizer,
                                                  loss_fn_local_top, ep, only_global_model)
    valid_total_loss, valid_cross_entropy, valid_acc = test(model_global, model_local, model_top,
                                                            test_loss_fn_local_top, ep, valid_loader, only_global_model)
    if valid_cross_entropy < min_cross_entropy:
        min_cross_entropy = valid_cross_entropy
        torch.save(model_global.state_dict(), 'model_global_cross_entropy.pth')
        torch.save(model_local.state_dict(), 'model_local_cross_entropy.pth')
        torch.save(model_top.state_dict(), 'model_top_cross_entropy.pth')

if only_global_model:
    print('\nUse global model:')
else:
    print('\nUse TCACNet:')

model_global.load_state_dict(torch.load('model_global_cross_entropy.pth'))
model_local.load_state_dict(torch.load('model_local_cross_entropy.pth'))
model_top.load_state_dict(torch.load('model_top_cross_entropy.pth'))
start_time=time.time()
valid_total_loss, valid_cross_entropy, valid_acc = test(model_global, model_local, model_top,
                                                        test_loss_fn_local_top, 0, test_loader, only_global_model)
end_time=time.time()
time=end_time-start_time
print(time/11)