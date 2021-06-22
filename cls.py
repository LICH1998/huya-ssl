import torch
import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch.utils.data as data_utils

#手写数字识别的数据在toechvision的包中
#data
#存放地址  训练集标识  转变为Tensor 下载
train_data = dataset.MNIST(root="minist",
                           train=True,
                           transform=transforms.ToTensor,
                           download=True)

test_data = dataset.MNIST(root="minist",
                           train=False,
                           transform=transforms.ToTensor,
                           download=False)
#batchsize  每次丢进去一个小的训练集 dataloader每次从dataset中取batchsize的数据
#shuffle是将数据集dataset打乱再取，这样防止数据次序带来的问题
train_loader = data_utils.DataLoader(dataset=train_data,
                                     batch_size=64,
                                     shuffle=True)
test_loader = data_utils.DataLoader(dataset=test_data,
                                     batch_size=64,
                                     shuffle=True)

#net
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #定义卷积层
        self.conv = torch.nn.Sequential(
            #灰度图输入为1输出为32
            torch.nn.Conv2d(1, 32, kernel_size=5, padding=32),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        #定义线性层 28*28经过Maxpool变成14*14 再加conv2d变为14 * 14 * 32，输出是0-9判别为10维
        self.fc = torch.nn.Linear(14 * 14 * 32, 10)

    def forward(self, x):
        out = self.conv(x)
        #经过卷积层变为n*c*h*w的四维向量儿fc的输入向量为一维，因此将out拉成一维向量
        out = out.view(out.size()[0], -1)
        out = self.fc(out)
        return out

cnn = CNN()
cnn = cnn.cuda()
#loss 分类问题采用交叉熵
loss_func = torch.nn.CrossEntropyLoss()
#optimizer  Adam 傻瓜，适用于数据统计特性不好，误差曲面复杂，速度快
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.01)
#training  enumerate用于一个可遍历的数据对象
for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        images = images.cuda()
        labels = labels.cuda()

        outputs = cnn(images)
        loss = loss_func(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("epoch is {}, ite is {}/{},"
          " loss is {}".format(epoch+1, i, len(train_data) // 64,
                                          loss.item()))

    #test
    loss_test = 0
    accuracy = 0
    for i, (images, labels) in enumerate(test_loader):
        images = images.cuda()
        labels = labels.cuda()
        outputs = cnn(images)
        # [batchsize]
        # outputs = batchsize * cls_num
        loss_test += loss_func(outputs, labels)
        #torch.max(a,1) 返回每一行中最大值的那个元素，且返回其索引
        _, pred = outputs.max(1)
        accuracy += (pred == labels).sum().item()

    accuracy = accuracy / len(test_data)
    loss_test = loss_test / (len(test_data) // 64)

    print("epoch is {}, accuracy is {}, "
          "loss test is {}".format(epoch + 1,
                                   accuracy,
                                   loss_test.item()))
#save
torch.save(cnn, "model/mnist_model.pkl")
#load

#inference