
import os.path
import time
import torch.cuda
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
from load_module import Load
from train_module import Train
from valid_module import Valid
from model import VGG16


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    print("PyTorch 版本： ", torch.__version__)
    print("CUDA 版本： ", torch.version.cuda)
    print("cuDNN 版本： ", torch.backends.cudnn.version())
    print("显卡名称： ", torch.cuda.get_device_name(0))

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    model = VGG16("VGG16", init_weights=True)
    loss_func = nn.CrossEntropyLoss()
    optimizer = opt.SGD(model.parameters(), lr=0.001, momentum=0.9)

    load = Load()
    train_loader, valid_loader = load.load_data("./flowers", "./datasets", 32, 100)

    epochs = 100

    min_loss = 1.0
    max_loss = 0.0

    min_train_acc = 1.0
    max_train_acc = 0.0

    min_valid_acc = 1.0
    max_valid_acc = 0.0

    train = Train()
    valid = Valid()

    Loss = []
    Train_Acc = []
    Valid_Acc = []

    model_path = "./model/VGG16-Flower.pth"
    if not os.path.exists("./model"):
        os.mkdir("./model")

    img_path = "./img/VGG16-Flower.jpg"
    if not os.path.exists("./img"):
        os.mkdir("./img")

    print("start time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    for epoch in range(0, epochs + 1):
        loss, train_acc = train.train_method(model, device, train_loader, optimizer, loss_func, epoch)

        Loss.append(loss)
        Train_Acc.append(train_acc)

        if loss < min_loss:
            min_loss = loss
        if loss > max_loss:
            max_loss = loss

        if train_acc < min_train_acc:
            min_train_acc = train_acc
        if train_acc > max_train_acc:
            max_train_acc = train_acc
            torch.save(model.state_dict(), model_path)

        valid_acc = valid.valid_method(model, device, valid_loader, epoch)

        Valid_Acc.append(valid_acc)

        if valid_acc < min_valid_acc:
            min_valid_acc = valid_acc
        if valid_acc > max_valid_acc:
            max_valid_acc = valid_acc

    print("end time: ", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

    print("Finish Training !")

    plt.figure(figsize=(5, 7))
    plt.subplot(3, 1, 1)
    plt.plot(Loss)
    plt.title("Loss")
    plt.xticks(torch.arange(0, epochs + 1, 10))
    plt.yticks(torch.arange(min_loss, max_loss + 0.3, 0.3))

    plt.subplot(3, 1, 2)
    plt.plot(Train_Acc)
    plt.title("Train_Acc")
    plt.xticks(torch.arange(0, epochs + 1, 10))
    plt.yticks(torch.arange(min_train_acc, max_train_acc, 0.1))
    plt.xlabel("best train acc: " + str(max_train_acc))

    plt.subplot(3, 1, 3)
    plt.plot(Valid_Acc)
    plt.title("Valid_Acc")
    plt.xticks(torch.arange(0, epochs + 1, 10))
    plt.yticks(torch.arange(min_valid_acc, max_valid_acc, 0.1))
    plt.xlabel("best valid acc: " + str(max_valid_acc))

    # plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.35)
    plt.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.1, hspace=0.5)
    plt.savefig(img_path)
    plt.show()


if __name__ == "__main__":
    main()
