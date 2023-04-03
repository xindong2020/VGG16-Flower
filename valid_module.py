
import sys
import torch
import torch.utils.data as data
from tqdm import tqdm
from model import VGG16


class Valid:
    def valid_method(self, model: VGG16, device: torch.device, valid_loader: data.DataLoader, epoch):

        model.eval()
        model = model.to(device)

        total = 0.0
        correct = 0.0

        with torch.no_grad():
            valid_bar = tqdm(valid_loader, file=sys.stdout)
            for step, data in enumerate(valid_bar):
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                predicts = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += torch.eq(predicts, labels).sum().item()
                valid_bar.desc = "Valid Epoch: {:d}, Acc: {:.3f}".format(epoch, 100 * (correct / total))

        return round(correct / total, 3)
