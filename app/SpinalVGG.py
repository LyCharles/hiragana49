import torch
import torch.nn as nn
import torch.nn.functional as F

#Same Model as Used During Training, for Loading When Making Predictions on the Web Server

Half_width = 128
layer_width = 128
class SpinalVGG(nn.Module):
    def two_conv_pool(self, in_channels, f1, f2):
        return nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def three_conv_pool(self, in_channels, f1, f2, f3):
        return nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def __init__(self, num_classes=49):
        super(SpinalVGG, self).__init__()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)

        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True))
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True))
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True))
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(Half_width + layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True))
        self.fc_out = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(layer_width * 4, num_classes))

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)

        x1 = self.fc_spinal_layer1(x[:, 0:Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([x[:, Half_width:2*Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([x[:, 0:Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([x[:, Half_width:2*Half_width], x3], dim=1))

        x = torch.cat([x1, x2, x3, x4], dim=1)
        x = self.fc_out(x)
        return F.log_softmax(x, dim=1)