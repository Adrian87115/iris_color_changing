import torch
import torch.nn as nn
import torch.nn.functional as f

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.in_channels = 1
        self.out_channels = 1
        self.num_features = 20
        self.encoder1 = self._block(self.in_channels, self.num_features)
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder2 = self._block(self.num_features, self.num_features * 2)
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder3 = self._block(self.num_features * 2, self.num_features * 4)
        self.max_pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder4 = self._block(self.num_features * 4, self.num_features * 8)
        self.max_pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.bottleneck = self._block(self.num_features * 8, self.num_features * 16)
        self.up_conv4 = nn.ConvTranspose2d(self.num_features * 16, self.num_features * 8, kernel_size = 2, stride = 2)
        self.decoder4 = self._block(self.num_features * 16, self.num_features * 8)
        self.up_conv3 = nn.ConvTranspose2d(self.num_features * 8, self.num_features * 4, kernel_size = 2, stride = 2)
        self.decoder3 = self._block(self.num_features * 8, self.num_features * 4)
        self.up_conv2 = nn.ConvTranspose2d(self.num_features * 4, self.num_features * 2, kernel_size = 2, stride = 2)
        self.decoder2 = self._block(self.num_features * 4, self.num_features * 2)
        self.up_conv1 = nn.ConvTranspose2d(self.num_features * 2, self.num_features, kernel_size = 2, stride = 2)
        self.decoder1 = self._block(self.num_features * 2, self.num_features)
        self.conv = nn.Conv2d(in_channels = self.num_features, out_channels = self.out_channels, kernel_size = 1)

    @staticmethod
    def _block(in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels = in_channels,
                      out_channels = out_channels,
                      kernel_size = 3,
                      padding = 1,
                      bias = False),
            nn.ReLU(inplace = True),
            nn.Conv2d(in_channels = out_channels,
                      out_channels = out_channels,
                      kernel_size = 3,
                      padding = 1,
                      bias = False),
            nn.ReLU(inplace=True))

    def crop(self, enc, dec):
        _, _, h, w = dec.size()
        enc = f.interpolate(enc, size = (h, w), mode = "bilinear", align_corners = False)
        return enc

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.max_pool1(enc1))
        enc3 = self.encoder3(self.max_pool2(enc2))
        enc4 = self.encoder4(self.max_pool3(enc3))
        bottleneck = self.bottleneck(self.max_pool4(enc4))
        dec4 = self.up_conv4(bottleneck)
        dec4 = torch.cat((dec4, self.crop(enc4, dec4)), dim = 1)
        dec4 = self.decoder4(dec4)
        dec3 = self.up_conv3(dec4)
        dec3 = torch.cat((dec3, self.crop(enc3, dec3)), dim = 1)
        dec3 = self.decoder3(dec3)
        dec2 = self.up_conv2(dec3)
        dec2 = torch.cat((dec2, self.crop(enc2, dec2)), dim = 1)
        dec2 = self.decoder2(dec2)
        dec1 = self.up_conv1(dec2)
        dec1 = torch.cat((dec1, self.crop(enc1, dec1)), dim = 1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))