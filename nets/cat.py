import torch
import torch.nn as nn

class CBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FeatureFusionNet(nn.Module):
    def __init__(self):
        super(FeatureFusionNet, self).__init__()

        # Define CBR operations for each input
        self.cbr_up1 = nn.Sequential(
            CBR(64, 64),
            CBR(64, 64)
        )
        self.cbr_up2 = nn.Sequential(
            CBR(128, 128),
            CBR(128, 128)
        )
        self.cbr_up3 = nn.Sequential(
            CBR(256, 256),
            CBR(256, 256)
        )
        self.cbr_up4 = nn.Sequential(
            CBR(512, 512),
            CBR(512, 512)
        )

        # Define upsampling operations
        self.upsample_up2 = nn.Upsample(scale_factor=(2, 2), mode='bilinear', align_corners=True)  # Adjusted upsampling factor
        self.upsample_up3 = nn.Upsample(scale_factor=(4, 4), mode='bilinear', align_corners=True)  # Adjusted upsampling factor
        self.upsample_up4 = nn.Upsample(scale_factor=(8, 8), mode='bilinear', align_corners=True)  # Adjusted upsampling factor

        # Define the final CBR operation
        self.final_cbr = CBR(64 + 128 +  256+512, 64)

    def forward(self, up1, up2, up3, up4):
        # Apply CBR operations for each input
        up1_cbr = self.cbr_up1(up1)
        up2_cbr = self.cbr_up2(up2)
        up3_cbr = self.cbr_up3(up3)
        up4_cbr = self.cbr_up4(up4)

        # Upsample all inputs except the first one
        up2_cbr_up = self.upsample_up2(up2_cbr)
        up3_cbr_up = self.upsample_up3(up3_cbr)
        up4_cbr_up = self.upsample_up4(up4_cbr)

        # Feature fusion: concatenate feature maps along the channel dimension
        fused_features = torch.cat((up1_cbr, up2_cbr_up, up3_cbr_up, up4_cbr_up), dim=1)

        # Apply the final CBR operation
        output = self.final_cbr(fused_features)

        return output

# Instantiate the network
up1 = torch.randn(2, 64, 256, 256)
up2 = torch.randn(2, 128, 128, 128)
up3 = torch.randn(2, 256, 64, 64)
up4 = torch.randn(2, 512, 32, 32)
net = FeatureFusionNet()
output = net(up1, up2, up3, up4)
# Print the output shape
print(output.shape)
