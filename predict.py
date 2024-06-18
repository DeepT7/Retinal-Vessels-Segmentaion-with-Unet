import os
import cv2
import time
import random
import numpy as np

import torch 
import torch.nn as nn 

# Build Conv_layer
class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x
# Build Encoder section
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p
    
# Build Decoder section
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x
# Build Unet architecture
class build_unet(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        # Bottleneck
        self.b = conv_block(512, 1024)

        # Decoder
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        # Classifier
        self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        
    def forward(self, inputs):
        # Encoder
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        # Bottleneck
        b = self.b(p4)

        # Decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        outputs = self.outputs(d4)

        return outputs
    
def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

checkpoint_path = 'results/augment/checkpoint.pth'
image_path = 'test/image/9.png'

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = build_unet()
# model = model.to(device)
# model.load_state_dict(torch.load(checkpoint_path, map_location=device))
# model.eval()

# image = cv2.imread(image_path, cv2.IMREAD_COLOR)
# x = np.transpose(image, (2, 0, 1))
# x = x/255.0
# x = np.expand_dims(x, axis = 0)
# x = x.astype(np.float32)
# x = torch.from_numpy(x)
# x = x.to(device)

# with torch.no_grad():
#     # Predict
#     y_pred = model(x)
#     y_pred = torch.sigmoid(y_pred)
#     y_pred = y_pred[0].cpu().numpy() 
#     y_pred = np.squeeze(y_pred, axis = 0)
#     y_pred = y_pred > 0.5 
#     y_pred = np.array(y_pred, dtype=np.uint8)

# y_pred = mask_parse(y_pred)
# y_pred = y_pred*255

# cv2.imshow('Ouput', y_pred)
# cv2.waitKey(0)
# cv2.destroyAllWindows()