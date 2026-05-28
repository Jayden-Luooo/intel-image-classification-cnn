from torch import nn

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),  # 负责提取特征
            nn.BatchNorm2d(16),              # 负责稳定训练
            nn.ReLU(),                       # 负责增加非线性
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)                  # 负责缩小图片尺寸
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),  # Extracts features using 32 filters while keeping the same image size. 
            nn.BatchNorm2d(32),               # Normalizes the 32 feature maps to make training more stable.
            nn.ReLU(),                        # Adds non-linearity so the model can learn complex patterns.
            nn.MaxPool2d(2),                  # Downsamples the feature maps by 2 to reduce spatial size.
        )
        self.block_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), # Compresses each feature map to one value.
            nn.Flatten(),                 # Converts [batch, 64, 1, 1] to [batch, 64].
            nn.Dropout(p=0.3),            # Reduces overfitting before classification.
            nn.Linear(64 * 4 * 4, num_classes)    # Maps 64 CNN features to class scores.
        )

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        return self.classifier(x)