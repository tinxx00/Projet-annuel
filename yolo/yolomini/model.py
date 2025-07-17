# 📁 yolomini.py
import torch
import torch.nn as nn

class YOLOMini(nn.Module):
    def __init__(self, num_classes=10):
        super(YOLOMini, self).__init__()
        self.num_classes = num_classes

        # Backbone simplifié
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Heads
        self.bbox_head = nn.Linear(64, 4 * 2)  # 2 boxes par image
        self.conf_head = nn.Linear(64, 1 * 2)
        self.class_head = nn.Linear(64, num_classes * 2)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), -1)

        bbox = torch.sigmoid(self.bbox_head(x)).view(x.size(0), 2, 4)
        conf = self.conf_head(x).view(x.size(0), 2)
        cls = self.class_head(x).view(x.size(0), 2, self.num_classes)

        return bbox, conf, cls
