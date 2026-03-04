# feature_extractor.py
import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=10, dropout=0.2):
        super().__init__()
        self.model_name = model_name

        if 'efficientnet' in model_name:
            backbone = getattr(models, model_name)(pretrained=True)
            self.features = backbone.features
            self.classifier_in_features = backbone.classifier[1].in_features
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.classifier_in_features, num_classes)
            )
            # 定义要提取的层名（对应 features 的索引）
            self.extract_layers = {
                'feat2': 2,
                'feat4': 4,
                'feat6': 6,
                'feat8': 8
            }
        elif 'resnet' in model_name:
            backbone = getattr(models, model_name)(pretrained=True)
            self.conv1 = backbone.conv1
            self.bn1 = backbone.bn1
            self.relu = backbone.relu
            self.maxpool = backbone.maxpool
            self.layer1 = backbone.layer1
            self.layer2 = backbone.layer2
            self.layer3 = backbone.layer3
            self.layer4 = backbone.layer4
            self.classifier_in_features = backbone.fc.in_features
            self.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.classifier_in_features, num_classes)
            )
            self.extract_layers = {
                'layer2': 'layer2',
                'layer3': 'layer3',
                'layer4': 'layer4'
            }
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        self.activations = {}

    def forward(self, x):
        self.activations.clear()

        if 'efficientnet' in self.model_name:
            x = self.features[0](x)  # stem
            for i in range(1, len(self.features)):
                x = self.features[i](x)
                if i in self.extract_layers.values():
                    # 反查 layer name
                    for name, idx in self.extract_layers.items():
                        if idx == i:
                            self.activations[name] = x
        else:  # ResNet
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x); self.activations['layer2'] = x
            x = self.layer3(x); self.activations['layer3'] = x
            x = self.layer4(x); self.activations['layer4'] = x

        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        logits = self.classifier(x)
        return logits