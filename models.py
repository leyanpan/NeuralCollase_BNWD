import torch
import torchvision.models as models
import torch.nn as nn
from mlp import MLP

def load_vgg_model(model_type, bn=False, num_classes=10, input_channels=3):
    if model_type == 'vgg11':
        if bn:
            model = models.vgg11_bn(pretrained=False, num_classes=num_classes)
        else:
            model = models.vgg11(pretrained=False, num_classes=num_classes)
    elif model_type == 'vgg13':
        if bn:
            model = models.vgg13_bn(pretrained=False, num_classes=num_classes)
        else:
            model = models.vgg13(pretrained=False, num_classes=num_classes)
    elif model_type == 'vgg16':
        if bn:
            model = models.vgg16_bn(pretrained=False, num_classes=num_classes)
        else:
            model = models.vgg16(pretrained=False, num_classes=num_classes)
    elif model_type == 'vgg19':
        if bn:
            model = models.vgg19_bn(pretrained=False, num_classes=num_classes)
        else:
            model = models.vgg19(pretrained=False, num_classes=num_classes)
    else:
        raise ValueError('Invalid VGG model type: ' + model_type)

    # Modify the first layer to accept the specified number of input channels
    model.features[0] = torch.nn.Conv2d(input_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    # Swap the dropout layers with batch normalization in the final classifier when bn=True
    if bn:
        for m in model.classifier.modules():
            if isinstance(m, torch.nn.Dropout):
                m = torch.nn.BatchNorm1d(4096)

    # Get a list of convolutional layers in the features network and all linear layers in the classifier network
    modules = []
    for m in model.features.modules():
        if isinstance(m, torch.nn.Conv2d):
            modules.append(m)
    for m in model.classifier.modules():
        if isinstance(m, torch.nn.Linear):
            modules.append(m)

    return model, modules

def get_model(model_type, num_classes, in_channels, device, args):
  hooked_modules = []
  if model_type == 'ResNet':
    resnet = models.resnet18(pretrained=False, num_classes=num_classes)
    resnet.conv1 = nn.Conv2d(in_channels, resnet.conv1.weight.shape[0], 3, 1, 1, bias=False) # Small dataset filter size used by He et al. (2015)
    resnet.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0)
    if args.bn:
      num_features = resnet.fc.in_features
      model = nn.Sequential(
        resnet.conv1,
        resnet.bn1,
        resnet.relu,
        resnet.maxpool,
        resnet.layer1,
        resnet.layer2,
        resnet.layer3,
        resnet.layer4,
        resnet.avgpool,  # Replace the previous average pooling
        nn.Flatten(),
        nn.BatchNorm1d(num_features),  # Batch normalization layer
        resnet.fc)
    else:
      model = resnet
    # register hook that saves last-layer input into features
    classifier = resnet.fc
    if args.early_layers:
        hooked_modules = [resnet.conv1, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4, resnet.fc]
    else:
        hooked_modules = [resnet.fc]

  elif model_type == 'MLP':
    depths = layer_width = [args.input_dim] + [args.hidden_layer_width] * (args.model_depth_MLP - 1) + [num_classes]
    C = num_classes
    model = MLP(layer_width, bn=args.bn, weight_norm=args.weight_norm)
    classifier = model.last_layer
    if args.early_layers:
        for m in model.feature:
            if isinstance(m, nn.Linear):
                hooked_modules += [m]
            hooked_modules += [model.last_layer]
    else:
        hooked_modules = [model.last_layer]

  elif model_type.startswith('vgg'):
    model, modules = load_vgg_model(model_type, args.bn, num_classes, in_channels)
    classifier = model.classifier[-1]
    if args.early_layers:
        hooked_modules = modules
    else:
        hooked_modules = [classifier]

  model = model.to(device)

  return model, classifier, hooked_modules