# import torch
# import torchvision
# print(torch.__version__)
# print(torchvision.__version__)


from ultralytics.utils.checks import parse_requirements

print(parse_requirements(package='ultralytics'))