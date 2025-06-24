# added by me
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models import model_dict
# ...existing code...
import argparse
import torch
import torch.nn as nn

def print_layer_output_shapes(model, input_shape):
    def register_hook(module):
        def hook(module, input, output):
            class_name = module.__class__.__name__
            module_idx = len(summary)
            m_key = f"{module_idx}_{class_name}"
            summary[m_key] = output.shape
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList) and not (module == model):
            hooks.append(module.register_forward_hook(hook))

    summary = {}
    hooks = []
    model.apply(register_hook)
    dummy_input = torch.randn(*input_shape)
    model(dummy_input)
    for k, v in summary.items():
        print(f"{k}: {v}")
    for h in hooks:
        h.remove()

input_shape = {
    'cifar10':(1, 3, 32, 32),
    'cifar100':(1, 3, 32, 32),
    'imgnet':(1, 3, 224, 224),
}

# 例: ResNet18の構造を表示
# model = resnet18(num_classes=10)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="vgg8", help="モデル名（例: resnet38, vgg8 など）")
    parser.add_argument("--num_classes", type=int, default=10, help="出力クラス数（デフォルト: 10）")
    parser.add_argument("--input_type", type=str, default="cifar10", help="入力タイプ（デフォルト: cifar10）")
    args = parser.parse_args()

    if args.model_name not in model_dict:
        print(f"モデル {args.model_name} は model_dict に存在しません。")
        sys.exit(1)

    model = model_dict[args.model_name](num_classes=args.num_classes)
    print_layer_output_shapes(model, input_shape[args.input_type])

"""
model_dict = {
    'resnet38': resnet38,
    'resnet110': resnet110,
    'resnet116': resnet116,
    'resnet14x2': resnet14x2,
    'resnet38x2': resnet38x2,
    'resnet110x2': resnet110x2,
    'resnet8x4': resnet8x4,
    'resnet14x4': resnet14x4,
    'resnet32x4': resnet32x4,
    'resnet38x4': resnet38x4,
    'vgg8': vgg8_bn,
    'vgg13': vgg13_bn,
    'MobileNetV2': mobile_half,
    'MobileNetV2_1_0': mobile_half_double,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
    'ShuffleV2_1_5': ShuffleV2_1_5,
    
    'ResNet18': resnet18,
    'ResNet34': resnet34,
    'ResNet50': resnet50,
    'resnext50_32x4d': resnext50_32x4d,
    'ResNet10x2': wide_resnet10_2,
    'ResNet18x2': wide_resnet18_2,
    'ResNet34x2': wide_resnet34_2,
    'wrn_50_2': wide_resnet50_2,
    
    'MobileNetV2_Imagenet': mobilenet_v2,
    'ShuffleV2_Imagenet': shufflenet_v2_x1_0,
}
"""