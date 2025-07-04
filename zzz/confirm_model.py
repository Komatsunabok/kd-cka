import torch

from models import resnet

# モデルのパス
model_path = r".\save\teachers\models\resnet32x4_vanilla\resnet32x4_best.pth"

# ファイルのロード
try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    print("Model checkpoint loaded successfully!")
    print("Checkpoint keys:", checkpoint.keys())
except Exception as e:
    print(f"Failed to load model checkpoint: {e}")

"""
Model checkpoint loaded successfully!
Checkpoint keys: dict_keys(['epoch', 'model', 'accuracy', 'optimizer'])
"""

# ResNet-32x4の定義
class ResNet32x4(resnet.ResNet):
    def __init__(self):
        # ResNet-32x4の深さを明示的に指定
        depth = 32  # ResNet-32の深さ
        num_blocks = (depth - 2) // 6  # BasicBlockの数を計算
        super(ResNet32x4, self).__init__(resnet.BasicBlock, [num_blocks] * 4)
        self.fc = torch.nn.Linear(512 * 4, 10)  # CIFAR-10用
# モデルをインスタンス化
model = ResNet32x4()

# PyTorchでロード可能か確認
try:
    model.load_state_dict(checkpoint['state_dict'])
    print("Model weights loaded successfully!")
except Exception as e:
    print(f"Failed to load model weights: {e}")

# モデルの構造を確認
print(model)

# テスト
# ダミーデータを使ったモデルの確認
dummy_input = torch.randn(1, 3, 32, 32)  # CIFAR-10の入力サイズ
output = model(dummy_input)
print("Model output shape:", output.shape)

