import json
import matplotlib.pyplot as plt

# train_acc_history.jsonを読み込む
with open("../save/teachers/models/resnet32x4_vanilla_cifar10_trial_0_epochs_240_bs_64/train_acc_history.json", "r") as f:
    data = json.load(f)

train_acc = data["train_acc"]

# グラフを描画
plt.plot(train_acc, label="Train Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training Accuracy Over Epochs")
plt.legend()

# 最高値を枠外に表示
max_acc = max(train_acc)
max_epoch = train_acc.index(max_acc) + 1
plt.gcf().text(0.7, 0.92, f"Max: {max_acc:.4f} (Epoch {max_epoch})", fontsize=10, bbox=dict(facecolor='white', alpha=0.7))

plt.show()