from medmnist import PathMNIST
import torchvision.transforms as transforms
import torch
import matplotlib
matplotlib.use('Agg')  # ✅ 添加这行解决 PyCharm 后端错误
import matplotlib.pyplot as plt
# 加载 PathMNIST 数据集
transform = transforms.ToTensor()
dataset = PathMNIST(root='./data',split='train', download=True, transform=transform)

# 标签编号与名称对照表
label_names = {
    0: 'Adipose',
    1: 'Background',
    2: 'Debris',
    3: 'Lymphocytes',
    4: 'Mucus',
    5: 'Smooth muscle',
    6: 'Normal mucosa',
    7: 'Cancer stroma',
    8: 'Adenocarcinoma'
}

# 收集每类图像（各一张）
class_examples = {}
for img, label in dataset:
    label_int = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
    if label_int not in class_examples:
        class_examples[label_int] = img
    if len(class_examples) == 9:
        break

# 显示为 3x3 图像网格
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
for i in range(9):
    row, col = divmod(i, 3)
    img = class_examples[i]
    axes[row, col].imshow(img.permute(1, 2, 0))
    axes[row, col].set_title(f"{i}: {label_names[i]}")
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig("output.png")