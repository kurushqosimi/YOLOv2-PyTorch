import torch
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
from torch.utils.data import DataLoader

transform = T.Compose([
    T.Resize((416, 416)),  # В YOLOv2 размер обычно 416x416
    T.ToTensor(),
])

# Pascal VOC 2012 train
train_dataset = VOCDetection(root='./data', year='2012', image_set='train', download=True, transform=transform)

def collate_fn(batch):
    return tuple(zip(*batch))

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)

import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))

class Darknet19(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            ConvBlock(32, 64, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            ConvBlock(64, 128, 3, 1, 1),
            ConvBlock(128, 64, 1, 1, 0),
            ConvBlock(64, 128, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            ConvBlock(128, 256, 3, 1, 1),
            ConvBlock(256, 128, 1, 1, 0),
            ConvBlock(128, 256, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            ConvBlock(512, 256, 1, 1, 0),
            ConvBlock(256, 512, 3, 1, 1),
            nn.MaxPool2d(2, 2),

            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
            ConvBlock(1024, 512, 1, 1, 0),
            ConvBlock(512, 1024, 3, 1, 1),
        )

    def forward(self, x):
        return self.features(x)

# Проверим, как работает backbone
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
darknet19 = Darknet19().to(device)

# Проверка размера выхода
x = torch.randn(1, 3, 416, 416).to(device)
print(darknet19(x).shape)  # Должен быть [1, 1024, 13, 13]

class YOLOv2(nn.Module):
    def __init__(self, num_classes=20, num_anchors=5):
        super().__init__()
        self.darknet = Darknet19()
        self.head = nn.Conv2d(1024, num_anchors * (5 + num_classes), kernel_size=1)

    def forward(self, x):
        x = self.darknet(x)
        x = self.head(x)
        # меняем форму на (batch_size, S, S, anchors × (5 + num_classes))
        x = x.permute(0, 2, 3, 1)
        return x

# Проверим, как работает YOLOv2-head
model = YOLOv2().to(device)
x = torch.randn(1, 3, 416, 416).to(device)
print(model(x).shape)  # должен быть [1, 13, 13, 125]

anchors = torch.tensor([
    [116,90], [156,198], [373,326], [30,61], [62,45]
]).float().to(device)

def yolo_v2_loss(predictions, targets, anchors, S=13, num_classes=20, lambda_coord=5, lambda_noobj=0.5):
    mse = nn.MSELoss(reduction='sum')

    # masks для ячеек с объектами и без
    obj_mask = targets[..., 4] > 0
    noobj_mask = targets[..., 4] == 0

    # Координаты и размеры
    coord_loss = lambda_coord * mse(predictions[obj_mask][..., :4], targets[obj_mask][..., :4])

    # Confidence loss
    conf_loss_obj = mse(predictions[obj_mask][..., 4], targets[obj_mask][..., 4])
    conf_loss_noobj = lambda_noobj * mse(predictions[noobj_mask][..., 4], targets[noobj_mask][..., 4])

    # Class loss
    class_loss = mse(predictions[obj_mask][..., 5:], targets[obj_mask][..., 5:])

    total_loss = coord_loss + conf_loss_obj + conf_loss_noobj + class_loss
    return total_loss / predictions.shape[0]  # усреднение по batch_size

def prepare_yolov2_targets(batch_targets, anchors, S=13, num_classes=20):
    batch_size = len(batch_targets)
    num_anchors = anchors.size(0)
    target_tensor = torch.zeros(batch_size, S, S, num_anchors, 5 + num_classes).to(device)

    anchor_boxes = anchors / 416  # приводим к [0,1] (относительный размер якорей)

    for b, target in enumerate(batch_targets):
        objs = target['annotation']['object']
        if not isinstance(objs, list):
            objs = [objs]

        for obj in objs:
            bbox = obj['bndbox']
            class_label = obj['name']

            xmin = float(bbox['xmin'])
            ymin = float(bbox['ymin'])
            xmax = float(bbox['xmax'])
            ymax = float(bbox['ymax'])

            # координаты центра bbox в относительных единицах
            x_center = ((xmin + xmax) / 2) / 416
            y_center = ((ymin + ymax) / 2) / 416
            width = (xmax - xmin) / 416
            height = (ymax - ymin) / 416

            if width <= 0 or height <= 0:
                continue

            # Индексы ячейки сетки
            cell_x = int(x_center * S)
            cell_y = int(y_center * S)

            cell_x = min(cell_x, S - 1)
            cell_y = min(cell_y, S - 1)

            # Координаты центра относительно ячейки
            x_cell = x_center * S - cell_x
            y_cell = y_center * S - cell_y

            # Найдём лучший anchor box (с наибольшим IoU)
            ious = []
            for anchor in anchor_boxes:
                anchor_w, anchor_h = anchor
                intersection = min(width, anchor_w) * min(height, anchor_h)
                union = (width * height) + (anchor_w * anchor_h) - intersection
                iou = intersection / union
                ious.append(iou)

            best_anchor = torch.argmax(torch.tensor(ious)).item()

            # Заполняем target-тензор
            target_tensor[b, cell_y, cell_x, best_anchor, :4] = torch.tensor([x_cell, y_cell, width, height])
            target_tensor[b, cell_y, cell_x, best_anchor, 4] = 1.0  # confidence = 1
            class_idx = VOC_CLASSES.index(class_label)
            target_tensor[b, cell_y, cell_x, best_anchor, 5 + class_idx] = 1.0  # класс (one-hot)

    return target_tensor

VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
    'sheep', 'sofa', 'train', 'tvmonitor'
]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLOv2().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10

anchors = torch.tensor([
    [116,90], [156,198], [373,326], [30,61], [62,45]
]).float().to(device)

model.train()
for epoch in range(num_epochs):
    epoch_loss = 0

    for images, targets in train_loader:
        images = torch.stack(images).to(device)
        predictions = model(images)
        
        targets_tensor = prepare_yolov2_targets(targets, anchors)
        
        predictions = predictions.view(-1, 13, 13, 5, 25)
        loss = yolo_v2_loss(predictions, targets_tensor, anchors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}] — Loss: {epoch_loss:.4f}')

print('🎉 Обучение YOLOv2 завершено!')

import matplotlib.pyplot as plt
from torchvision.ops import nms

def visualize_yolov2(model, dataset, idx=0, threshold=0.3, S=13, anchors=None):
    model.eval()
    img, _ = dataset[idx]
    img_tensor = img.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    predictions = predictions.view(S, S, 5, 25).cpu()

    boxes, scores, labels = [], [], []

    cell_size = 416 / S

    for i in range(S):
        for j in range(S):
            for b in range(5):
                conf = predictions[i, j, b, 4]
                if conf > threshold:
                    class_probs = predictions[i, j, b, 5:]
                    best_class_prob, class_label = torch.max(class_probs, dim=0)

                    final_conf = conf * best_class_prob
                    if final_conf < threshold:
                        continue

                    x_cell, y_cell, w, h = predictions[i, j, b, :4]

                    anchor = anchors[b]

                    x_center = (j + torch.sigmoid(x_cell)) * cell_size
                    y_center = (i + torch.sigmoid(y_cell)) * cell_size
                    width = anchor[0] * torch.exp(w)
                    height = anchor[1] * torch.exp(h)

                    xmin = max(x_center - width / 2, 0)
                    ymin = max(y_center - height / 2, 0)
                    xmax = min(x_center + width / 2, 416)
                    ymax = min(y_center + height / 2, 416)

                    boxes.append([xmin, ymin, xmax, ymax])
                    scores.append(final_conf.item())
                    labels.append(VOC_CLASSES[class_label])

    if len(boxes) == 0:
        print("Нет уверенных предсказаний. Попробуй снизить порог threshold.")
        return

    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)
    keep = nms(boxes_tensor, scores_tensor, 0.5)

    plt.figure(figsize=(10, 10))
    plt.imshow(img.permute(1, 2, 0))
    ax = plt.gca()

    for idx in keep:
        xmin, ymin, xmax, ymax = boxes_tensor[idx]
        label = labels[idx]
        conf = scores_tensor[idx]

        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color='lime', linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin, ymin - 10, f'{label}: {conf:.2f}', color='yellow', fontsize=12, weight='bold')

    plt.title('YOLOv2 Detection Results')
    plt.axis('off')
    plt.show()

# Пример визуализации:
visualize_yolov2(model, train_dataset, idx=0, threshold=0.02, anchors=anchors)

# For downloading model
#torch.save(model.state_dict(), 'yolov2_model.pth')

# if you are using google collab uncommit code below
# from google.colab import files
# files.download('yolov2_model.pth')
