import torch
import torchvision.transforms as T
from torchvision.datasets import VOCDetection
import matplotlib.pyplot as plt
from torchvision.ops import nms
from model import YOLOv2  # Скопируй туда же архитектуру модели YOLOv2

VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
    'sheep', 'sofa', 'train', 'tvmonitor']

device = torch.device('cpu')

model = YOLOv2().to(device)
model.load_state_dict(torch.load('yolov2_model.pth', map_location=device))
model.eval()

transform = T.Compose([
    T.Resize((416, 416)),
    T.ToTensor(),
])

dataset = VOCDetection(root='./data', year='2012', image_set='train', download=True, transform=transform)

anchors = torch.tensor([
    [116,90], [156,198], [373,326], [30,61], [62,45]
]).float()

def visualize(idx, threshold=0.2, S=13):
    img, _ = dataset[idx]
    img_tensor = img.unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor).view(S, S, 5, 25).cpu()

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

    boxes_tensor = torch.tensor(boxes)
    scores_tensor = torch.tensor(scores)
    keep = nms(boxes_tensor, scores_tensor, 0.5)

    plt.figure(figsize=(10,10))
    plt.imshow(img.permute(1,2,0))
    ax = plt.gca()

    for idx in keep:
        xmin, ymin, xmax, ymax = boxes_tensor[idx]
        label = labels[idx]
        conf = scores_tensor[idx]

        rect = plt.Rectangle((xmin,ymin), xmax - xmin, ymax - ymin, fill=False, color='lime', linewidth=2)
        ax.add_patch(rect)
        ax.text(xmin, ymin - 10, f'{label}: {conf:.2f}', color='yellow', fontsize=12, weight='bold')

    plt.title('YOLOv2 local test')
    plt.axis('off')
    plt.show()

visualize(0)