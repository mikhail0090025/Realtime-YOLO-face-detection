import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from PIL import Image
import requests
import cv2
target_size = (280, 280)

def postprocess(predictions, num_classes=2):
    batch_size = predictions.shape[0]
    grid_y = predictions.shape[2]
    grid_x = predictions.shape[3]
    predictions = predictions.permute(0, 2, 3, 1)

    # Activations
    predictions[:, :, :, 0:2] = torch.sigmoid(predictions[:, :, :, 0:2])  # t_x, t_y
    predictions[:, :, :, 2:4] = torch.exp(predictions[:, :, :, 2:4])     # t_w, t_h
    predictions[:, :, :, 4] = torch.sigmoid(predictions[:, :, :, 4])     # objectness
    predictions[:, :, :, 5:] = torch.softmax(predictions[:, :, :, 5:], dim=-1)   # classes

    return predictions

def get_objects_from_tensor(tensor, num_classes=2, threshold=0.8):
    result = tensor.detach().cpu().numpy()
    if result.shape[0] == 1 and len(result.shape) == 4:
        result = result[0]
    width = result.shape[1]
    height = result.shape[0]
    objects = []
    for i in range(width):
        for j in range(height):
            classes = []
            obj = result[j, i, 4] # objectness
            if obj < threshold:
                continue
            x_offset = result[j, i, 0]
            y_offset = result[j, i, 1]
            w = result[j, i, 2]
            h = result[j, i, 3]
            for n in range(num_classes):
                classes.append(result[j, i, 5 + n])
            ### class_id, centerX_norm, centerY_norm, width_norm, height_norm
            class_id = classes.index(max(classes))
            objects.append([class_id, (i + x_offset) / width, (j + y_offset) / height, (1.0 / width) * w, (1.0 / height) * h, obj])
    return objects

def NMS(objects, max_iou=0.6):
    if not objects:
        return []

    if isinstance(objects, list):
        objects = np.array(objects)
    elif isinstance(objects, torch.Tensor):
        objects = objects.detach().cpu().numpy()

    keep_indices = []

    for i in range(len(objects)):
        if i not in keep_indices:
            keep_indices.append(i)
            for j in range(i + 1, len(objects)):
                if j not in keep_indices:
                    obj1 = objects[i]
                    obj2 = objects[j]
                    if obj1[0] != obj2[0]:
                        continue

                    x1, y1, w1, h1 = obj1[1], obj1[2], obj1[3], obj1[4]
                    x2, y2, w2, h2 = obj2[1], obj2[2], obj2[3], obj2[4]
                    objectness1 = obj1[5]
                    objectness2 = obj2[5]

                    x11 = x1 - w1 / 2  # x_min
                    y11 = y1 - h1 / 2  # y_min
                    x12 = x1 + w1 / 2  # x_max
                    y12 = y1 + h1 / 2  # y_max

                    x21 = x2 - w2 / 2  # x_min
                    y21 = y2 - h2 / 2  # y_min
                    x22 = x2 + w2 / 2  # x_max
                    y22 = y2 + h2 / 2  # y_max

                    x_inter_left = max(x11, x21)
                    y_inter_top = max(y11, y21)
                    x_inter_right = min(x12, x22)
                    y_inter_bottom = min(y12, y22)

                    inter_width = max(0, x_inter_right - x_inter_left)
                    inter_height = max(0, y_inter_bottom - y_inter_top)
                    inter_area = inter_width * inter_height

                    area1 = w1 * h1
                    area2 = w2 * h2

                    union_area = area1 + area2 - inter_area

                    if union_area == 0:
                        iou = 0
                    else:
                        iou = inter_area / (area1 if area1 < area2 else area2)

                    if iou > max_iou:
                        if objectness1 > objectness2 and j in keep_indices:
                            keep_indices.remove(j)
                        if objectness2 > objectness1 and i in keep_indices:
                            keep_indices.remove(i)

    return objects[keep_indices]

def get_predictions(image_np, model, device='cpu', threshold=0.5, num_classes=1, max_iou=0.5, target_size=(280, 280)):
    orig_height, orig_width = image_np.shape[:2]

    # Подготовка изображения для модели
    image_resized = Image.fromarray(image_np).convert("RGB").resize(target_size, Image.BILINEAR)
    image_tensor = torch.from_numpy(np.array(image_resized)/255.0).permute(2,0,1).unsqueeze(0).float().to(device)

    # Предсказания
    model.eval()
    with torch.no_grad():
        predictions = model(image_tensor)
        predictions = postprocess(predictions, num_classes=num_classes)

    # Извлекаем объекты
    objects = get_objects_from_tensor(predictions, num_classes=num_classes, threshold=threshold)
    objects = NMS(objects, max_iou=max_iou)
    print("Objects:", objects)

    boxes, classes = objects[:,1:5], objects[:,0]
    return boxes, classes

def load_checkpoint(model, optimizer, scheduler, filepath='checkpoint.pth', device='cpu'):
    checkpoint = torch.load(filepath, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    print(f"Чекпоинт загружен из {filepath}, эпоха: {epoch}, лосс: {loss}")

    return model, optimizer, scheduler, epoch, loss