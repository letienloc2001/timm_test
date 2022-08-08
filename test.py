def pretty_print(raw_text: str, text_color: str, background_color: str = '', style: str = 'normal', end: str = '\n', string_return: bool = False):
    style_dict = {
        'n': 0, 'normal'    : 0,
        'b': 1, 'bold'      : 1,
        'l': 2, 'light'     : 2,
        'i': 3, 'italicized': 3,
        'u': 4, 'underlined': 4,
        'blink'  : 5,
        'inverse': 7,
        'hidden' : 8,
        'strikethrough': 9
    }

    color_dict = {
        'black' : 0,
        'red'   : 1,
        'green' : 2,
        'yellow': 3,
        'blue'  : 4,
        'purple': 5,
        'cyan'  : 6,
        'white' : 7,
    }

    Text_color, Background_color = text_color.split(' '), background_color.split(' ')

    style_code = str(style_dict[style]) + ';'
    text_color_code = str(color_dict[Text_color[-1]] + (90 if Text_color[0] == 'bright' else 30))
    background_color_code = (';' + str(color_dict[Background_color[-1]] + (100 if Background_color[0] == 'bright' else 40))) if Background_color[0] != '' else ''
    formatted_text = '\033[' + style_code + text_color_code + background_color_code + 'm' + raw_text + '\033[0m'
    if string_return:
        return formatted_text
    else: 
        print(formatted_text, end=end)


import numpy as np
import torch
import os
import random
from app.models.classification.timm.timm import create_model
from torchvision import transforms as t
import torchvision

torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

NUM_EPOCHS = 15
NUM_CLASSES = 2
BATCH_SIZE = 4
LEARNING_RATE = 0.0005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001

NUM_CROPS = 10
RESIZE_SIZE  = [1000,1000]
CROP_SIZE = [400,400]


transform_TenCrop = t.Compose([
    t.Resize(size=RESIZE_SIZE),
    t.TenCrop(size=CROP_SIZE),
    t.Lambda(lambda crops: [t.ToTensor()(crop) for crop in crops]),
    t.Lambda(lambda crops: [t.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))(crop) for crop in crops]),
    t.Lambda(lambda crops: torch.stack(crops))               
])

transform_CenterCrop = t.Compose([
    t.Resize(size=RESIZE_SIZE),
    t.CenterCrop(size=CROP_SIZE),
    t.ToTensor(),
    t.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])

DATASET   = '/content/test/dataset'
TRAIN_SET = os.path.join(DATASET, 'train')
VAL_SET   = os.path.join(DATASET, 'validation')
TEST_SET  = os.path.join(DATASET, 'test')

train_dataset = torchvision.datasets.ImageFolder(root=TRAIN_SET, transform=transform_TenCrop)
val_dataset   = torchvision.datasets.ImageFolder(root=VAL_SET,   transform=transform_CenterCrop)
test_dataset  = torchvision.datasets.ImageFolder(root=TEST_SET,  transform=transform_TenCrop)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = torch.utils.data.DataLoader(dataset=val_dataset,   batch_size=BATCH_SIZE, shuffle=True)
test_loader  = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

model = create_model('mixnet_s', pretrained=True, num_classes=2)
model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
model.to(device)

criterion = torch.nn.CrossEntropyLoss()

optimizer  = torch.optim.SGD(
    filter(lambda p: p.requires_grad, model.parameters()), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY, 
    momentum=MOMENTUM
)

min_valid_loss = np.inf
saving_flag = False

pretty_print('🚀 START TRAINING ...', text_color='purple', style='bold')

train_loss_vals = []
val_loss_vals = []
train_accu_vals = []
val_accu_vals = []
for epoch in range(NUM_EPOCHS):
    pretty_print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}: ', text_color='cyan')

    # TRAIN LOOP
    train_loss = 0.0
    correct = 0
    total = 0
    train_epoch_loss = []
    model.train()  
    NUM_CROPS = 10   
    for i, (images, labels) in enumerate(train_loader):
        print('\r', end='')
        print(f'{100*i/len(train_loader):.2f} % ' + '-'*int(100*i/len(train_loader)), end='')
        crop_list = images.tolist()
        for crop_idx in range(NUM_CROPS):
            cropped_images = torch.Tensor([crop_list[batch_idx][crop_idx] for batch_idx in range(images.size(0))])
            
            cropped_images, labels = cropped_images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(cropped_images)
            _, predicteds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicteds == labels).sum().item()

            loss = criterion(outputs, labels)
            loss.backward()
            train_epoch_loss.append(loss.item())
            optimizer.step()
            train_loss += loss.item()
        
    print('\r', end='')
    print(f'🫠 Training Accuracy  : {100 * correct / total:.2f} %, Training Loss  : {train_loss / (len(train_loader)*NUM_CROPS):.5f}')
    train_loss_vals.append(sum(train_epoch_loss)/len(train_epoch_loss))
    train_accu_vals.append(100 * correct / total)

    # VALIDATION LOOP
    valid_loss = 0.0
    correct = 0
    total = 0
    val_epoch_loss = []
    model.eval() 
    NUM_CROPS = 1    
    for i, (cropped_images, labels) in enumerate(val_loader):
        print('\r', end='')
        print(f'{100*i/len(train_loader):.2f} % ' + '-'*int(100*i/len(train_loader)), end='')
        # crop_list = images.tolist()
        # for crop_idx in range(NUM_CROPS):
        #     cropped_images = torch.Tensor([crop_list[batch_idx][crop_idx] for batch_idx in range(images.size(0))])
        cropped_images, labels = cropped_images.to(device), labels.to(device)

        outputs = model(cropped_images)

        _, predicteds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicteds == labels).sum().item()

        loss = criterion(outputs, labels)
        val_epoch_loss.append(loss.item())
        valid_loss += loss.item()

    print('\r', end='')
    print(f'🫠 Validation Accuracy: {100 * correct / total:.2f} %, Validation Loss: {valid_loss / (len(val_loader)*NUM_CROPS):.5f}')
    val_loss_vals.append(sum(val_epoch_loss)/len(val_epoch_loss))
    val_accu_vals.append(100 * correct / total)

    # SAVE CHECKPOINT MODEL
    if min_valid_loss > valid_loss:
        pretty_print(f'🎯 CHECKPOINT: Validation Loss ({min_valid_loss / (len(val_loader)*NUM_CROPS):.5f} ==> {valid_loss / (len(val_loader)*NUM_CROPS):.5f})', text_color='green')
        pretty_print(f'📥 SAVING MODEL ...', text_color='green')
        min_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_model.pth')
        saving_flag = True