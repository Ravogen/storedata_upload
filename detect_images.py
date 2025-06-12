from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import torch
import torch.nn.functional as F
import torch.nn as nn
import uuid
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import faiss
from collections import Counter
import pickle

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
def get_calssification_model(modelpath,number_calsses,pretrainet=False):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = models.resnet34(pretrained=pretrainet)


    model.fc = nn.Linear(model.fc.in_features, number_calsses)
    model.load_state_dict(torch.load(modelpath, map_location=torch.device('cpu')))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    embedding_model = nn.Sequential(*list(model.children())[:-1])  # EI KLASSIFIKAATIOTA
    embedding_model.eval()

    return embedding_model

def image_normalization(image_np):
    image_pil = Image.fromarray(image_np)


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])


    image_tensor = transform(image_pil)

    return image_tensor


def delete_stacked_boxes(boxes, threshold=0.9):
    """
    Poistaa laatikot, joiden pinta-alasta yli `threshold` osuu toiseen laatikkoon.

    boxes: lista bokseja muodossa [x_center, y_center, w, h, conf, cls]
    return: lista säilytettävistä bokseista
    """
    print("ENNEN POISTOA",len(boxes))
    keep = []
    for i, box1 in enumerate(boxes):
        x1_min, y1_min, x1_max, y1_max, conf1, cls1 = box1.tolist()
        w1=x1_max-x1_min
        h1=y1_max-y1_min
        area1 = w1 * h1

        remove = False
        for j, box2 in enumerate(boxes):
            if i == j:
                continue  #Älä vertaa itseensä
            x2_min, y2_min, x2_max, y2_max, conf2, cls2 = box2.tolist()


            # Laske leikkaus
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)

            inter_w = max(0, inter_x_max - inter_x_min)
            inter_h = max(0, inter_y_max - inter_y_min)
            inter_area = inter_w * inter_h

            # Jos box1 on yli threshold-osuudelta box2:n sisällä → poista
            if inter_area / area1 > threshold:
                remove = True
                break

        if not remove:
            keep.append(box1)
    print("POISTON JÄLKEEN",len(keep))
    return keep





def full_model_tester_resnet(image_data,YOLOmodel,emb_resnetModel,index,label_list):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Varmista että koko täsmää
        transforms.ToTensor(),  # Muuttaa [0,255] -> [0,1] ja (H, W, C) -> (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])




    #test_path = './tmp/traintestset_2105'
    test_path=r'D:/tmp/FullTestSet_2005more'
    test_dataset = datasets.ImageFolder(test_path, transform=transform)
    class_names = test_dataset.classes

    pil_image = Image.open(image_data).convert('RGB')  # tai 'L' jos haluat harmaasävykuvan

    # Muunna NumPy-taulukoksi
    image_np = np.array(pil_image)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #img=img.rotate(-90)


    height, width = image_np.shape[:2]

    results = YOLOmodel(image_np)[0]
    boxes = results.boxes.data


    unique_labels = set(label_list)



    import time

    print(f"Eri luokkia yhteensä: {len(unique_labels)}")

    aika_mallinnukseen=0
    boxes=delete_stacked_boxes(boxes)

    predictions=[]
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()


        w, h = x2 - x1, y2 - y1
        cropped = image_np[int(y1):int(y2), int(x1):int(x2)]
        if conf<0.1:

            continue
        if cls==1:
            print("hintalappu",conf)
            predictions.append({
                "bounding_box": {
                    "left": int(x1) / width,
                    "top": int(y1) / height,
                    "width": (int(x2) - int(x1)) / width,
                    "height": (int(y2) - int(y1)) / height},
                "tag_name": "hintalappu",
                # "probability":np.exp(-D[0][0])
                "probability": 0.99
            })
            continue
        img_pil = Image.fromarray(cropped)
        img_tensor=transform(img_pil).unsqueeze(0)

        with torch.no_grad():
            embedding = emb_resnetModel(img_tensor)  # esim. nn.Sequential(...)
            embedding = embedding.view(embedding.size(0), -1).cpu().numpy()  # shape: (1, 2048)
            #embedding=embedding.view(1, -1).cpu().numpy()
            klassifikaatio_aika = time.time()
            faiss.normalize_L2(embedding)
            D, I = index.search(embedding, k=5)
        aika_mallinnukseen+=time.time()-klassifikaatio_aika
        top_labels = [label_list[i] for i in I[0]]

        #predicted = Counter(top_labels).most_common(1)[0][0]

        predicted=top_labels[0]
        if cls!=1:
            tag_name=class_names[predicted]
        else:
            tag_name="hintalappu"


        if D[0][0]>0.3 and cls!=1:
            predictions.append({
                "bounding_box": {
                    "left": int(x1) / width,
                    "top": int(y1) / height,
                    "width": (int(x2) - int(x1)) / width,
                    "height": (int(y2) - int(y1)) / height},
                "tag_name": "[Auto-Generated] Other Products",
                #"probability": np.exp(-D[0][0])
                "probability": np.exp(-D[0][0]*1.666)
                #"probability": float(D[0][0])
            })
            continue

        predictions.append({
            "bounding_box":{
        "left": int(x1)/width,
        "top": int(y1)/height,
        "width":(int(x2) - int(x1))/width,
        "height": (int(y2) - int(y1))/height},
        "tag_name": tag_name,
        #"probability":np.exp(-D[0][0])
        "probability": np.exp(-D[0][0]*1.666)
        #"probability": float(D[0][0])
        })


        #plt.imshow(cropped)
        #plt.title(class_names[predicted]+" "+str(probability))

        #plt.show()
    if len(boxes)!=0:
        print("AIKA KLASSIFIKAATIOLLE", aika_mallinnukseen)
        print("KA", aika_mallinnukseen / len(boxes))

    return predictions
