from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
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
import json
import hyllyn_sisalto as hs
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

def transdorm(left_original,width,top,height,matrix):

    oikea_ylakulma = np.array([[left_original + width, top]], dtype=np.float32)
    oikea_ylakulma = np.array([oikea_ylakulma], dtype=np.float32)

    oikea_alakulma = np.array([[left_original + width, top + height]], dtype=np.float32)
    oikea_alakulma = np.array([oikea_alakulma], dtype=np.float32)

    vasen_ylakulma = np.array([[left_original, top]], dtype=np.float32)
    vasen_ylakulma = np.array([vasen_ylakulma], dtype=np.float32)

    vasen_alakulma = np.array([[left_original, top + height]], dtype=np.float32)
    vasen_alakulma = np.array([vasen_alakulma], dtype=np.float32)

    oikea_ylakulma_transformed = cv2.perspectiveTransform(oikea_ylakulma, matrix)
    oikea_alakulma_transformed = cv2.perspectiveTransform(oikea_alakulma, matrix)
    vasen_ylakulma_transformed = cv2.perspectiveTransform(vasen_ylakulma, matrix)
    vasen_alakulma_transformed = cv2.perspectiveTransform(vasen_alakulma, matrix)

    left_original = vasen_alakulma_transformed[0][0][0]
    top = oikea_ylakulma_transformed[0][0][1]
    height =oikea_alakulma_transformed[0][0][1] - oikea_ylakulma_transformed[0][0][1]
    width = oikea_ylakulma_transformed[0][0][0] - vasen_ylakulma_transformed[0][0][0]



    return left_original,width,top,height



def full_model_tester_resnet(image_data,YOLOmodel,emb_resnetModel,index,label_list):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Varmista että koko täsmää
        transforms.ToTensor(),  # Muuttaa [0,255] -> [0,1] ja (H, W, C) -> (C, H, W)
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    with open("nimi.json", "r", encoding="utf-8") as f:
        nimet = json.load(f)


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
    ylin_tunnistus=10000
    alin_tunnistus=0
    predictions=[]
    hintalaput_for_transformation={}
    count_hintalappu=0


    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()
        if conf<0.15:
            continue
        if cls==1:
            hintalaput_for_transformation["hintalappu " + str(count_hintalappu)] = {"top": int(y1) / height,
                                                                                    "height": (int(y2) - int(y1)) / height,
                                                                                    "left":int(x1) / width,
                                                                                    "width": (int(x2) - int(x1)) / width,
                                                                                    "prob": float(0.99)}
    #matrix, new_width, original_valid_hintalaput, hyllyjen_maara = hs.get_params_for_transorm(hintalaput_for_transformation,width, height)
    #matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box.tolist()


        w, h = x2 - x1, y2 - y1

        if conf<0.15:

            continue




        left= int(x1) / width
        top=int(y1) / height
        crop_width=(int(x2) - int(x1)) / width
        crop_height=(int(y2) - int(y1)) / height

        cropped = image_np[int(y1):int(y2), int(x1):int(x2)]
        if cls!=1:
            img_pil = Image.fromarray(cropped)
            img_tensor = transform(img_pil).unsqueeze(0)
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
            if tag_name in nimet:
                EAN = nimet[tag_name][1]
                tag_name=nimet[tag_name][0]

            else:
                EAN=None

            #left, crop_width, top, crop_height=transdorm(left,crop_width,top,crop_height,matrix)
            if D[0][0]>0.3 and cls!=1:
                predictions.append({
                    "bounding_box": {
                        "left": left,
                        "top": top,
                        "width": crop_width,
                        "height": crop_height},
                    "tag_name": "[Auto-Generated] Other Products",
                    "EAN":None,
                    #"probability": np.exp(-D[0][0])
                    "probability": np.exp(-D[0][0]*1.666)
                    #"probability": float(D[0][0])
                })
                continue
            if int(top*height)<ylin_tunnistus and cls!=1:
                #ei huomioida hintalappuja koska ne on tuotteiden alla

                ylin_tunnistus=int(top*height)

            if int((top+crop_height)*height)> alin_tunnistus:
                alin_tunnistus = int((top+crop_height)*height)
            predictions.append({
                "bounding_box":{
                        "left": left,
                        "top": top,
                        "width": crop_width,
                        "height": crop_height},
            "tag_name": tag_name,
            "EAN":EAN,
            #"probability":np.exp(-D[0][0])
            "probability": np.exp(-D[0][0]*1.666)
            #"probability": float(D[0][0])
            })
        else:
            if int((top + crop_height) * height) > alin_tunnistus:
                alin_tunnistus = int((top + crop_height) * height)
            predictions.append({
                "bounding_box": {
                    "left": left,
                    "top": top,
                    "width": crop_width,
                    "height": crop_height},
                "tag_name": "hintalappu",
                # "probability":np.exp(-D[0][0])
                "probability": 0.99
            })
            continue


        #plt.imshow(cropped)
        #plt.title(class_names[predicted]+" "+str(probability))

        #plt.show()
    if len(boxes)!=0:
        print("AIKA KLASSIFIKAATIOLLE", aika_mallinnukseen)
        print("KA", aika_mallinnukseen / len(boxes))
    kuvan_uusi_korkeus=alin_tunnistus-ylin_tunnistus
    poistettavat=[]

    skaalauskerroin=height/kuvan_uusi_korkeus
    for prediction in predictions:
        prediction["bounding_box"]["top"] -= ylin_tunnistus/height
        prediction["bounding_box"]["top"]*=skaalauskerroin
        prediction["bounding_box"]["height"]*=skaalauskerroin
        if prediction["bounding_box"]["top"]<0:

            poistettavat.append(prediction)
        #elif prediction["bounding_box"]["top"]+prediction["bounding_box"]["height"]>
    #for i in poistettavat:
    #    for j in predictions:
    #        if i==j:
    new_result = [x for x in predictions if x not in poistettavat]
    print(alin_tunnistus,ylin_tunnistus)
    #print(predictions)
    #print(1/0)


    return new_result,alin_tunnistus,ylin_tunnistus
