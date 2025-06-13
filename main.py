from pymongo import MongoClient
import os
from dotenv import load_dotenv
from bson import ObjectId
from scipy.ndimage import zoom
import json
import aws_haku
load_dotenv()
import cv2
from PIL import Image
import io
from datetime import datetime
import matplotlib.pyplot as plt
import torch.nn as nn
import uuid
from torchvision import models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch
from detect_images import full_model_tester_resnet
from ultralytics import YOLO
import faiss
import hyllyn_sisalto as hs
from collections import Counter
import pickle


client = MongoClient(os.getenv("MONGO_URI"))
unifiedDB=client["unifiedDB_TEST"]
def get_company_id(companyName: str):
    companies=unifiedDB["companies"]
    return companies.find_one({"companyName":companyName})["companyId"]


def get_companys_salesivisitids(companyName: str):

    salesvisits=unifiedDB["salesvisits_embedded"]

    start_date = datetime(2025, 5, 10)
    end_date = datetime(2025, 5, 23, 23, 59, 59)

    companyId=get_company_id(companyName)

    current_visits=salesvisits.find({
                "companyId": companyId,
                "startTime": {"$gte": start_date, "$lte": end_date}
                            })

    return list(current_visits)

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


def get_imageset_imageData(imageset,companyName):
    """

    :param salesvisit:
    :return: imagedata object
    """

    imageurls=[]
    overlaySide=None
    tags=None
    for image in imageset["images"]:
        imageUrl="https://ravogen-app-bucket-prod.s3.amazonaws.com/" + companyName + "/shelfimages/fullsize/" + image["imageUrl"]
        imageurls.append(imageUrl)
        overlaySide=image["overlaySide"]
        tags=image["tags"]
    perushylly=False
    virkkarit=False
    for i in tags:
        print("---------------------")
        print(i)
        if i["categoryName"]=='Sijoittelun tyyppi' and i["tagId"]=='681859f3fbe06906ee9f7f64':
            perushylly=True
        if i["categoryName"]=='Tuotekategoriat' and i['tagId']=='66dfe1f27cf914a09c9f7f79_0':
            virkkarit=True

    if virkkarit==False:
        return None
    if perushylly==False:
        return None
    imageurls=['https://ravogen-app-bucket-prod.s3.amazonaws.com/Lejos/shelfimages/fullsize/ff02f51f-3d49-4aee-8ac3-db43d8272897_1.jpeg', 'https://ravogen-app-bucket-prod.s3.amazonaws.com/Lejos/shelfimages/fullsize/ff02f51f-3d49-4aee-8ac3-db43d8272897_2.jpeg', 'https://ravogen-app-bucket-prod.s3.amazonaws.com/Lejos/shelfimages/fullsize/ff02f51f-3d49-4aee-8ac3-db43d8272897_3.jpeg', 'https://ravogen-app-bucket-prod.s3.amazonaws.com/Lejos/shelfimages/fullsize/ff02f51f-3d49-4aee-8ac3-db43d8272897_4.jpeg']


    image_contents, kuvat, image_key_vaara, panoramaside = aws_haku.get_image(imageurls,panoramaside=overlaySide)

    results=[]

    YOLOmodel = YOLO(
        r"C:\Users\VeikkoHerola\OneDrive - Ravogen Oy\Documents\MALLEJA\runs2-6-25-1024pix\detect\train\weights\best.pt")

    emb_resnetModel = get_calssification_model(
        r"C:\Users\VeikkoHerola\OneDrive - Ravogen Oy\Documents\MALLEJA\ResNet34_12-6_21img_500sku.pth",
        number_calsses=500)

    index = faiss.read_index(
        r"C:\Users\VeikkoHerola\Documents\GitHub\Resnet_model\faiss_base\faiss_index_resnet_ownModel_12-6_500.bin")

    with open(
            r"C:\Users\VeikkoHerola\Documents\GitHub\Resnet_model\faiss_base\label_list_resnet_ownModel_12-6_500.pkl",
            "rb") as f:
        label_list = pickle.load(f)
    count=0
    for image_content in image_contents:
        image_data=io.BytesIO(image_content.getvalue())
        result,alin_tunnistus,ylin_tunnistus=full_model_tester_resnet(image_data,YOLOmodel,emb_resnetModel,index,label_list)

        original_height = kuvat[count].shape[0]
        kuvat[count]=kuvat[count][ylin_tunnistus:alin_tunnistus+1]
        scale = original_height / kuvat[count].shape[0]
        kuvat[count] = zoom(kuvat[count], (scale, 1, 1), order=1)
        results.append(result)
        im = Image.fromarray(kuvat[count])
        im.save(f"img_{count}.jpeg", format="JPEG")
        count+=1

    with open("tiedosto.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(1/0)
    hyllyn_sisalto, tuotteet_hyllyissa, hintalaput_hyllyissa, hintalaput = hs.tuotteiden_sijainnit_dev(results, kuvat,
                                                                                                       iteration20=True,
                                                                                                       transform=True,
                                                                                                       plot=True)


    return hyllyn_sisalto
def load_imagedata(companyname):
    salesvisit_coll=unifiedDB["salesvisits_embedded"]
    salesvisits=get_companys_salesivisitids(companyname)
    for salesvisit in salesvisits:
        for imageset in salesvisit["imagesets"]:
            #if imageset["imageData"]!=None:
            #    print("data valmiina")
            #    continue
            imageData=get_imageset_imageData(imageset, companyname)
            if imageData==None:
                continue
            #print(1/0)
            salesvisitId=salesvisit["_id"]
            imagesetId=imageset["_id"]
            print(salesvisitId)
            salesvisit_coll.update_one(
                {
                    "_id": salesvisitId,
                    "imagesets._id": imagesetId
                },
                {
                    "$set": {
                        "imagesets.$.imageData": imageData
                    }
                }
            )


load_imagedata("Lejos")