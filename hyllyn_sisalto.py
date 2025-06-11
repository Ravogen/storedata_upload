import cv2
import os
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
import hyllyhahmotelma_dev as hh_dev


# import Heikki_on_azure as HOA

def list_files_in_directory(path):
    # TODO
    """

    :param path: aws bucketin nimi
    :return: lista bucketin tiedostonimistä
    """
    return []


def save_dict_to_json(directory_path, file_name, data):
    # TODO
    """

    :param directory_path: aws3 bucket sijainti
    :param file_name: millä nimellä tallennetaan
    :param data: dict
    :return:
    """

    data = json.dump(data)




def prediction_to_dict(pred):
    return {
        'probability': pred.probability,
        'tag_id': str(pred.tag_id),
        'tag_name': pred.tag_name,
        'tag_type': pred.tag_type,
        'bounding_box': {
            'left': pred.bounding_box.left,
            'top': pred.bounding_box.top,
            'width': pred.bounding_box.width,
            'height': pred.bounding_box.height
        }
    }


def tuotteiden_sijainnit_dev(result_list: list, images: list,  iteration20=False, transform=False,
                             plot=False):
    """
    Tämä funktio saa yhden hyllyn kuvan/kuvat ja sen/niiden tuloksen/tulokset. kuvalista on siis lista, jossa on
    panoramakuva palasteltuna, results lista on lista jossa kunkin palan tulokset.

    huom. jos hylly on saatu yhteen kuvaan, listat tulee yhden mittaisena sisältäen yhden kuvan ja sen tulokset
    (15% poistettu jo kuvia hakiessa s3:sta)
    :param result_list: lista eri kuvien tuloksista
    :param images: kuvat
    :param image_key:
    :return:
    """

    # kokonaisen kuvan leveys (koska kuva voi olla palasteltu
    kokoleveys = 0
    kokokorkeus = 0
    for i in images:
        height_pala, width_pala = i.shape[:2]
        kokoleveys += width_pala
        kokokorkeus = height_pala
    print("kuvan leveys alkuperäisestä kuvasta", kokoleveys)

    box_jsons = []

    # tarkistetaan onko kuvalle jo boundingboxit tallennettu

    count = 0  # tuote lkm laskuri

    tuotteet_hyllyissa = {}

    alku = 0
    loppu = 0

    hintalaput = {}
    transform_matrises = {}
    transformed_limits = 0
    count_hintalappu = 0
    # fig, ax = plt.subplots(1)
    hyllyjen_maarat_laupista=[]
    for i in range(len(result_list)):
        print("KUVA", i, "MENOSSA")

        # fig, ax = plt.subplots(1)
        image = images[i]
        results = result_list[i]
        height_pala, width_pala = image.shape[:2]

        # ax.imshow(image)

        print(image.shape[1] + loppu)

        box_json = {}


        hintalaput_for_transformation = {}
        alin_tuote_top = 0

        for prediction in results:
            """
            KÄSITELLÄÄN ENSIN HINTALAPUT. NIIDEN AVULLA MUODOSTETAAN TARVITTAESSA TRANFORMAATIOMATRIISI KUVAN VÄÄNTÖÄ VARTEN
            SAMALLA KARSITAAN HINTALAPUT, JOTKA OVAT KAUKANA MUISTA HINTALAPUISTA (TODENNÄKÖISESTI VIRHETUNNISTUKSIA)
            """
            if not isinstance(prediction, dict):
                prediction = prediction_to_dict(prediction)

            tuote = prediction["tag_name"]
            # if left_original>width_pala:
            #    print(1/0)
            #    continue

            left = round(prediction["bounding_box"]["left"] * image.shape[1])
            top = round(prediction["bounding_box"]["top"] * image.shape[0])
            width = round(prediction["bounding_box"]["width"] * image.shape[1])
            height = round(prediction["bounding_box"]["height"] * image.shape[0])
            if left + width > width_pala:
                width = left - width_pala

            if tuote == "hintalappu" and prediction["probability"] > 0.98:
                hintalaput_for_transformation["hintalappu " + str(count_hintalappu)] = {"top": top,
                                                                                        "height": height,
                                                                                        "left": left,
                                                                                        "width": width,
                                                                                        "prob": float(
                                                                                            prediction["probability"])}
                count_hintalappu += 1
                print("HINTALAPPYMYSTEERI","hintalappu " + str(count_hintalappu))
                continue
        if transform:
            """
            original_valid_hintalaput ON SIIS HINTALAPUT ORIGINAALEILLA KOORDINAATEILLA, JA SIIVOTTUNA EPÄILYTTÄVISTÄ LAPUISTA
            """
            matrix, new_width, original_valid_hintalaput,hyllyjen_maara = get_params_for_transorm(hintalaput_for_transformation,
                                                                                   width_pala, height_pala)
            hyllyjen_maarat_laupista.append(hyllyjen_maara)
            transform_matrises[i] = [transformed_limits, matrix, transformed_limits + new_width]
            transformed_limits += new_width
        else:
            """
            JOS EI KÄYTETÄ TRANSFORMOINTIA, EPÄILYTTÄVÄT LAPUT POISTETAAN MYÖHEMMÄSSÄ VAIHEESSA 
            """
            original_valid_hintalaput = hintalaput_for_transformation

        for prediction in results:
            """
            TÄSSÄ LOOPISSA KÄSITELLÄÄN TUOTTEET. JOS HALUTAAN TRANSORMOIDA, KOORDINAATIT VÄÄNNETÄÄN.

            """
            if not isinstance(prediction, dict):
                prediction = prediction_to_dict(prediction)
            count += 1
            if prediction["probability"] <= 0.1 and prediction["tag_name"] == "hintalappu":
                continue
            # if prediction.probability >= 0.50 or prediction.tag_name == "hintalappu":

            if prediction["probability"] < 0.3 and prediction[
                "tag_name"] != '[Auto-Generated] Other Products':
                prediction["tag_name"] = '[Auto-Generated] Other Products'
                prediction["probability"] = prediction["probability"]

            if prediction["probability"] >= 0.5 or prediction[
                "tag_name"] == "hintalappu":
                left_original = round(prediction["bounding_box"]["left"] * image.shape[1])
                top = round(prediction["bounding_box"]["top"] * image.shape[0])
                width = round(prediction["bounding_box"]["width"] * image.shape[1])
                height = round(prediction["bounding_box"]["height"] * image.shape[0])
                if transform:
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

                    left_original = int(vasen_alakulma_transformed[0][0][0])
                    top = int(oikea_ylakulma_transformed[0][0][1])
                    height = int(oikea_alakulma_transformed[0][0][1] - oikea_ylakulma_transformed[0][0][1])
                    width = int(oikea_ylakulma_transformed[0][0][0] - vasen_ylakulma_transformed[0][0][0])

                if prediction["tag_name"] == '[Auto-Generated] Other Products':
                    prediction["tag_name"] = "Auto" + str(count)

                # ei haluta väärinpäin olveia tölkkejä
                if prediction["tag_name"] == "tolkin_katto":
                    continue
                left = left_original + loppu
                if prediction["tag_name"] == "hintalappu":
                    # KÄSITELLLÄÄN MYÖHEMMIN
                    # hintalaput["hintalappu " + str(count)] = {"top": top,
                    #                                          "height": height,
                    #                                          "left": left,
                    #                                          "width": width,
                    #                                          "prob": prediction.probability}

                    continue
                if top > alin_tuote_top:
                    alin_tuote_top = top
                if prediction["tag_name"] == "welldone_385g_condensed_milk_musta":
                    prediction["tag_name"] = "welldone_385g_condensed_milk_sininen"
                # print(prediction.tag_name,prediction.probability)
                tuotteet_hyllyissa[prediction["tag_name"] + " " + str(count)] = {"top": top,
                                                                                 "height": height,
                                                                                 "left": left,
                                                                                 "width": width,
                                                                                 "prob": float(
                                                                                     prediction["probability"])}

        for lappu in original_valid_hintalaput.keys():
            """
            KÄSITELLÄÄN HINTALAPUT NYT ERIKSEEN.
            """
            hintalappu = original_valid_hintalaput[lappu]
            left_original = round(hintalappu["left"])
            top = round(hintalappu["top"])
            width = round(hintalappu["width"])
            height = round(hintalappu["height"])
            if transform:
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

                left_original = int(vasen_alakulma_transformed[0][0][0])
                top = int(oikea_ylakulma_transformed[0][0][1])
                height = int(oikea_alakulma_transformed[0][0][1] - oikea_ylakulma_transformed[0][0][1])
                width = int(oikea_ylakulma_transformed[0][0][0] - vasen_ylakulma_transformed[0][0][0])

            left = left_original + loppu
            hintalaput[lappu] = {"top": top,
                                 "height": height,
                                 "left": left,
                                 "width": width,
                                 "prob": float(hintalappu["prob"])}

        loppu += width_pala
    # TODO Heikki tähän väliin
    # panoramakuva=cv2.hconcat(images)
    # HOA.labeloi_klikkailemalla(panorama_kuva=panoramakuva,tuotekuvat_path="C:/Users/VeikkoHerola/Desktop/tuotekuvia_oljy/",
    #                           tuotteet_hyllyissa=tuotteet_hyllyissa,keys_to_model="sec_oliivi.json")

    # hintalaput = {}

    # import random
    # keys_to_remove = random.sample(list(hintalaput.keys()), int(len(hintalaput)/ 1.2))
    # for key in keys_to_remove:
    #    del hintalaput[key]

    # save_dict_to_json("C:/Users/VeikkoHerola/PycharmProjects/product_search_2/bounging boxes",image_key,box_json)
    print("HINTALAPPUJA", len(list(hintalaput.keys())))
    print("TUOTTEITA", len(list(tuotteet_hyllyissa.keys())))

    if len(hintalaput.keys()) > 2:
        hintalaput_clean, hintalaput_hyllyissa, kmeans_init_HINTALAPPU, hyllyjen_maara_HINTALAPPU = kmeans_initial_and_clustered_objects(
            hintalaput.copy(), kokoleveys, kokokorkeus)
        tuotteet_hyllyissa, tuotteet_hyllyissa_IGNORE, kmeans_init_TUOTE, hyllyjen_maara_TUOTE = kmeans_initial_and_clustered_objects(
            tuotteet_hyllyissa.copy(),
            kokoleveys,
            kokokorkeus)
        print("HINTALAPPUJA", len(list(hintalaput_clean.keys())))
        if hyllyjen_maara_TUOTE > 2 + hyllyjen_maara_HINTALAPPU or isinstance(kmeans_init_HINTALAPPU, str):
            kmeans_init = kmeans_init_TUOTE
            hyllyjen_maara = hyllyjen_maara_TUOTE


        else:
            kmeans_init = kmeans_init_HINTALAPPU

            # print(kmeans_init)
            if alin_tuote_top > kmeans_init[-1][1]:
                print("LISÄTÄÄN HYLLY MANUAALISESTI")
                "LISÄTÄÄN HYLLY KOSKA ALIMMAN HINTALAPPURIVIN LAPUOLELLA ON TUOTTEITA"
                uusix = kmeans_init[-1][0]
                uusiy = height_pala
                kmeans_init = np.vstack([kmeans_init, [uusix, uusiy]])
                hyllyjen_maara_HINTALAPPU += 1
            hyllyjen_maara = hyllyjen_maara_HINTALAPPU
    elif len(tuotteet_hyllyissa.keys()) > 2:
        tuotteet_hyllyissa, tuotteet_hyllyissa_IGNORE, kmeans_init_TUOTE, hyllyjen_maara_TUOTE = kmeans_initial_and_clustered_objects(
            tuotteet_hyllyissa,
            kokoleveys,
            kokokorkeus)
        kmeans_init = kmeans_init_TUOTE
        hyllyjen_maara = hyllyjen_maara_TUOTE
        hintalaput_hyllyissa = {}
        hintalaput_clean = hintalaput
    else:

        kmeans_init = "k-means++"
        hyllyjen_maara = 1
        hintalaput_hyllyissa = {}
        hintalaput_clean = hintalaput

    print("HYLLYJEN MÄÄRÄ", hyllyjen_maara)
    print("TUOTTEITA", len(list(tuotteet_hyllyissa.keys())))
    print(hintalaput_clean)
    if iteration20:
        hyllyn_sisalto, tuotteet_hyllyissa = hh_dev.hyllyhahmotelma_ilmanhyllyja(tuotteet_hyllyissa, hintalaput_clean,
                                                                                 hintalaput_hyllyissa,
                                                                                 hyllyjen_maara=hyllyjen_maara,
                                                                                 kmeans_initial=kmeans_init,
                                                                                 branditasolla=False, plot=plot)

    if transform:
        # JOS KOORDINAATISTO ON VÄÄNNETTY, SE PITÄÄ PALAUTTAA.
        # MUUTEN ESIMERKIKSI HINTALAPPUJEN LUKEMINEN MENEE PERSEELLEEN
        hyllyn_sisalto, hintalaput = return_original_transform(hyllyn_sisalto, hintalaput, transform_matrises)


    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))  # 1 rivi, 2 saraketta
    if plot:
        image = cv2.hconcat(images)
        fig, ax = plt.subplots()
        # axs[0].imshow(image)
        ax.imshow(image)
    # axs[1].set_xlim(0, image.shape[1])
    # axs[1].set_ylim(image.shape[0], 0)
    #hyllyn_sisalto = muut_tuotteet_filter(hyllyn_sisalto)
    #hyllyn_sisalto = yhdista_valituotteet_yhdeksi(hyllyn_sisalto)
    hyllyn_sisalto = yhdista_vierekkaiset_samat_tuotteet(hyllyn_sisalto)
    hyllyn_sisalto = overlap_filter(hyllyn_sisalto)
    if plot:
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
        for hylly in hyllyn_sisalto.keys():
            print("hyllyssä", hylly, "tuotteita", len(hyllyn_sisalto[hylly]))
            for value in hyllyn_sisalto[hylly]:
                # Luo suorakulmio annetuilla arvoilla
                if "Borges" in value['tuotenimi']:
                    rect = patches.Rectangle((value['left'], value['top']), value['width'], value['height'],
                                             linewidth=1, edgecolor='r', facecolor="red", alpha=0.5)
                else:
                    rect = patches.Rectangle((value['left'], value['top']), value['width'], value['height'],
                                             linewidth=1, edgecolor='r', facecolor=colors[int(hylly)], alpha=0.5)
                ax.add_patch(rect)
                ax.text(value['left'], value['top'] + int(value["height"]),
                        value['tuotenimi'] + " " + str(value["kpl"]),
                        fontsize=8, color='blue', rotation=25)
        for lappu in hintalaput.keys():
            rect = patches.Rectangle((hintalaput[lappu]['left'], hintalaput[lappu]['top']), hintalaput[lappu]['width'],
                                     hintalaput[lappu]['height'],
                                     linewidth=2, edgecolor='r', facecolor="purple", alpha=0.5)
            ax.add_patch(rect)
        plt.show()
    hintalaput_hyllyissa = {str(key): value for key, value in hintalaput_hyllyissa.items()}

    return hyllyn_sisalto, tuotteet_hyllyissa, hintalaput_hyllyissa, hintalaput


def return_original_transform(hyllyn_sisalto, hintalaput, transform_matrises):
    """
    Muuttaa hintalappujen ja tuotteiden sijainnit alkuperäiseen koordinaatistoon.
    Tähän käytetään tranformaatiomatriisin käänteismatriisia


    :param hyllyn_sisalto: hyllynsisältö, jonka koordinaatisto on transformoitu
    :param hintalaput: hintalaput, jonka koordinaatisto on transformoitu
    :param transform_matrises (dict): trasformaatioon käytetyt matriisit. avaimena kuvan indeksi, arvona lista jossa kyseisen väännetyn
                kuvan vaikutusalueen vasen reuna, itse matriisi ja vaikutusalueen oikea reuna
    :return:
    """

    for hylly in hyllyn_sisalto.keys():
        for tuote in hyllyn_sisalto[hylly]:
            left = tuote["left"]
            top = tuote["top"]
            height = tuote["height"]
            width = tuote["width"]
            tuotteelle_loytyi_kaanteismatriisi = False
            miinustettava_pikselimaara=0
            for matris_index in transform_matrises:
                if left >= transform_matrises[matris_index][0] and left < transform_matrises[matris_index][2]:
                    matrix_inv = np.linalg.inv(transform_matrises[matris_index][1])
                    miinustettava_pikselimaara=transform_matrises[matris_index][0]
                    tuotteelle_loytyi_kaanteismatriisi = True
                    break
            if tuotteelle_loytyi_kaanteismatriisi == False:
                print("TUOTE EI OLE MINKÄÄN TRANSFORMAATIOMATRIISIN VAIKUTUKSEN ALLA")
                print(1 / 0)
            oikea_ylakulma = np.array([[left + width-miinustettava_pikselimaara, top]], dtype=np.float32)
            oikea_ylakulma = np.array([oikea_ylakulma], dtype=np.float32)

            oikea_alakulma = np.array([[left + width-miinustettava_pikselimaara, top + height]], dtype=np.float32)
            oikea_alakulma = np.array([oikea_alakulma], dtype=np.float32)

            vasen_ylakulma = np.array([[left-miinustettava_pikselimaara, top]], dtype=np.float32)
            vasen_ylakulma = np.array([vasen_ylakulma], dtype=np.float32)

            vasen_alakulma = np.array([[left-miinustettava_pikselimaara, top + height]], dtype=np.float32)
            vasen_alakulma = np.array([vasen_alakulma], dtype=np.float32)

            oikea_ylakulma_transformed = cv2.perspectiveTransform(oikea_ylakulma, matrix_inv)
            oikea_alakulma_transformed = cv2.perspectiveTransform(oikea_alakulma, matrix_inv)
            vasen_ylakulma_transformed = cv2.perspectiveTransform(vasen_ylakulma, matrix_inv)
            vasen_alakulma_transformed = cv2.perspectiveTransform(vasen_alakulma, matrix_inv)

            left = int(vasen_alakulma_transformed[0][0][0])+miinustettava_pikselimaara
            top = int(oikea_ylakulma_transformed[0][0][1])
            height = int(oikea_alakulma_transformed[0][0][1] - oikea_ylakulma_transformed[0][0][1])
            width = int(oikea_ylakulma_transformed[0][0][0] - vasen_ylakulma_transformed[0][0][0])

            tuote["left"] = left
            tuote["top"] = top
            tuote["height"] = height
            tuote["width"] = width

    for lappu in hintalaput.keys():

        left = hintalaput[lappu]["left"]
        top = hintalaput[lappu]["top"]
        height = hintalaput[lappu]["height"]
        width = hintalaput[lappu]["width"]
        tuotteelle_loytyi_kaanteismatriisi = False
        miinustettava_pikselimaara=0
        for matris_index in transform_matrises:
            if left >= transform_matrises[matris_index][0] and left < transform_matrises[matris_index][2]:
                matrix_inv = np.linalg.inv(transform_matrises[matris_index][1])
                miinustettava_pikselimaara = transform_matrises[matris_index][0]
                tuotteelle_loytyi_kaanteismatriisi = True
                break
        if tuotteelle_loytyi_kaanteismatriisi == False:
            print("TUOTE EI OLE MINKÄÄN TRANSFORMAATIOMATRIISIN VAIKUTUKSEN ALLA")
            print(1 / 0)
        oikea_ylakulma = np.array([[left + width-miinustettava_pikselimaara, top]], dtype=np.float32)
        oikea_ylakulma = np.array([oikea_ylakulma], dtype=np.float32)

        oikea_alakulma = np.array([[left + width-miinustettava_pikselimaara, top + height]], dtype=np.float32)
        oikea_alakulma = np.array([oikea_alakulma], dtype=np.float32)

        vasen_ylakulma = np.array([[left-miinustettava_pikselimaara, top]], dtype=np.float32)
        vasen_ylakulma = np.array([vasen_ylakulma], dtype=np.float32)

        vasen_alakulma = np.array([[left-miinustettava_pikselimaara, top + height]], dtype=np.float32)
        vasen_alakulma = np.array([vasen_alakulma], dtype=np.float32)

        oikea_ylakulma_transformed = cv2.perspectiveTransform(oikea_ylakulma, matrix_inv)
        oikea_alakulma_transformed = cv2.perspectiveTransform(oikea_alakulma, matrix_inv)
        vasen_ylakulma_transformed = cv2.perspectiveTransform(vasen_ylakulma, matrix_inv)
        vasen_alakulma_transformed = cv2.perspectiveTransform(vasen_alakulma, matrix_inv)

        left = int(vasen_alakulma_transformed[0][0][0])+miinustettava_pikselimaara
        top = int(oikea_ylakulma_transformed[0][0][1])
        height = int(oikea_alakulma_transformed[0][0][1] - oikea_ylakulma_transformed[0][0][1])
        width = int(oikea_ylakulma_transformed[0][0][0] - vasen_ylakulma_transformed[0][0][0])

        hintalaput[lappu]["left"] = left
        hintalaput[lappu]["top"] = top
        hintalaput[lappu]["height"] = height
        hintalaput[lappu]["width"] = width

    return hyllyn_sisalto, hintalaput


def get_params_for_transorm(hintalaput, kokoleveys, kokokorkeus):
    hintalaput, _1, kmeans_init, hyllyjen_maara = kmeans_initial_and_clustered_objects(hintalaput, kokoleveys,
                                                                                       kokokorkeus)
    print("KERROKSIEN MÄÄRÄ ALUKSI", hyllyjen_maara)
    if isinstance(kmeans_init, str) or len(hintalaput) < 4 or hyllyjen_maara < 3:
        "HINTALAPUISTA EI SAATU KLUSTEREITA, ELI KMEANS INITIAL ARVOJA EI OLE"
        return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), kokoleveys, hintalaput,hyllyjen_maara
    # print(_1)
    koordinaatit_for_transform = []
    ylin_piste = [0, 0]
    alin_piste = [0, 10000]
    for lappu in hintalaput.keys():
        if hintalaput[lappu]["top"] + hintalaput[lappu]["height"] > ylin_piste[1]:
            ylin_piste = [hintalaput[lappu]["left"], hintalaput[lappu]["top"] + hintalaput[lappu]["height"]]

        if hintalaput[lappu]["top"] + hintalaput[lappu]["height"] < alin_piste[1]:
            alin_piste = [hintalaput[lappu]["left"], hintalaput[lappu]["top"] + hintalaput[lappu]["height"]]

        koordinaatit_for_transform.append(
            [hintalaput[lappu]["left"], hintalaput[lappu]["top"] + hintalaput[lappu]["height"]])

    toiseksi_ylin_piste = [0, 0]
    toiseksi_alin_piste = [0, 10000]

    #Algoritmi, joka hakee toiseksi ylimmän ja toiseksi alimman pisteen. Lähdetään siis kallistamaan kulmaa sen suuntaan
    for lappu in hintalaput.keys():

        piste = hintalaput[lappu]["top"] + hintalaput[lappu]["height"]
        if piste > toiseksi_ylin_piste[1] and piste < ylin_piste[1]:
            toiseksi_ylin_piste = [hintalaput[lappu]["left"], hintalaput[lappu]["top"] + hintalaput[lappu]["height"]]

        if piste < toiseksi_alin_piste[1] and piste > alin_piste[1]:
            toiseksi_alin_piste = [hintalaput[lappu]["left"], hintalaput[lappu]["top"] + hintalaput[lappu]["height"]]
    """
    # Algorimti, joka etsii keskimääräisen pisteen sijainnnin. Lähdetään siis kallistamaan kesimääräisen
    # pisteen suuntaan

    toiseksi_ylin_piste = [0, 0]
    toiseksi_alin_piste = [0, 0]
    for lappu in hintalaput.keys():
        x = hintalaput[lappu]["left"]
        y = hintalaput[lappu]["top"] + hintalaput[lappu]["height"]
        toiseksi_ylin_piste[0] += x
        toiseksi_ylin_piste[1] += y
        toiseksi_alin_piste[0] += x
        toiseksi_alin_piste[1] += y
    toiseksi_alin_piste[0] = toiseksi_alin_piste[0] / len(hintalaput.keys())
    toiseksi_ylin_piste[0] = toiseksi_ylin_piste[0] / len(hintalaput.keys())
    toiseksi_alin_piste[1] = toiseksi_alin_piste[1] / len(hintalaput.keys())
    toiseksi_ylin_piste[1] = toiseksi_ylin_piste[1] / len(hintalaput.keys())
    print("PISTE", toiseksi_alin_piste, toiseksi_ylin_piste)
    """
    # YLÄPUOLEN SUORAN YHTÄLÖ A * x + b
    if toiseksi_ylin_piste[0] <= ylin_piste[0]:
        print("YLÄPUOLI, KULMAKERROIN POSITIIVINEN")
        A1, b1 = find_transform_points(koordinaatit_for_transform, ylin_piste, kokoleveys, ylapuoli=True,
                                       kulmakerroin_pos=True)
    else:
        A1, b1 = find_transform_points(koordinaatit_for_transform, ylin_piste, kokoleveys, ylapuoli=True,
                                       kulmakerroin_pos=False)
        print("YLÄPUOLI, KULMAKERROIN NEGATIIVINEN")

    # ALAPUOLEN SUORAN YHTÄLÖ A*x+b
    if toiseksi_alin_piste[0] >= alin_piste[0]:
        print("ALAPUOLI, KULMAKERROIN POSITIIVINEN")
        A2, b2 = find_transform_points(koordinaatit_for_transform, alin_piste, kokoleveys, ylapuoli=False,
                                       kulmakerroin_pos=True)
    else:
        A2, b2 = find_transform_points(koordinaatit_for_transform, alin_piste, kokoleveys, ylapuoli=False,
                                       kulmakerroin_pos=False)
        print("ALAPUOLI, KULMAKERROIN NEGATIIVINEN")
    # vasen yläkulma
    x1 = 0
    y1 = b1
    print("YLÄPUOLEN KULMAKERROIN", A1)
    print("ALAPUOLEN KULMAKERROIN", A2)
    # oikea yläkulma
    x2 = kokoleveys
    y2 = A1 * kokoleveys + b1

    # oikea alakulma
    x3 = kokoleveys
    y3 = A2 * kokoleveys + b2

    # vasen alakulma
    x4 = 0
    y4 = b2

    # VÄÄNNETTÄVÄT PISTEET
    src_points = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])

    # KOHDEPISTEET
    dst_points = np.float32([[0, max(b1, A1 * kokoleveys + b1)],
                             [kokoleveys, max(b1, A1 * kokoleveys + b1)],
                             [kokoleveys, min(b2, A2 * kokoleveys + b2)],
                             [0, min(b2, A2 * kokoleveys + b2)]])

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    transformed_points = []

    # KOSKA YLIMENEVÄT ALUEET VOI OLLA NEGATIIVISIA, PITÄÄ SKAALATA
    for point in dst_points:
        point_h = np.array([point[0], point[1], 1])
        transformed = np.dot(matrix, point_h)
        transformed /= transformed[2]
        transformed_points.append((transformed[0], transformed[1]))
    transformed_points = np.array(transformed_points)

    # UUDET ALUEEN RAJAT
    min_x = int(min(transformed_points[:, 0]))
    max_x = int(max(transformed_points[:, 0]))
    min_y = int(min(transformed_points[:, 1]))
    max_y = int(max(transformed_points[:, 1]))

    offset_x = -min_x if min_x < 0 else 0
    offset_y = -min_y if min_y < 0 else 0

    uusi_width = max_x - min_x
    uusi_height = max_y - min_y

    dst_points_adjusted = dst_points + np.float32([offset_x, offset_y])

    matrix_adjusted = cv2.getPerspectiveTransform(src_points, dst_points_adjusted)

    ylin_piste = min_y

    return matrix_adjusted, uusi_width, hintalaput,hyllyjen_maara


def find_transform_points(data, rakennuspiste, max_x, ylapuoli=True, kulmakerroin_pos=True):
    import time
    y = rakennuspiste[1]
    A = 0
    b = rakennuspiste[1]
    kulmakerroin_saavutettu = False
    start_time = time.time()
    while not kulmakerroin_saavutettu:

        if time.time() - start_time > 1.5:  # 1 sekunnin aikaraja
            print("Loop kesti yli sekunnin, breikataan pois.")
            A = 0
            b = rakennuspiste[1]
            return A, b

        for point in data:

            # EI HUOMIOIDA RAKENNUSPISTETTÄ
            if rakennuspiste[0] == point[0] and rakennuspiste[1] == point[1]:
                continue

            if ylapuoli:
                if A * point[0] + b < point[1]:
                    print(A * point[0] + b)
                    print(point)
                    kulmakerroin_saavutettu = True
                    break
            else:
                if A * point[0] + b > point[1]:
                    kulmakerroin_saavutettu = True
                    break

        if not kulmakerroin_saavutettu:

            if kulmakerroin_pos:

                A += 1 / max_x

            else:

                A -= 1 / max_x

            b = rakennuspiste[1] - A * rakennuspiste[0]

    if abs(A) > 0.4:
        print("KULMAKERTOIMEKSI TULOSS", A, "SKIPATAAN")
        A = 0
        b = rakennuspiste[1]

    x = np.linspace(0, max_x, 100)
    y = A * x + b
    #plt.plot(x, y)
    #plt.title(str(A)+" * "+"x"+" + "+str(b))
    data = np.array(data)
    print("HINTALAPPUJA", len(data))
    x = data[:, 0]
    y = data[:, 1]
    #plt.scatter(x,y)
    #plt.show()
    print("PALAUTETTAVA FUNKTIO:", A, b)
    return A, b


def kmeans_initial_and_clustered_objects(hintalaput, kokoleveys, kokokorkeus):
    koordinaatit = []

    for lappu in hintalaput.keys():
        koordinaatit.append([hintalaput[lappu]["left"] / (kokoleveys * 2),
                             (-hintalaput[lappu]["top"] - hintalaput[lappu]["height"]) / (kokokorkeus / 60)])

    data = np.array(koordinaatit)

    #x_arvot = [koord[0] for koord in koordinaatit]
    # y_arvot = [koord[1] for koord in koordinaatit]
    # plt.scatter(x_arvot, y_arvot)

    # plt.show()
    if len(data) < 2:
        clusterer = HDBSCAN(min_cluster_size=2, cluster_selection_epsilon=2)
        cluster_labels = [0] * len(data)
    else:
        clusterer = HDBSCAN(min_cluster_size=min(len(data), 2), cluster_selection_epsilon=2)
        cluster_labels = clusterer.fit_predict(data)
    #plt.scatter(data[:, 0], data[:, 1], c=cluster_labels, cmap='viridis', s=50)
    #plt.title(len(set(cluster_labels)))
    #plt.colorbar(label='Klusteri')
    #plt.show()
    hintalaput_hyllyissa = {}
    count = 0
    should_delete = []
    for lappu in hintalaput.keys():
        hylly = cluster_labels[count]
        if hylly == -1:
            should_delete.append(lappu)
            count += 1
            continue
        if hylly not in hintalaput_hyllyissa.keys():
            hintalaput_hyllyissa[hylly] = [(int(hintalaput[lappu]["left"] + hintalaput[lappu]["width"] / 2),
                                            hintalaput[lappu]["top"] + hintalaput[lappu]["height"])]
        else:
            hintalaput_hyllyissa[hylly].append((int(hintalaput[lappu]["left"] + hintalaput[lappu]["width"] / 2),
                                                hintalaput[lappu]["top"] + hintalaput[lappu]["height"]))
        count += 1
    for i in should_delete:
        if len(hintalaput) > 2 and "hintalappu" in i:
            # print("POISTETAAN ",i)
            del hintalaput[i]
    kmeans_init = []
    for hylly in hintalaput_hyllyissa.keys():
        koords = np.array(hintalaput_hyllyissa[hylly])
        sarakkeiden_keskiarvot = np.mean(koords, axis=0).tolist()
        kmeans_init.append(sarakkeiden_keskiarvot)
    kmeans_init = np.array(kmeans_init)


    if len(list(kmeans_init)):
        kmeans_init = kmeans_init[kmeans_init[:, 1].argsort()]
    else:
        kmeans_init = "k-means++"
    hyllyt = set(cluster_labels)
    if -1 in hyllyt and len(hyllyt) > 1:
        hyllyt.remove(-1)
    hyllyjen_maara = len(hyllyt)
    print("Hyllyjen maara", hyllyjen_maara)
    return hintalaput, hintalaput_hyllyissa, kmeans_init, hyllyjen_maara


def muut_tuotteet_filter(hyllyn_sisalto):
    """

    :param hyllyn_sisalto:
    :return: hyllyn sisalto, josta poistettu muut tuotteet, mikäli ne ei ole kategorian kanssa samassa "HYLLYSSÄ"
    """
    filtered_hyllyn_sisalto = hyllyn_sisalto.copy()

    # jos jokin hylly on täynnä pelkkää muut_tuotteet=> ei huomioida SoS:ssa
    for hylly in filtered_hyllyn_sisalto.keys():
        hyllyssa_kategoriaa = False
        for tuote in filtered_hyllyn_sisalto[hylly]:
            tuotenimi = tuote["tuotenimi"].split(" ")[0]
            tuotenimi = tuotenimi.replace("corega", "muut_tuotteet")
            tuotenimi = tuotenimi.replace("oralhygiene", "muut_tuotteet")
            tuotenimi = tuotenimi.replace("oralhygiene", "muut_tuotteet")

            if "muut_tuotteet" not in tuotenimi:
                hyllyssa_kategoriaa = True
                break
        if hyllyssa_kategoriaa == False:
            filtered_hyllyn_sisalto[hylly] = []

    # etsitaan reunimmaiset kategorian tuotteet ja filteröidään niiden ulkopuoliset autogen tuotteet.
    vasen_laita = 10000
    oikea_laita = 0
    borgesit = 0
    for hylly in filtered_hyllyn_sisalto.keys():

        for tuote in filtered_hyllyn_sisalto[hylly]:
            if "Borges" in tuote["tuotenimi"]:
                borgesit += tuote["kpl"]
            if "muut_tuotteet" not in tuote["tuotenimi"]:
                if tuote["left"] < vasen_laita:
                    vasen_laita = tuote["left"]
                if tuote["left"] + tuote["width"] > oikea_laita:
                    oikea_laita = tuote["left"] + tuote["width"]
    # filteröidään reunoissa olevat muut_tuotteet pois
    print(vasen_laita, oikea_laita)
    print("BORGESIA", borgesit)
    # for hylly in filtered_hyllyn_sisalto.keys():
    #    tuotelista = [tuote for tuote in filtered_hyllyn_sisalto[hylly] if
    #                  tuote["left"] + tuote["width"] / 2 > vasen_laita
    #                  and tuote["left"] + tuote["width"] / 2 < oikea_laita]#
    #
    #    filtered_hyllyn_sisalto[hylly] = tuotelista

    for hylly in filtered_hyllyn_sisalto.keys():
        # poistetaan reunimmaiset muutt tuotteet

        while True:

            if len(filtered_hyllyn_sisalto[hylly]) == 0:
                break
            if "muut" in filtered_hyllyn_sisalto[hylly][0]["tuotenimi"]:
                print("poistetaan", filtered_hyllyn_sisalto[hylly][0]["tuotenimi"])
                filtered_hyllyn_sisalto[hylly].pop(0)

            elif "muut" in filtered_hyllyn_sisalto[hylly][-1]["tuotenimi"]:
                print("poistetaan", filtered_hyllyn_sisalto[hylly][-1]["tuotenimi"])
                filtered_hyllyn_sisalto[hylly].pop()

            elif "oralhygiene" in filtered_hyllyn_sisalto[hylly][-1]["tuotenimi"]:
                print("poistetaan", filtered_hyllyn_sisalto[hylly][-1]["tuotenimi"])
                filtered_hyllyn_sisalto[hylly].pop()

            elif "oralhygiene" in filtered_hyllyn_sisalto[hylly][0]["tuotenimi"]:
                print("poistetaan", filtered_hyllyn_sisalto[hylly][0]["tuotenimi"])
                filtered_hyllyn_sisalto[hylly].pop(0)

            elif "corega" in filtered_hyllyn_sisalto[hylly][0]["tuotenimi"]:
                print("poistetaan", filtered_hyllyn_sisalto[hylly][0]["tuotenimi"])
                filtered_hyllyn_sisalto[hylly].pop(0)

            elif "corega" in filtered_hyllyn_sisalto[hylly][-1]["tuotenimi"]:
                print("poistetaan", filtered_hyllyn_sisalto[hylly][-1]["tuotenimi"])
                filtered_hyllyn_sisalto[hylly].pop()
            else:
                break

    return filtered_hyllyn_sisalto


def overlap_filter(hyllyn_sisalto):
    """
    poistetaan tuotteet jotka ovat duplikaatteja johtuen liiallisesta overlapista
    :param hyllyn_sisalto:
    :param douple_matces: {hylly: [ylimaarainen_tuote1, ylimaarainen_tuote2...] }
    :return:
    """
    douple_matces = {}

    for hylly in hyllyn_sisalto.keys():
        print("KÄSITELLÄÄN HYLLYÄ", hylly)

        for i in range(len(hyllyn_sisalto[hylly])):
            tuote = hyllyn_sisalto[hylly][i]

            tuotenimi = tuote["tuotenimi"].split(" ")[0]
            tuote_lkm = tuote["kpl"]

            for j in range(i + 1, len(hyllyn_sisalto[hylly])):
                muutuote = hyllyn_sisalto[hylly][j]
                muutuotenimi = muutuote["tuotenimi"].split(" ")[0]

                muutuote_lkm = muutuote["kpl"]

                if muutuotenimi == tuotenimi:
                    # poistetaan aina se alue, kummassa on vähemmän tuotteita

                    if hylly not in douple_matces:
                        if tuote_lkm >= muutuote_lkm:
                            douple_matces[hylly] = [muutuote["tuotenimi"]]
                        else:
                            douple_matces[hylly] = [tuote["tuotenimi"]]
                    else:
                        if tuote_lkm >= muutuote_lkm:
                            douple_matces[hylly].append(muutuote["tuotenimi"])
                        else:
                            douple_matces[hylly].append(tuote["tuotenimi"])

    for hylly in douple_matces.keys():
        # HYLLY JOSSA AINAKIN YKSI DUPLIKAATTI
        hyllyn_sisalto[hylly] = [
            tuote for tuote in hyllyn_sisalto[hylly]
            if tuote["tuotenimi"] not in douple_matces[hylly]
        ]

    return hyllyn_sisalto


def yhdista_vierekkaiset_samat_tuotteet(hyllynsialto):
    """

    :param hyllynsialto: alkuperäien hyllyn sisältö
    :return: hyllynsisältö, jossa vierekkäiset samat skut yhdistetty
    """

    poistettavat = []
    for hylly in hyllynsialto:

        hyllyntuotteet = hyllynsialto[hylly].copy()

        for i in range(len(hyllyntuotteet)):
            if hyllyntuotteet[i] in poistettavat:
                continue
            if i + 1 >= len(hyllyntuotteet):
                continue
            add_i = 0

            while i + add_i + 1 < len(hyllyntuotteet) and hyllyntuotteet[i]["tuotenimi"].split(" ")[0] == \
                    hyllyntuotteet[i + add_i + 1]["tuotenimi"].split(" ")[0]:
                print("RUNDI MENOSSA", hyllyntuotteet[i]["tuotenimi"])
                top = min(hyllyntuotteet[i]["top"], hyllyntuotteet[i + add_i + 1]["top"])
                left = hyllyntuotteet[i]["left"]
                width = hyllyntuotteet[i + add_i + 1]["left"] + hyllyntuotteet[i + add_i + 1]["width"] - left
                height = max(hyllyntuotteet[i]["top"] + hyllyntuotteet[i]["height"],
                             hyllyntuotteet[i + add_i + 1]["top"] + hyllyntuotteet[i + add_i + 1]["height"]) - top

                if hyllyntuotteet[i]["top"] + hyllyntuotteet[i]["height"] / 2 < hyllyntuotteet[i + add_i + 1][
                    "top"]:
                    hyllyntuotteet[i]["tuotekerrokset"] += hyllyntuotteet[i + add_i + 1]["tuotekerrokset"]

                elif hyllyntuotteet[i + add_i + 1]["top"] + hyllyntuotteet[i + add_i + 1]["height"] / 2 < \
                        hyllyntuotteet[i]["top"]:
                    hyllyntuotteet[i]["tuotekerrokset"] += hyllyntuotteet[i + add_i + 1]["tuotekerrokset"]

                hyllyntuotteet[i]["top"] = top
                hyllyntuotteet[i]["left"] = left
                hyllyntuotteet[i]["height"] = height
                hyllyntuotteet[i]["width"] = width
                hyllyntuotteet[i]["kpl"] += hyllyntuotteet[i + add_i + 1]["kpl"]
                hyllyntuotteet[i]["kpl_raw"] += hyllyntuotteet[i + add_i + 1]["kpl_raw"]

                poistettavat.append(hyllyntuotteet[i + add_i + 1])
                print("POISTETTAVIIN", hyllyntuotteet[i + add_i + 1]["tuotenimi"])

                if add_i > 3:
                    break
                add_i += 1
        hyllynsialto[hylly] = hyllyntuotteet
    for hylly in hyllynsialto.keys():
        for poistettava in poistettavat:

            if poistettava in hyllynsialto[hylly]:
                # print("POISTETAAN",poistettava["tuotenimi"])
                hyllynsialto[hylly].remove(poistettava)

    return hyllynsialto


def yhdista_valituotteet_yhdeksi(hyllyn_sisalto):
    for hylly in hyllyn_sisalto.keys():
        hyllylista = hyllyn_sisalto[hylly]
        for i in range(len(hyllyn_sisalto[hylly])):

            if i == 0 or i + 1 == len(hyllylista):
                continue
            if hyllylista[i + 1]["tuotenimi"] == "hellmanns_625ml_original_majonees 44":
                hyllyn_sisalto[hylly][i + 1]["tuotenimi"] = "hellmanns_900ml_original_majonees 44"
                hyllylista[i + 1]["tuotenimi"] = "hellmanns_900ml_original_majonees 44"
            edellinen_tuote = hyllylista[i - 1]["tuotenimi"].split(" ")[0]
            seuraava_tuote = hyllylista[i + 1]["tuotenimi"].split(" ")[0]
            tuote = hyllylista[i]["tuotenimi"].split(" ")[0]
            id = tuote[1]

            """TARKISTETAAN että käsiteltävät tuotteet ovat vierekkäin. Jos joku tuotteista on jonkun tuotteen päällä
            => tulee virheitä"""

            ed_tuote_top = hyllylista[i - 1]["top"]
            ed_tuote_low = hyllylista[i - 1]["top"] + hyllylista[i - 1]["height"]

            tuote_mid = hyllylista[i]["top"] + hyllylista[i]["height"] / 2

            se_tuote_top = hyllylista[i + 1]["top"]
            se_tuote_low = hyllylista[i + 1]["top"] + hyllylista[i + 1]["height"]

            if tuote_mid < ed_tuote_top or tuote_mid > ed_tuote_low:
                continue
            if tuote_mid < se_tuote_top or tuote_mid > se_tuote_low:
                continue
            """Vaaditaan myös, että ympäröivien tuotteiden lukumäärä pitää olla isompi kuin muutettavan"""
            if hyllylista[i - 1]["kpl"] + hyllylista[i + 1]["kpl"] <= hyllylista[i]["kpl"]:
                continue
            "JOS TUOTTEIDEN NIMET MÄTSÄÄ, YHDISTETÄÄN"
            if edellinen_tuote == seuraava_tuote and "muut_" not in edellinen_tuote and "muu_" not in edellinen_tuote:

                if tuote != edellinen_tuote:
                    print("TUOTE", hyllyn_sisalto[hylly][i]["tuotenimi"], "MUUTTUU")
                    hyllyn_sisalto[hylly][i]["tuotenimi"] = edellinen_tuote + " " + id

    return hyllyn_sisalto