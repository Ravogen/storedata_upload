import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
import math
import random
import cv2
from sklearn.cluster import KMeans


def hyllyorientaatio(hyllyn_sisalto):
    for hylly in hyllyn_sisalto.keys():
        hyllylista = hyllyn_sisalto[hylly]
        for i in range(len(hyllyn_sisalto[hylly])):

            if i == 0 or i + 1 == len(hyllylista):
                continue
            edellinen_tuote = hyllylista[i - 1]["tuotenimi"].split(" ")[0]
            seuraava_tuote = hyllylista[i + 1]["tuotenimi"].split(" ")[0]
            tuote = hyllylista[i]["tuotenimi"].split(" ")
            id = tuote[1]
            tuote = tuote[0]

            if edellinen_tuote == seuraava_tuote and "muut_" not in edellinen_tuote and "muu_" not in edellinen_tuote:
                if tuote != edellinen_tuote:
                    hyllyn_sisalto[hylly][i]["tuotenimi"] = edellinen_tuote + " " + id

    tuoteblokit = {}
    for hylly in hyllyn_sisalto.keys():
        tuoteblokit[hylly] = {}
        for i in range(len(hyllyn_sisalto[hylly])):
            tuote = hyllyn_sisalto[hylly][i]

            try:
                if "_" in tuote["tuotenimi"]:
                    tuotenimi = tuote["tuotenimi"].split("_")[0]
                else:
                    tuotenimi = tuote["tuotenimi"].split(" ")[0]
            except:
                tuotenimi = tuote["tuotenimi"].split(" ")[0]
            if tuotenimi == "Auto":
                continue

            if tuotenimi not in tuoteblokit[hylly]:
                tuoteblokit[hylly][tuotenimi] = {}
                block_left = tuote["left"]
                block_top = tuote["top"]
                block_bottom = tuote["top"] + tuote["height"]
                block_right = tuote["left"] + tuote["width"]
                tuoteblokit[hylly][tuotenimi]["left"] = block_left
                tuoteblokit[hylly][tuotenimi]["top"] = block_top
                tuoteblokit[hylly][tuotenimi]["bottom"] = block_bottom
                tuoteblokit[hylly][tuotenimi]["right"] = block_right
                continue

            block_left = tuote["left"]
            block_top = tuote["top"]
            block_bottom = tuote["top"] + tuote["height"]
            block_right = tuote["left"] + tuote["width"]
            if block_left < tuoteblokit[hylly][tuotenimi]["left"]:
                tuoteblokit[hylly][tuotenimi]["left"] = block_left

            if block_top < tuoteblokit[hylly][tuotenimi]["top"]:
                tuoteblokit[hylly][tuotenimi]["top"] = block_top

            if block_bottom > tuoteblokit[hylly][tuotenimi]["bottom"]:
                tuoteblokit[hylly][tuotenimi]["bottom"] = block_bottom
            tuoteblokit[hylly][tuotenimi]["right"] = block_right

    brandiblokit = {}
    for hylly in tuoteblokit.keys():
        for brandi in tuoteblokit[hylly].keys():
            if brandi in brandiblokit.keys():
                brandiblokit[brandi]["width"].append(
                    tuoteblokit[hylly][brandi]["right"] - tuoteblokit[hylly][brandi]["left"])
                brandiblokit[brandi]["top"] = min(tuoteblokit[hylly][brandi]["top"], brandiblokit[brandi]["top"])
                brandiblokit[brandi]["bottom"] = max(tuoteblokit[hylly][brandi]["bottom"],
                                                     brandiblokit[brandi]["bottom"])
            else:
                brandiblokit[brandi] = {
                    "width": [tuoteblokit[hylly][brandi]["right"] - tuoteblokit[hylly][brandi]["left"]],
                    "top": tuoteblokit[hylly][brandi]["top"],
                    "bottom": tuoteblokit[hylly][brandi]["bottom"]}
    brandiorientaatio = {}
    for brandi in brandiblokit:

        keskileveys = sum(brandiblokit[brandi]["width"]) / len(brandiblokit[brandi]["width"])
        if keskileveys > brandiblokit[brandi]["bottom"] - brandiblokit[brandi]["top"]:
            brandiorientaatio[brandi] = "horizontal"
        else:
            brandiorientaatio[brandi] = "vertical"
    for i in brandiorientaatio.keys():
        print(i, brandiorientaatio[i])

    return brandiorientaatio


def hyllyhahmotelma(hyllyn_sisalto, hintalaput=None, tuote_taso=True):
    for hylly in hyllyn_sisalto.keys():
        hyllylista = hyllyn_sisalto[hylly]
        for i in range(len(hyllyn_sisalto[hylly])):

            if i == 0 or i + 1 == len(hyllylista):
                continue
            edellinen_tuote = hyllylista[i - 1]["tuotenimi"].split(" ")[0]
            seuraava_tuote = hyllylista[i + 1]["tuotenimi"].split(" ")[0]
            tuote = hyllylista[i]["tuotenimi"].split(" ")
            id = tuote[1]
            tuote = tuote[0]

            if edellinen_tuote == seuraava_tuote:
                if tuote != edellinen_tuote:
                    hyllyn_sisalto[hylly][i]["tuotenimi"] = edellinen_tuote + " " + id
    count = 0
    print(hyllyn_sisalto)

    tuoteblokit = {}
    count = 0

    for hylly in hyllyn_sisalto.keys():
        tuoteblokit[hylly] = {}
        edellinen_tuote = ""
        for i in range(len(hyllyn_sisalto[hylly])):
            tuote = hyllyn_sisalto[hylly][i]

            tuotenimi = tuote["tuotenimi"].split(" ")[0]
            if tuote_taso:
                try:
                    if "_" in tuote["tuotenimi"]:
                        tuotenimi = tuote["tuotenimi"].split("_")[0]
                    else:
                        tuotenimi = tuote["tuotenimi"].split(" ")[0]
                except:
                    tuotenimi = tuote["tuotenimi"].split(" ")[0]
                if tuotenimi == "Auto":
                    continue
            print(".......")
            tuotenimi = tuotenimi + str(count)
            print(edellinen_tuote)
            print(tuotenimi)
            if edellinen_tuote != tuotenimi:

                if tuotenimi in tuoteblokit[hylly]:
                    count += 1
                    tuotenimi = tuotenimi + str(count)
            print(tuotenimi)

            if tuotenimi not in tuoteblokit[hylly]:
                tuoteblokit[hylly][tuotenimi] = {}
                block_left = tuote["left"]
                block_top = tuote["top"]
                block_bottom = tuote["top"] + tuote["height"]
                block_right = tuote["left"] + tuote["width"]
                tuoteblokit[hylly][tuotenimi]["left"] = block_left
                tuoteblokit[hylly][tuotenimi]["top"] = block_top
                tuoteblokit[hylly][tuotenimi]["bottom"] = block_bottom
                tuoteblokit[hylly][tuotenimi]["right"] = block_right
                edellinen_tuote = tuotenimi
                continue

            block_left = tuote["left"]
            block_top = tuote["top"]
            block_bottom = tuote["top"] + tuote["height"]
            block_right = tuote["left"] + tuote["width"]

            if block_left < tuoteblokit[hylly][tuotenimi]["left"]:
                tuoteblokit[hylly][tuotenimi]["left"] = block_left

            if block_top < tuoteblokit[hylly][tuotenimi]["top"]:
                tuoteblokit[hylly][tuotenimi]["top"] = block_top

            if block_bottom > tuoteblokit[hylly][tuotenimi]["bottom"]:
                tuoteblokit[hylly][tuotenimi]["bottom"] = block_bottom
            tuoteblokit[hylly][tuotenimi]["right"] = block_right
            edellinen_tuote = tuotenimi

    data = tuoteblokit
    print("BLOKIT")
    print(tuoteblokit)
    items = [(shelf, brand, coords) for shelf, brands in data.items() for brand, coords in brands.items()]

    # Get a colormap with enough colors
    num_items = len(items)
    # fig, ax = plt.subplots(1)
    for i, (shelf, brand, coords) in enumerate(items):
        print(shelf, brand, coords)
        left = coords['left']
        top = coords['top']
        bottom = coords['bottom']
        right = coords['right']
        width = right - left
        height = bottom - top
        rect = patches.Rectangle((left, top), width, height, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        ax.text(left, top, f"{brand}", color='black')

    for key in hintalaput.keys():
        rect = patches.Rectangle((hintalaput[key]["left"], hintalaput[key]["top"]), hintalaput[key]["width"],
                                 hintalaput[key]["height"], linewidth=2, edgecolor='red', facecolor='red')
        ax.add_patch(rect)
        ax.text(left, top, f"{brand}", color='red')

    plt.xlim(0, 8000)  # Aseta x-akselin rajoitus (tarpeen mukaan)
    plt.ylim(4000, 0)  # Y-akselin rajoitus, käänteinen (0 ylhäällä)
    # plt.show()
    # plt.show()


def bresenham_line(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        # Lisätään nykyinen piste listaan
        points.append([x0, y0])

        # Tarkistetaan, ollaanko lopullisessa pisteessä
        if x0 == x1 and y0 == y1:
            break

        # Lasketaan virhetermillä seuraava piste
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy

    return points


def hyllyhahmotelma_ilmanhyllyja(tuotteet, hintalaput, hintalaput_hyllyissa, hyllyjen_maara, kmeans_initial="k-means++",
                                 branditasolla=False, plot=False):
    import time
    start = time.time()
    if branditasolla:
        keys = list(tuotteet.keys())
        for key in keys:
            tuote = tuotteet[key]
            if "_" in key:
                id = key.split(" ")[1]
                brandi = key.split("_")[0]

                del tuotteet[key]
                tuotteet[brandi + " " + id] = tuote
    tuote_ka_leveys = {}
    for tuote in tuotteet.keys():
        tuotteet[tuote]["kpl"] = 1
        tuotenimi = tuote.split(" ")[0]
        if tuotenimi in tuote_ka_leveys.keys():
            tuote_ka_leveys[tuotenimi].append(tuotteet[tuote]["width"])
        else:
            tuote_ka_leveys[tuotenimi] = [tuotteet[tuote]["width"]]
    #############################################################
    tuote_ka_korkeus = {}
    for tuote in tuotteet.keys():

        tuotenimi = tuote.split(" ")[0]
        if tuotenimi in tuote_ka_korkeus.keys():
            tuote_ka_korkeus[tuotenimi].append(tuotteet[tuote]["height"])
        else:
            tuote_ka_korkeus[tuotenimi] = [tuotteet[tuote]["height"]]
    ########################################################

    tuotteet_original = tuotteet.copy()
    koko_hyllyn_ka_leveys = 0
    lkm = 0
    for tuote in tuotteet.keys():
        koko_hyllyn_ka_leveys += tuotteet[tuote]["width"]
        lkm += 1
    if lkm == 0:
        return {"0": []}, {}
    koko_hyllyn_ka_leveys = koko_hyllyn_ka_leveys / lkm

    keys = list(tuotteet.keys())
    print("START")
    jo_kasitellyt = []
    xlim = 0
    ylim = 0
    max_prob = 0
    print(len(tuotteet.keys()) - 1, hyllyjen_maara, "HAHAA")
    for key in keys:

        if key in jo_kasitellyt:
            continue
        tuote = tuotteet[key]
        if "Auto" in key:
            continue
        if tuote["top"] + tuote["height"] > ylim:
            ylim = tuote["top"] + tuote["height"]
        if tuote["left"] + tuote["width"] > xlim:
            xlim = tuote["left"] + tuote["width"]

        x_1 = tuote["left"] + tuote["width"]
        y_1 = tuote["top"] + tuote["height"]

        while True:
            min_dis = 10000
            lahin_tuote = ""
            lahin_tuotenimi = ""
            for key2 in keys:

                if key2 in jo_kasitellyt:
                    continue
                muu_tuote = tuotteet[key2]

                if muu_tuote["top"] + muu_tuote["height"] > ylim:
                    ylim = muu_tuote["top"] + muu_tuote["height"]
                if muu_tuote["left"] + muu_tuote["width"] > xlim:
                    xlim = muu_tuote["left"] + muu_tuote["width"]

                if key == key2:
                    continue
                x_2 = muu_tuote["left"]
                y_2 = muu_tuote["top"] + muu_tuote["height"]

                etaisyys = math.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)

                if etaisyys < min_dis:
                    min_dis = etaisyys
                    lahin_tuote = muu_tuote
                    lahin_tuotenimi = key2
            if len(tuotteet.keys()) - 1 < hyllyjen_maara:
                """
                paallekkain olevia tuotteita ei yhdistetä, jos tuotteiden määrä tippuu alle hyllymäärän.
                se tarkoittaisi tod.näk. eri hyllyissä olevien tuotteiden yhdistämistä ja johtaisi 
                virheisiin k-meansissa
                """
                break
            if key.split(" ")[0] == lahin_tuotenimi.split(" ")[0]:

                left = min(lahin_tuote["left"], tuote["left"])
                right = max(tuote["left"] + tuote["width"], lahin_tuote["left"] + lahin_tuote["width"])
                width = right - left
                top = min(lahin_tuote["top"], tuote["top"])
                bottom = max(tuote["top"] + tuote["height"], lahin_tuote["top"] + lahin_tuote["height"])
                height = bottom - top
                hintalappu_tuotteen_sisalla = False
                for hintalappu in hintalaput.keys():
                    hintalappu = hintalaput[hintalappu]
                    hintalappu_left = hintalappu["left"]
                    hintalappu_width = hintalappu["width"]
                    hintalappu_top = hintalappu["top"]
                    hintalappu_height = hintalappu["height"]
                    if hintalappu_top + hintalappu_height > ylim:
                        ylim = hintalappu_top + hintalappu_height
                    if hintalappu_left + hintalappu_width > xlim:
                        xlim = hintalappu_left + hintalappu_width

                    if hintalappu_left > left and hintalappu_top > top:
                        if hintalappu_left + hintalappu_width < left + width and hintalappu_top + hintalappu_height < top + height:
                            hintalappu_tuotteen_sisalla = True

                    if hintalappu_left + hintalappu_width > left and hintalappu_top > top:
                        if hintalappu_left + hintalappu_width < left + width and hintalappu_top + hintalappu_height < top + height:
                            hintalappu_tuotteen_sisalla = True
                if hintalappu_tuotteen_sisalla:
                    break

                total_prob = float(round(tuote["prob"] + lahin_tuote["prob"]))
                kpl = tuote["kpl"] + lahin_tuote["kpl"]
                if total_prob > max_prob:
                    max_prob = total_prob
                tuotteet[key] = {'top': top, 'height': height, 'left': left, 'width': width, "prob": total_prob,
                                 "kpl": kpl,"EAN":tuote["EAN"]}
                tuote = tuotteet[key]

                x_1 = tuote["left"] + tuote["width"]
                y_1 = tuote["top"] + tuote["height"]

                del tuotteet[lahin_tuotenimi]
                jo_kasitellyt.append(lahin_tuotenimi)
            else:

                break
    for tuote in tuotteet.keys():
        tuotenimi = tuote.split(" ")[0]
        tuotteet[tuote]["ka_leveys"] = round(sum(tuote_ka_leveys[tuotenimi]) / len(tuote_ka_leveys[tuotenimi]))
        tuotteet[tuote]["ka_korkeus"] = round(sum(tuote_ka_korkeus[tuotenimi]) / len(tuote_ka_korkeus[tuotenimi]))

    # päälekkäin samassa hyllyssä olevat tuotteet
    keys = list(tuotteet.keys())
    print(len(tuotteet.keys()) - 1, hyllyjen_maara, "HÖHÖÖ")
    jo_kasitellyt = []
    for key in keys:
        if key in jo_kasitellyt:
            continue
        tuote = tuotteet[key]
        if "Auto" in tuote:
            continue
        tuote_left = tuote["left"]
        tuote_width = tuote["width"]
        tuote_top = tuote["top"]
        tuote_height = tuote["height"]
        for key2 in keys:
            if key2 in jo_kasitellyt:
                continue
            muutuote = tuotteet[key2]
            if key == key2:
                continue
            muutuote_left = muutuote["left"]
            muutuote_width = muutuote["width"]
            muutuote_top = muutuote["top"]
            muutuote_height = muutuote["height"]
            muutuote_x = muutuote_left + muutuote_width / 2
            if muutuote_x > tuote_left and muutuote_x < tuote_left + tuote_width:
                if key.split(" ")[0] == key2.split(" ")[0]:

                    # TODO
                    """
                    tarkistus, päällekkäin olevia tuotteita ei yhdistettäisi, mikäli kahden saman hyllyn
                    hintalapun välinen viiva menee tuotteiden välistä.
                    Käytetään bresenhamnin algoritmiä
                    """
                    print("#####")

                    if len(tuotteet.keys()) - 1 < hyllyjen_maara:
                        """
                        paallekkain olevia tuotteita ei yhdistetä, jos tuotteiden määrä tippuu alle hyllymäärän.
                        se tarkoittaisi tod.näk. eri hyllyissä olevien tuotteiden yhdistämistä ja johtaisi 
                        virheisiin k-meansissa
                        """
                        continue

                    if tuote_top + tuote_height >= muutuote_top:

                        if tuote_top < muutuote_top:

                            tuotteet[key]["left"] = min(tuote_left, muutuote_left)
                            tuotteet[key]["height"] = muutuote_top + muutuote_height - tuote_top
                            tuotteet[key]["width"] = max(tuote_left + tuote_width,
                                                         muutuote_left + muutuote_width) - min(tuote_left,
                                                                                               muutuote_left)
                            jo_kasitellyt.append(key2)
                            if "tuotekerrokset" not in tuotteet[key]:
                                if "tuotekerrokset" in tuotteet[key2]:
                                    tuotteet[key]["tuotekerrokset"] = tuotteet[key2]["tuotekerrokset"] + 1
                                else:
                                    tuotteet[key]["tuotekerrokset"] = 2
                            else:
                                tuotteet[key]["tuotekerrokset"] += 1
                            del tuotteet[key2]

    """
    for tuote in tuotteet.keys():
        tuote_left = tuote["left"]
        tuote_width = tuote["width"]
        tuote_top =tuote["top"]
        tuote_height=tuote["height"]

        for hintalappu in hintalaput.keys():
            hintalappu_left = hintalappu["left"]
            hintalappu_width = hintalappu["width"]
            hintalappu_top = hintalappu["top"]
            hintalappu_height = hintalappu["height"]

            if 

    """
    # KUVAN RAJOISSA PITÄÄ OTTAA HUOMIOON MYÖS HINTALAPPUJEN SIJAINNIT
    for lappu in hintalaput.keys():
        hintalaput[lappu]
        if hintalaput[lappu]["top"] + hintalaput[lappu]["height"] > ylim:
            ylim = hintalaput[lappu]["top"] + hintalaput[lappu]["height"]
    print(len(tuotteet.keys()) - 1, hyllyjen_maara)
    img_height = ylim
    img_width = xlim

    image = np.zeros((img_height, img_width, 1), dtype=np.int16)

    # varataan valkoinen hintalapuille, ja hyllyrajalle
    jo_valitut_varit = [[255], [254]]

    # prob_scale = 255 / max_prob

    def generate_unique_color():
        while True:
            color = [random.randint(0, 2550) for _ in range(1)]
            if color not in jo_valitut_varit:
                jo_valitut_varit.append(color)
                break

        return color

    for product, coords in tuotteet.items():
        top, left = coords['top'], coords['left']
        height, width = coords['height'], coords['width']
        if "Auto" in product:
            color = 253
            jo_valitut_varit.append(color)
        else:
            color = generate_unique_color()
        # color = []
        # for i in range(3):
        #    color.append(coords["prob"] * prob_scale)
        if top >= 0:
            image[top:top + height, left:left + width] = color
        else:
            # TÄSSÄ TAPAUKSESSA KUVAN VÄÄNTÖ ON LUONUT NEGATIIVISIA KOORDINAATTEJA
            image[0:0 + height + top, left:left + width] = color

    for product, coords in hintalaput.items():
        top, left = coords['top'], coords['left']
        height, width = coords['height'], coords['width']

        color = [255]

        image[top:top + height, left:left + width] = color
    column_sums = np.sum(image, axis=(0, 2))
    zero_sum_columns = np.where(column_sums == 0)[0]

    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    for hylly in hintalaput_hyllyissa.keys():
        points = hintalaput_hyllyissa[hylly]
        points = sorted(points, key=lambda point: point[0])
        color = [254]  # Valkoinen väri (RGB)
        thickness = 1  # Viivan paksuus
        for i in range(len(points) - 1):
            start_point = points[i]
            end_point = points[i + 1]

            cv2.line(image, start_point, end_point, color, thickness)
        cv2.line(image, (0, points[0][1]), (points[0][0], points[0][1]), color, thickness)
        cv2.line(image, (points[-1][0], points[-1][1]), (img_width, points[0][1]), color, thickness)

    keys = list(tuotteet.keys())
    for key in keys:

        tuote = tuotteet[key]
        x_1 = tuote["left"] + tuote["width"]
        y_1 = tuote["top"] + tuote["height"]

        lahin_hintalappu_oikealla = None
        min_etaisyys_oikealla = xlim

        for hintalappu in hintalaput.keys():

            hintalappu = hintalaput[hintalappu]

            # OIKEALLA
            x_2 = hintalappu["left"]
            y_2 = hintalappu["top"]
            if x_1 - tuote["width"] > x_2 or y_2 < tuote["top"]:
                continue
            if x_1 > x_2 + hintalappu["width"] / 2 or y_2 < tuote["top"]:
                continue
            etaisyys_oik = math.sqrt((x_1 - x_2) ** 2 + (y_1 - y_2) ** 2)
            if etaisyys_oik < min_etaisyys_oikealla:
                min_etaisyys_oikealla = etaisyys_oik
                lahin_hintalappu_oikealla = hintalappu

        tuotteet[key]["hintalappu"] = lahin_hintalappu_oikealla
        # if lahin_hintalappu_vasemmalla != None and plot:
        #    axs[0].plot([x_1, lahin_hintalappu_oikealla["left"]], [y_1, lahin_hintalappu_oikealla["top"]], color='red')

        x_1_vasen = tuote["left"]
        y_1_vasen = tuote["top"] + tuote["height"]

        lahin_hintalappu_vasemmalla = None
        min_etaisyys_vasemmalla = xlim

        for hintalappu in hintalaput.keys():

            hintalappu = hintalaput[hintalappu]

            # VASEMMALLA
            # x_2 = hintalappu["left"] + hintalappu["width"]
            x_2 = hintalappu["left"]
            y_2 = hintalappu["top"]
            if hintalappu["left"] > x_1_vasen or y_2 < tuote["top"]:
                continue
            etaisyys_vas = math.sqrt((x_1_vasen - x_2) ** 2 + (y_1_vasen - y_2) ** 2)
            if etaisyys_vas < min_etaisyys_vasemmalla:
                min_etaisyys_vasemmalla = etaisyys_vas
                lahin_hintalappu_vasemmalla = hintalappu
        tuotteet[key]["hintalappu_vasen"] = lahin_hintalappu_vasemmalla
        if lahin_hintalappu_vasemmalla != None and plot:
            axs[0].plot([x_1_vasen, lahin_hintalappu_vasemmalla["left"]],
                        [y_1_vasen, lahin_hintalappu_vasemmalla["top"]], color='blue')

        lahin_hintalappu_keskella = None
        min_etaisyys_keskella = xlim

        for hintalappu in hintalaput.keys():

            hintalappu = hintalaput[hintalappu]

            # KESKELLÄ
            x_2 = hintalappu["left"]
            # x_2 = hintalappu["left"]
            y_2 = hintalappu["top"]
            if hintalappu["left"] > x_1_vasen + tuote["width"] or y_2 < tuote["top"]:
                continue
            etaisyys_kes = math.sqrt((x_1_vasen - x_2) ** 2 + (y_1_vasen - y_2) ** 2)
            if etaisyys_kes < min_etaisyys_keskella:
                min_etaisyys_keskella = etaisyys_kes
                lahin_hintalappu_keskella = hintalappu
        tuotteet[key]["hintalappu_keskella"] = lahin_hintalappu_keskella
        if lahin_hintalappu_vasemmalla != None and plot:
            axs[0].plot([x_1_vasen, lahin_hintalappu_keskella["left"]],
                        [y_1_vasen, lahin_hintalappu_keskella["top"]], color='green')

    print(len(tuotteet.keys()) - 1, hyllyjen_maara)
    if plot:
        axs[0].imshow(image)
        axs[0].set_title('Kuva 1')
    # laajennus
    keys = list(tuotteet.keys())
    for key in keys:
        tuote = tuotteet[key]
        top = tuote["top"]
        height = tuote["height"]
        left = tuote["left"]
        width = tuote["width"]

        color = image[int((top + height / 2)), int((left + width / 2))]
        if "Auto" in tuote:
            continue
        if tuote["hintalappu"] != None:

            kohde_x = tuote["hintalappu"]["left"]
            kohde_y = tuote["hintalappu"]["top"]
            lisattava_leveys = 0
            lisattava_korekus = 0

            # MAARITETAAN LISÄTTÄVÄ LEVEYS
            if kohde_x > left + width:
                for i in range(0, kohde_x - (left + width)):
                    if left + width + i + 1 in zero_sum_columns:
                        break

                    pystyvektori = image[max(0, top):top + height, left + width + i:left + width + i + 1]
                    # ei oteta huomioon jos oikealla tulee vastaan hintalappu tai hyllyraja
                    pystyvektori[np.all(pystyvektori == [255], axis=-1)] = [0]
                    pystyvektori[np.all(pystyvektori == [254], axis=-1)] = [0]

                    if np.sum(pystyvektori, axis=(0, 2)) != 0:

                        break
                    else:
                        image[top:top + height, left + width + i:left + width + i + 1] = color
                        lisattava_leveys = i

            elif kohde_x + tuote["hintalappu"]["width"] < left + width:
                count = -1

                while True:
                    break
                    count += 1
                    pystyvektori = image[top:top + height, left + width + count:left + width + count + 1]
                    # ei oteta huomioon jos oikealla tulee vastaan hintalappu
                    # pystyvektori[np.all(pystyvektori == [255, 255, 255], axis=-1)] = [0, 0, 0]
                    # ei oteta huomioon jos tulee hyllyraja
                    pystyvektori[np.all(pystyvektori == [254], axis=-1)] = [0]

                    if np.sum(pystyvektori, axis=(0, 2)) != 0 or left + width + count + 1 > xlim:

                        break
                    else:
                        image[top:top + height, left + width + count:left + width + count + 1] = color
                        lisattava_leveys = count

            # LISÄTTÄVÄ KORKEUS
            if kohde_y > top + height:
                for i in range(0, kohde_y - (top + height)):
                    if top < 0:
                        break
                    vaakavektoi = image[max(0, top) + height + i:max(0, top) + height + i + 1,
                                  left:left + width + lisattava_leveys + 1]

                    # if np.any(np.sum(vaakavektoi, axis=(1, 2)) != 0):

                    if np.sum(vaakavektoi, axis=(1, 2)) != 0:

                        break
                    else:
                        image[max(0, top) + height + i:max(0, top) + height + i + 1,
                        left:left + width + lisattava_leveys] = color
                        lisattava_korekus = i
            tuotteet[key]["height"] += lisattava_korekus + 1
            tuotteet[key]["width"] += lisattava_leveys + 1

        # VASEN PUOLI
        tuote = tuotteet[key]
        top = tuote["top"]
        height = tuote["height"]
        left = tuote["left"]
        width = tuote["width"]

        if tuote["hintalappu_vasen"] == None:
            continue
        if tuote["hintalappu_keskella"] != tuote["hintalappu_vasen"]:
            continue
        kohde_x = tuote["hintalappu_vasen"]["left"] + tuote["hintalappu_vasen"]["width"]
        kohde_y = tuote["hintalappu_vasen"]["top"]
        lisattava_korkeus = 0
        vahennetaan_leftista = 0

        # MAARITETAAN leftin vähennys
        if kohde_x < left:
            for i in range(0, left - kohde_x):

                pystyvektori = image[max(0, top):top + height, left - i - 1:left - i]
                pystyvektori[np.all(pystyvektori == [254], axis=-1)] = [0]
                # ei oteta huomioon jos oikealla tulee vastaan hintalappu
                if left - i - 1 in zero_sum_columns:
                    break
                if np.sum(pystyvektori, axis=(0, 2)) != 0:

                    break
                else:

                    image[max(0, top):top + height, left - i - 1:left - i] = color
                    vahennetaan_leftista = i

        tuotteet[key]["width"] = tuotteet[key]["left"] + tuotteet[key]["width"] - (
                tuotteet[key]["left"] - vahennetaan_leftista)
        tuotteet[key]["left"] -= vahennetaan_leftista

        # LISÄTTÄVÄ KORKEUS
        if kohde_y > top + height:
            for i in range(0, kohde_y - (top + height)):
                vaakavektoi = image[top + height + i:top + height + i + 1, left - vahennetaan_leftista:left + width]

                if np.sum(vaakavektoi, axis=(1, 2)) != 0:

                    break
                else:
                    image[top + height + i:top + height + i + 1, left - vahennetaan_leftista:left + width] = color
                    lisattava_korkeus = i
        tuotteet[key]["height"] += lisattava_korkeus

        # Plottaa rajattu alue
        # plt.imshow(pystyvektori)
        # plt.title(f'Rajattu alue: top={top}, left={left}, height={height}, width={width}')
        # plt.show()

    if plot:
        for key, value in tuotteet.items():
            # Luo suorakulmio annetuilla arvoilla
            rect = patches.Rectangle((value['left'], value['top']), value['width'], value['height'],
                                     linewidth=1, edgecolor='r', facecolor='none')
            axs[1].add_patch(rect)

        axs[1].imshow(image)
        axs[1].set_title('Kuva 2')
    laajennusend = time.time()
    print("LAAJENNUKSET AIKA:", laajennusend - start)

    """
    Nyt alueiden laajennukset on tehty. Seuraavaksi täytetään tyhjät kohdat.
    """
    keskimaarainen_korkeus = 0
    count = 0
    ylin_tuote = 2000
    for tuote in tuotteet.keys():
        tuote = tuotteet[tuote]
        keskimaarainen_korkeus += tuote["height"]
        count += 1
        if tuote["top"] < ylin_tuote:
            ylin_tuote = tuote["top"]
    keskimaarainen_korkeus = int(keskimaarainen_korkeus / count)

    other_color = generate_unique_color()
    muut_tuotteet = {}
    tuote_id = 0

    for hintalappu in hintalaput.keys():

        hintalappu = hintalaput[hintalappu]

        hintalappu_left = hintalappu["left"]
        hintalappu_width = hintalappu["width"]
        hintalappu_top = hintalappu["top"]
        hintalappu_height = hintalappu["height"]
        # if hintalappu_top<ylin_tuote:
        #    continue
        if image.shape[1] < hintalappu_left + hintalappu_width + 1:
            continue
        uuden_tuotteet_top = hintalappu_top - 1 - keskimaarainen_korkeus
        if uuden_tuotteet_top < 0:
            uuden_tuotteet_top = 0

        count = 0

        while True:
            if image.shape[1] < hintalappu_left + hintalappu_width + 2 + count:
                break
            pystyvektori = image[uuden_tuotteet_top:hintalappu_top,
                           hintalappu_left + hintalappu_width + 1 + count:hintalappu_left + hintalappu_width + 2 + count].copy()
            pystyvektori[np.all(pystyvektori == [255], axis=-1)] = [0]
            pystyvektori[np.all(pystyvektori == [254], axis=-1)] = [0]
            if hintalappu_left + hintalappu_width + 2 + count in zero_sum_columns:
                break
            if np.sum(pystyvektori) != 0:
                break
            image[uuden_tuotteet_top:  hintalappu_top,
            hintalappu_left + hintalappu_width + 1:  hintalappu_left + hintalappu_width + 2 + count] = other_color

            count += 1

        if count != 0:
            muut_tuotteet["muut_tuotteet" + "_auto " + str(tuote_id)] = {"top": uuden_tuotteet_top,
                                                                         "left": hintalappu_left + hintalappu_width + 1,
                                                                         "height": hintalappu_top - uuden_tuotteet_top,
                                                                         "width": count}

        tuote_id += 1

    tuotteet.update(muut_tuotteet)

    print(len(tuotteet.keys()) - 1, hyllyjen_maara)
    """
    seuraavan loopin jälkeen RGB matriisin käyttö ei ole enää mahdollista (ei ole synkassa tuotteet dictin kanssa)
    """
    jo_kasitellyt = []
    for key in keys:
        if key in jo_kasitellyt:
            continue
        tuote = tuotteet[key]
        tuote_left = tuote["left"]
        tuote_width = tuote["width"]
        tuote_top = tuote["top"]
        tuote_height = tuote["height"]
        for key2 in keys:
            if key2 in jo_kasitellyt:
                continue
            muutuote = tuotteet[key2]
            if key == key2:
                continue
            muutuote_left = muutuote["left"]
            muutuote_width = muutuote["width"]
            muutuote_top = muutuote["top"]
            muutuote_height = muutuote["height"]
            muutuote_x = muutuote_left + muutuote_width / 2
            if len(tuotteet.keys()) - 1 < hyllyjen_maara:
                """
                paallekkain olevia tuotteita ei yhdistetä, jos tuotteiden määrä tippuu alle hyllymäärän.
                se tarkoittaisi tod.näk. eri hyllyissä olevien tuotteiden yhdistämistä ja johtaisi 
                virheisiin k-meansissa
                """
                continue
            if muutuote_x > tuote_left and muutuote_x < tuote_left + tuote_width:
                if key.split(" ")[0] == key2.split(" ")[0]:

                    if tuote_top + tuote_height >= muutuote_top:

                        if tuote_top <= muutuote_top:

                            tuotteet[key]["left"] = min(tuote_left, muutuote_left)
                            tuotteet[key]["height"] = muutuote_top + muutuote_height - tuote_top
                            tuotteet[key]["width"] = max(tuote_left + tuote_width,
                                                         muutuote_left + muutuote_width) - min(tuote_left,
                                                                                               muutuote_left)
                            tuotteet[key]["prob"] += muutuote["prob"]
                            tuotteet[key]["prob"] = round(tuotteet[key]["prob"])
                            tuotteet[key]["kpl"] += muutuote["kpl"]
                            if "tuotekerrokset" not in tuotteet[key]:
                                if "tuotekerrokset" in tuotteet[key2]:
                                    tuotteet[key]["tuotekerrokset"] = tuotteet[key2]["tuotekerrokset"] + 1
                                else:
                                    tuotteet[key]["tuotekerrokset"] = 1
                            else:
                                tuotteet[key]["tuotekerrokset"] += 1
                            jo_kasitellyt.append(key2)
                            del tuotteet[key2]

    ryhjatend = time.time()
    print("TYHJATKOHDAT", ryhjatend - laajennusend)
    """
    aloitetaan hyllyjen maaritys
    """
    image = np.zeros((img_height, img_width, 1), dtype=np.int16)

    # varataan valkoinen hintalapuille
    jo_valitut_varit = [[255], [254]]

    for product, coords in tuotteet.items():
        top, left = coords['top'], coords['left']
        height, width = coords['height'], coords['width']

        color = generate_unique_color()

        image[top:top + height, left:left + width] = color

    for product, coords in hintalaput.items():
        top, left = coords['top'], coords['left']
        height, width = coords['height'], coords['width']

        color = [255]

        image[top:top + height, left:left + width] = color

    # TODO olennaisten alueiden määristys zero_sum_columnien avulla

    valit = []
    keskiarvolista = []
    if len(zero_sum_columns) > 1:
        for i in range(max(zero_sum_columns) + 2):

            if i in zero_sum_columns:
                keskiarvolista.append(i)

            elif len(keskiarvolista) != 0 and len(keskiarvolista) >= 15:
                print(keskiarvolista)
                ka = sum(keskiarvolista) / len(keskiarvolista)
                valit.append(ka)

                keskiarvolista = []
            else:
                keskiarvolista = []
    print("HAVAITUT VÄLIT")

    image_width = image.shape[1]

    jo_kasitellyt = []

    features = []
    for key, values in tuotteet.items():
        y = -(values['top'] + values['height'] / 2) / (ylim / 30)
        x = (values['left'] + values['width'] / 2) / (xlim)
        features.append([x, y])

    # Muunna NumPy-taulukoksi
    X = np.array(features)

    # 2. Klusterointi KMeansilla

    if len(kmeans_initial) == 0:
        kmeans_initial = "k-means++"

    if isinstance(kmeans_initial, np.ndarray):
        print("SKAALATAAN INITIALVALUET")
        kmeans_initial[:, 1] -= keskimaarainen_korkeus / 2

        kmeans_initial[:, 1] /= -1

        kmeans_initial[:, 1] /= ylim / 30
        kmeans_initial[:, 0] /= xlim
    # plt.show()
    # plt.scatter(X[:,0],X[:,1],marker="o")
    # plt.scatter(kmeans_initial[:, 0], kmeans_initial[:, 1], marker="*")
    # plt.show()

    try:
        kmeans = KMeans(n_clusters=hyllyjen_maara, random_state=42, init=kmeans_initial)  # valitse klusterien lukumäärä
        kmeans.fit(X)
    except:
        print("HUOM; HYLLYJÄ EI VOITU MÄÄRITTÄÄ")
        kmeans = KMeans(n_clusters=min(len(X), hyllyjen_maara), random_state=42)  # valitse klusterien lukumäärä
        kmeans.fit(X)
    # 3. Tulostetaan klusterien labelit
    cluster_centers = kmeans.cluster_centers_

    # 3. Järjestä klusterit y-koordinaatin mukaan
    sorted_clusters = np.argsort(-cluster_centers[:, 1])  # Järjestä y-koordinaatin (korkeus) mukaan (laskeva järjestys)

    # 4. Muodosta uusi label-taulukko järjestyksen mukaan
    new_labels = np.zeros_like(kmeans.labels_)
    for new_label, old_label in enumerate(sorted_clusters):
        new_labels[kmeans.labels_ == old_label] = new_label
    print("KLUSTEROINTI", time.time() - ryhjatend)

    tuotteet_hyllyissa = {}
    tuotteet_keys = list(tuotteet.keys())
    for i in range(len(new_labels)):
        tuotteet[tuotteet_keys[i]]["tuotenimi"] = tuotteet_keys[i]

        if "muut_tuotteet" in tuotteet[tuotteet_keys[i]]["tuotenimi"]:

            tuotteet[tuotteet_keys[i]]["kpl_raw"] = round(
                tuotteet[tuotteet_keys[i]]["width"] / koko_hyllyn_ka_leveys)
            tuotteet[tuotteet_keys[i]]["kpl"] = round(
                tuotteet[tuotteet_keys[i]]["width"] / koko_hyllyn_ka_leveys)
            tuotteet[tuotteet_keys[i]]["prob"] = 0
            tuotteet[tuotteet_keys[i]]["koko_hyllyn_ka_leveys"] = round(koko_hyllyn_ka_leveys)
            tuotteet[tuotteet_keys[i]]["tuotekerrokset"] = 1
            tuotteet[tuotteet_keys[i]]["EAN"]=None
        else:
            tuotteet[tuotteet_keys[i]]["tuotekerrokset"] = round(
                tuotteet[tuotteet_keys[i]]["height"] / tuotteet[tuotteet_keys[i]]["ka_korkeus"])
            tuotteet[tuotteet_keys[i]]["kpl_raw"] = tuotteet[tuotteet_keys[i]]["kpl"]
            tuotteet[tuotteet_keys[i]]["kpl"] = round(
                tuotteet[tuotteet_keys[i]]["width"] / tuotteet[tuotteet_keys[i]]["ka_leveys"]) * \
                                                tuotteet[tuotteet_keys[i]]["tuotekerrokset"]

            tuotteet[tuotteet_keys[i]]["koko_hyllyn_ka_leveys"] = round(koko_hyllyn_ka_leveys)

        if len(valit) == 1 and 2 < 1:
            # JOSA KUVASSA ON YKSI JAKOVÄLI
            left = tuotteet[tuotteet_keys[i]]["left"]
            width = tuotteet[tuotteet_keys[i]]["width"]
            if valit[0] / image_width > 0.5:
                # JOS TURHA OSA KUVAA ON KUVAN OIKEASSA LAIDASSA

                if left > valit[0]:
                    continue
            else:
                # JOS TURHA OSA ON KUVAN VASEMMASSA LAIDASSA
                if left + width < valit[0]:
                    continue
        if len(valit) == 2 and 2 < 1:
            # JOS KUVASSA ON KAKSI JAKOVÄLIÄ, oletetaan, että kahden jakovälin ulkopuoliset tuotteet on turhia
            left = tuotteet[tuotteet_keys[i]]["left"]
            width = tuotteet[tuotteet_keys[i]]["width"]
            if left < valit[0] or left > valit[1]:
                continue

        if str(new_labels[i]) in tuotteet_hyllyissa.keys():
            tuotteet_hyllyissa[str(new_labels[i])].append(tuotteet[tuotteet_keys[i]])
        else:
            tuotteet_hyllyissa[str(new_labels[i])] = [tuotteet[tuotteet_keys[i]]]

    for hylly in tuotteet_hyllyissa.keys():
        tuotteet_hyllyissa[hylly] = sorted(tuotteet_hyllyissa[hylly], key=lambda x: x['left'])
    print("---------------")
    tuotteet_hyllyissa = {k: tuotteet_hyllyissa[k] for k in sorted(tuotteet_hyllyissa)}
    # for hylly in tuotteet_hyllyissa.keys():
    #    print(tuotteet_hyllyissa[hylly])
    print("---------------")

    colors = ["green", "white", "blue", "yellow", "black", "orange", "blue", "pink", "brown", "gray", "purple", "red"]
    count = 0
    # fig, axs = plt.subplots(1, 1, figsize=(10, 5))
    # print(zero_sum_columns)

    print("START")
    to_delete = []
    for hylly in tuotteet_hyllyissa.keys():
        for tuote1 in tuotteet_hyllyissa[hylly]:
            if "EAN" not in tuote1:
                print(tuote1)
                print(1/0)
            for tuote2 in tuotteet_hyllyissa[hylly]:
                if tuote1 == tuote2:
                    continue
                if tuote2 in to_delete or tuote1 in to_delete:
                    continue
                if "auto" in tuote1["tuotenimi"] or "Auto" in tuote1["tuotenimi"]:
                    continue
                if tuote1["tuotenimi"].split(" ")[0] == tuote2["tuotenimi"].split(" ")[0]:
                    tuote1_top = tuote1["top"]
                    tuote1_left = tuote1["left"]
                    tuote1_height = tuote1["height"]
                    tuote1_width = tuote1["width"]

                    tuote2_top = tuote2["top"]
                    tuote2_left = tuote2["left"]
                    tuote2_height = tuote2["height"]
                    tuote2_width = tuote2["width"]

                    if (tuote2_left + tuote2_width / 2) < tuote1_left + tuote1_width and (
                            tuote2_left + tuote2_width / 2) > tuote1_left:

                        if tuote2_top + tuote2_height / 2 < tuote1_top:

                            height = max(tuote1_top + tuote1_height, tuote2_top + tuote2_height) - min(tuote1_top,
                                                                                                       tuote2_top)
                            # width=max(tuote1_left+tuote1_width,tuote2_left+tuote2_width)-min(tuote1_left,tuote2_left)
                            width = tuote1_width
                            tuote1["left"] = tuote1_left
                            tuote1["top"] = min(tuote1_top, tuote2_top)
                            tuote1["width"] = width
                            tuote1["height"] = height
                            # tuote1["kpl"]+=tuote2["kpl"]
                            try:
                                tuote1["kpl"] = round(tuote1["width"] / tuote1["ka_leveys"]) * round(
                                    tuote1["height"] / tuote1["ka_korkeus"])
                            except:
                                print(tuote1)
                                print(1 / 0)
                            tuote1["tuotekerrokset"] += 1
                            tuote1["kpl_raw"] += tuote2["kpl_raw"]

                            to_delete.append(tuote2["tuotenimi"])
    if plot:
        for hylly in tuotteet_hyllyissa.keys():
            for value in tuotteet_hyllyissa[hylly]:
                # Luo suorakulmio annetuilla arvoilla
                rect = patches.Rectangle((value['left'], value['top']), value['width'], value['height'],
                                         linewidth=1, edgecolor='r', facecolor=colors[count])
                axs[2].add_patch(rect)
            count += 1
        axs[2].imshow(image)
        plt.show()
    for hylly in tuotteet_hyllyissa.keys():
        original_list = tuotteet_hyllyissa[hylly].copy()
        for tuote in original_list:
            if tuote["tuotenimi"] in to_delete:# or "Auto" in tuote["tuotenimi"]:
                #print("POISTETAAN HH_DEV", tuote["tuotenimi"])
                tuotteet_hyllyissa[hylly].remove(tuote)
                # del tuote

    return tuotteet_hyllyissa, tuotteet

