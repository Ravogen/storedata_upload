import numpy as np
import json
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import numpy as np
import json
from scipy import sparse
import cv2
# Oletetaan kuvan koko
img_height = 4080
img_width = 3060
with open("tiedosto.json", "r", encoding="utf-8") as f:
    data = json.load(f)
# Tunnistusdata (esimerkkinä; käytä omaasi tähän)
detections = data[0]  # <- korvaa omalla datalistallasi
# Selvitä ainutlaatuiset tag_namet
def create_tensor(detections):
    tag_names = list({item["tag_name"] for item in detections})

    # Luo tag_name → kerrosindeksi -sanakirja
    tag_to_index = {tag: i for i, tag in enumerate(tag_names)}

    # Luo tyhjä 3D-matriisi
    tensor = np.zeros((len(tag_names), img_height, img_width), dtype=np.float32)
    def gaussian_2d(shape, center, sigma):
        """ Luo 2D normaalijakauma, jossa maksimi on keskellä. """
        h, w = shape
        y = np.arange(0, h)[:, np.newaxis]
        x = np.arange(0, w)[np.newaxis, :]
        cy, cx = center
        gauss = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma**2))
        return gauss
    # Täytetään tensor
    for item in detections:
        tag = item["tag_name"]
        bbox = item["bounding_box"]

        # Laske bounding boxin keskipiste pikseliarvoina
        cx = int((bbox["left"] + bbox["width"] / 2) * img_width)
        cy = int((bbox["top"] + bbox["height"] / 2) * img_height)

        # Kirjoitetaan arvo 100 oikea   an kohtaan oikealla kerroksella
        layer = tag_to_index[tag]
        sigma = bbox["width"] * img_width  # käytetään leveyttä varianssina

        # Luo 2D-gaussi koko kuvan kokoisena (tai halutessa pienemmässä ikkunassa)
        gauss = gaussian_2d((img_height, img_width), (cy, cx), sigma)

        # Maksimiarvo normalisoidaan esim. 100:aan
        gauss *= 1 / gauss.max()

        # Summaa kerrokseen (jos päällekkäisyyksiä)
        tensor[layer] += gauss
        #if 0 <= cx < img_width and 0 <= cy < img_height:
        #    tensor[layer, cy, cx] = 100
        #    print("HEP")
    # Esimerkiksi: näytä tagin "redbull_tropical" kerros
    print(tensor.shape)
    import matplotlib.pyplot as plt

    #layer_index = tag_to_index["Red_Bull_The_Tropical_0,25l_tlk"]
    #plt.imshow(tensor[layer_index], cmap="gray")
    #plt.title("redbull_tropical")
    #plt.show()
    return tensor, tag_names





def find_best_shift(tensor1, tags1, tensor2, tags2):
    """
    Etsii parhaan siirtymän vain niistä kerroksista (tageista), jotka löytyvät molemmista tensoreista.

    Parameters:
        tensor1, tensor2 : np.ndarray, shape (n, H, W)
            Tensors from two images
        tags1, tags2 : list of str
            tag_name-järjestys tensorikerroksille

    Returns:
        dy, dx : paras pystysiirtymä ja vaakasiirtymä
        total_corr : korrelaatiokartta
    """
    # Selvitä yhteiset tagit ja niiden indeksit molemmissa tensoreissa
    common_tags = list(set(tags1) & set(tags2))
    print(common_tags)
    if not common_tags:
        raise ValueError("Ei yhteisiä tunnistuksia tensorien välillä")

    # Haetaan indeksit molemmissa tensoreissa
    idxs1 = [tags1.index(tag) for tag in common_tags]
    idxs2 = [tags2.index(tag) for tag in common_tags]
    from scipy import sparse
    # Ota vain yhteiset kerrokset samassa järjestyksessä
    t1_sub =tensor1[idxs1]
    t2_sub = tensor2[idxs2]
    print(t1_sub.shape)
    print(t2_sub.shape)
    kerrokset, rivit, sarakkeet = t2_sub.shape
    summat=[]
    x=[]
    for i in range(0,3060,50):
        shift = i
        x.append(shift)
        new_sarakkeet = sarakkeet + shift  # 3160

        # Luodaan tulostensori, aluksi nollat
        combined = np.zeros((kerrokset, rivit, new_sarakkeet))

        # Lasketaan indeksit, joilla tulo voidaan tehdä
        max_index = sarakkeet - shift  # 3060 - 100 = 2960

        # Täytetään tulostensorin sarakkeet 0..2959 kerto-alkioilla,
        # sijoitus oikeaan kohtaan sarakkeissa shift..(shift + max_index - 1)
        #combined[:, :, shift:shift + max_index] = t1_sub[:, :, :max_index] * t2_sub[:, :, shift:sarakkeet]
        combined[:, :, shift:sarakkeet] = t1_sub[:, :, shift:sarakkeet] * t2_sub[:, :, 0:sarakkeet - shift]
        # combined[:, :, 0:shift] jää nollaksi, koska oikean tensorin indeksi olisi negatiivinen,
        # ja combined[:, :, shift + max_index:] jää nollaksi, koska oikean tensorin indeksi ylittyy

          # (12, 4080, 3160)
        total_sum = np.sum(combined)
        print(combined.shape)
        print(total_sum)
        summat.append(total_sum)

        # Suorita korrelaatio
    max_arvon_indexi=summat.index(max(summat))

    #plt.plot(x,summat)
    #plt.show()
    return x[max_arvon_indexi]
    from scipy.signal import correlate2d
    n_layers, H, W = t1_sub.shape
    total_corr = np.zeros((2 * H - 1, 2 * W - 1))

    for i in range(n_layers):
        total_corr += correlate2d(t1_sub[i], t2_sub[i], mode='full')

    # Etsi maksimi
    max_idx = np.unravel_index(np.argmax(total_corr), total_corr.shape)
    dy = max_idx[0] - (H - 1)
    dx = max_idx[1] - (W - 1)

    return dy, dx, total_corr

def align_images_by_shift(img1, img2, dy, dx, method="average"):
    """
    Leikkaa ja kohdistaa kaksi np.array-kuvaa annetun (dy, dx)-siirtymän perusteella.
    Molemmat kuvat leikataan samankokoisiksi, vain päällekkäisestä alueesta.
    """
    H, W = img1.shape[:2]

    y1_start = max(0, dy)
    y2_start = max(0, -dy)
    y_end = min(H, H + dy) if dy < 0 else min(H - dy, H)

    x1_start = max(0, dx)
    x2_start = max(0, -dx)
    x_end = min(W, W + dx) if dx < 0 else min(W - dx, W)

    # Leikkaa molemmat kuvat
    crop1 = img1[y1_start:y1_start + y_end, x1_start:x1_start + x_end]
    crop2 = img2[y2_start:y2_start + y_end, x2_start:x2_start + x_end]

    if method == "average":
        result = ((crop1.astype(np.float32) + crop2.astype(np.float32)) / 2).astype(np.uint8)
    elif method == "overlay":
        result = crop2  # palautetaan vain toinen kuva, esim. overlay-tarkoituksessa
    else:
        raise ValueError("Tuntematon yhdistystapa")

    return result, crop1, crop2


def crop_images(original_images):
    cropped_images = [original_images[0]]
    for i in range(len(data) - 1):
        img = original_images[i+1]
        img = np.array(img)

        data1 = data[i]
        data2 = data[i + 1]

        tensor1, tags1 = create_tensor(data1)
        tensor2, tags2 = create_tensor(data2)

        indeksi = find_best_shift(tensor1, tags1, tensor2, tags2)
        # print(indeksi)
        indeksi = 3060 - indeksi
        img = img[:, indeksi:]
        cropped_images.append(img)
        # kokokuva = cv2.hconcat(images)
        # plt.imshow(kokokuva)
        # plt.show()



images=[]
img = Image.open(f"img_0.jpeg")
img = np.array(img)
images.append(img)

for i in range(len(data) - 1):
    img = Image.open(f"img_{i+1}.jpeg")
    img = np.array(img)

    data1 = data[i]
    data2 = data[i + 1]

    tensor1, tags1 = create_tensor(data1)
    tensor2, tags2 = create_tensor(data2)

    indeksi = find_best_shift(tensor1, tags1, tensor2, tags2)
    # print(indeksi)
    indeksi = 3060 - indeksi
    img=img[:, indeksi:]
    images.append(img)
    #kokokuva = cv2.hconcat(images)
    #plt.imshow(kokokuva)
    #plt.show()
kokokuva=cv2.hconcat(images)
plt.imshow(kokokuva)
plt.show()
# Lataa kuva
img1 = Image.open("img_0.jpeg")
img1 = np.array(img1)

img2 = Image.open("img_1.jpeg")
img2 = np.array(img2)
#
#
tensor1, tags1 = create_tensor(data[0])
tensor2, tags2 = create_tensor(data[1])

indeksi=find_best_shift(tensor1, tags1, tensor2, tags2)
#print(indeksi)
indeksi=3060-indeksi
#indeksi=1670
#indeksi= align_images_by_shift(img1, img2, dy, dx)
plt.imshow(img2[:, indeksi:])
plt.show()



kokokuva=cv2.hconcat([img1,img2[:, indeksi:]])

plt.imshow(kokokuva)
plt.show()

