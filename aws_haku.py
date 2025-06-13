import boto3
from PIL import Image, ExifTags
from io import BytesIO
import numpy as np
import cv2
import json
import logging
import os

from urllib.parse import urlparse
from pymongo import MongoClient
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import psutil

load_dotenv()
ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
SECRET_KEY = os.getenv("AWS_SECRET_KEY")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME_prod")

def get_image(IMAGE_KEYs,panoramaside):
    print(IMAGE_KEYs)
    logger = logging.getLogger(__name__)

    kuvien_maara=len(IMAGE_KEYs)
    logger.info("HAETAAN KUVIA KANNASTA")
    s3 = boto3.client(
        's3',
        aws_access_key_id=ACCESS_KEY,
        aws_secret_access_key=SECRET_KEY
    )

    def extract_key_from_url(url):
        parsed_url = urlparse(url)
        return parsed_url.path.lstrip('/')

    def check_key_exists(bucket_name, key):
        try:
            s3.head_object(Bucket=bucket_name, Key=key)
            return True
        except s3.exceptions.ClientError:
            return False

    def flatten_list(nested_list):
        flat_list = []
        for item in nested_list:
            if isinstance(item, list):
                flat_list.extend(flatten_list(item))
            else:
                flat_list.append(item)
        return flat_list

    IMAGE_KEYs = flatten_list(IMAGE_KEYs)
    if not isinstance(IMAGE_KEYs, list) or not all(isinstance(url, str) for url in IMAGE_KEYs):
        print(f"Invalid IMAGE_KEYs: Expected list of strings but got {type(IMAGE_KEYs)} with elements {IMAGE_KEYs}")
        return None

    image_keys = [extract_key_from_url(url) for url in IMAGE_KEYs]

    images = []
    for image_key in image_keys:
        if not check_key_exists(BUCKET_NAME, image_key):
            print(f'Virhe: Tiedostoa "{image_key}" ei löytynyt.')
            return None
        try:
            image_data = download_and_compress_image(s3, BUCKET_NAME, image_key)
            image_array = np.frombuffer(image_data, np.uint8)
            del image_data
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print("alkuperäinen koko numpy taulu",image.nbytes)
            images.append(image)
        except Exception as e:
            print(f'Virhe: {e}')
            return None
    logger.info("KUVAT HAETTU KANNASTA")
    print("KUVAT HAETTU KANNASTA")
    mongo_uri = os.getenv("MONGO_URI")

    print(panoramaside)
    if panoramaside=="right":
        images.reverse()

    if kuvien_maara > 1:
        if False:
            pass

        else:
            cropped_images = []
            for i, image in enumerate(images):

                if i == 0:

                    cropped_images.append(image)



                else:

                    height, width = image.shape[:2]


                    crop_start = int(0 * width)

                    cropped_image = image[:, crop_start:]

                    cropped_images.append(cropped_image)


            panorama=cv2.hconcat(cropped_images)


            #plt.imshow(panorama)
            #plt.show()

    else:
        panorama = images[0]

    base_dir = '/'.join(image_keys[0].split('/')[:-1])
    base_filename = image_keys[0].split('/')[-1].split('_')[0]
    panorama_key = f"{base_dir}/{base_filename}_panorama.jpg"

    if kuvien_maara != 1:
        save_panorama_to_s3(s3, BUCKET_NAME, panorama, panorama_key)
        image_key=panorama_key
    else:
        save_panorama_to_s3(s3, BUCKET_NAME, panorama, panorama_key)

        image_key = panorama_key
    logger.info("PILKOTAAN JA KOMPRESSOIDAAN")
    print("PILKOTAAN JA KOMPRESSOIDAAN")


    if kuvien_maara!=1:
        del panorama
        image_contents,kuvat= compress_manually_splitted(cropped_images)
    elif kuvien_maara==1:
        image_contents, kuvat = compress_manually_splitted([panorama])
    elif kuvien_maara==1:
        image_contents, kuvat = pilko_panorama(panorama)
    logger.info("PILKOTTU JA KOMPRESSOITU")
    print("PILKOTTU JA KOMPRESSOITU")
    return image_contents, kuvat, image_key, panoramaside

def download_and_compress_image(s3, bucket_name, image_key):
    print(image_key)
    response = s3.get_object(Bucket=bucket_name, Key=image_key)
    image_data = response['Body'].read()
    del response
    original_size = len(image_data)
    print("ALKUPERÄINEN KOKO", original_size)


    return image_data

def compress_image_to_target_size(img, target_size):
    lower_bound = 5
    upper_bound = 95
    print("ALOITETAAN KAANTÖ")
    buffer = BytesIO()
    img.save(buffer, format="JPEG", quality=100)
    new_size = buffer.tell()
    print("KÄÄNTÖ KAANTÖ")

    logger = logging.getLogger(__name__)
    process = psutil.Process()
    memory_usage = process.memory_info().rss
    memory_usage_mb = memory_usage / (1024 * 1024)
    logger.info(f'Muistin käyttö kompressoinnin alussa: {memory_usage_mb:.2f} MB')
    print(f'Muistin käyttö kompressoinnin alussa: {memory_usage_mb:.2f} MB')
    print("yksittäisen kuvan koko", new_size)
    if new_size < target_size:
        logger.info("KUVA VALMIIKSI ALLE TARPEEKSI PIENI")
        buffer.seek(0)
        return buffer
    logger.info("KUVAA PIENENNETÄÄN")
    while lower_bound <= upper_bound or new_size > target_size:
        mid_quality = (lower_bound + upper_bound) // 2
        # temp_buffer = BytesIO()
        # img.save(temp_buffer, 'JPEG', quality=mid_quality)
        # new_size = temp_buffer.tell()
        buffer.seek(0)  # Siirretään puskurin alkuun uudelleenkäyttöä varten
        buffer.truncate(0)  # Tyhjennetään puskurin sisältö
        img.save(buffer, format='JPEG', quality=mid_quality)

        new_size = buffer.tell()

        #print("kompressointi kaynnyssä", new_size)
        if new_size < target_size * 0.99:
            lower_bound = mid_quality + 1
            #print("new_size < target_size * 0.95")
        elif new_size > target_size:
            upper_bound = mid_quality - 1
            #print("upper_bound = mid_quality - 1")
        else:
            # temp_buffer.seek(0)
            # return temp_buffer
            #print("else")
            buffer.seek(0)

            return buffer

    # final_buffer = BytesIO()
    # img.save(final_buffer, 'JPEG', quality=upper_bound)
    # final_buffer.seek(0)
    buffer.seek(0)
    # return final_buffer
    return buffer

def correct_orientation(img):
    try:
        exif = img._getexif()
        if exif is not None:
            for tag, value in exif.items():
                decoded = ExifTags.TAGS.get(tag, tag)
                if decoded == "Orientation":
                    if value == 1:
                        break
                    elif value == 2:
                        img = img.transpose(Image.FLIP_LEFT_RIGHT)
                    elif value == 3:
                        img = img.transpose(Image.ROTATE_180)
                    elif value == 4:
                        img = img.transpose(Image.FLIP_TOP_BOTTOM)
                    elif value == 5:
                        img = img.transpose(Image.ROTATE_270).transpose(Image.FLIP_LEFT_RIGHT)
                    elif value == 6:
                        img = img.transpose(Image.ROTATE_270)
                    elif value == 7:
                        img = img.transpose(Image.ROTATE_90).transpose(Image.FLIP_LEFT_RIGHT)
                    elif value == 8:
                        img = img.transpose(Image.ROTATE_90)
                    break
    except AttributeError:
        pass
    return img

def pilko_panorama(panorama):
    ositus = 1
    height, width = panorama.shape[:2]
    kuvat_ei_sopivia = True
    while kuvat_ei_sopivia:
        palan_leveys = width // ositus
        if palan_leveys > 4040:
            ositus += 1
            continue
        else:
            kuvat_ei_sopivia = False

    alku = 0
    loppu = 0
    image_contents = []
    images = []
    for i in range(ositus):
        loppu += palan_leveys
        pala = panorama[:, alku:loppu]
        img = Image.fromarray(pala)
        #buffer = BytesIO()
        #img.save(buffer, format="PNG")
        #buffer.seek(0)
        #img = Image.open(buffer)
        image_data = compress_image_to_target_size(img, 4194304)
        del img
        image_content = BytesIO(image_data.read())
        image_contents.append(image_content)
        images.append(pala)
        alku = loppu
    del pala
    del image_data
    return image_contents, images


def compress_manually_splitted(images):
    compressed_images=[]
    image_contents=[]
    logger = logging.getLogger(__name__)
    process = psutil.Process()
    memory_usage = process.memory_info().rss
    memory_usage_mb = memory_usage / (1024 * 1024)
    logger.info(f'Muistin käyttö manuaalisesti splitattujen kuvien kompressointia aloittaessa: {memory_usage_mb:.2f} MB')
    print(f'Muistin käyttö manuaalisesti splitattujen kuvien kompressointia aloittaessa: {memory_usage_mb:.2f} MB')
    for i in images:
        img = Image.fromarray(i)
        #buffer = BytesIO()
        #img.save(buffer, format="PNG")
        #buffer.seek(0)
        #img = Image.open(buffer)
        image_data = compress_image_to_target_size(img, 4194304)
        del img
        image_content = BytesIO(image_data.read())
        image_contents.append(image_content)
        compressed_images.append(i)
    memory_usage = process.memory_info().rss
    memory_usage_mb = memory_usage / (1024 * 1024)
    logger.info(f'Muistin käyttö kompressoinnin jalkeen: {memory_usage_mb:.2f} MB')
    print(f'Muistin käyttö kompressoinnin jalkeen: {memory_usage_mb:.2f} MB')
    return image_contents,compressed_images


def save_panorama_to_s3(s3, bucket_name, image, key):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, buffer = cv2.imencode(".JPEG", image, [cv2.IMWRITE_JPEG_QUALITY, 100])
    image_content = BytesIO(buffer.tobytes())
    image_content.seek(0)

    try:
        #s3.put_object(Bucket=bucket_name, Key=key, Body=image_content, ContentType='image/jpeg')
        print(f'Successfully uploaded {key} to {bucket_name}')
    except Exception as e:
        print(f'Error uploading image to S3: {e}')

