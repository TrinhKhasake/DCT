import numpy as np
from scipy.fftpack import dct, idct
import cv2 as cv
import matplotlib.pyplot as plt
import random
import math
from PIL import Image

# File names
IMAGE_PATH = "image1.jpg"
WATERMARK_PATH = "watermark.jpg"
WATERMARKED_IMAGE_PATH = "Watermarked_Image.jpg"
EXTRACTED_WATERMARK_PATH = "watermarked_extracted.jpg"

# Constants
KEY = 50
BLOCK_SIZE = 8
WATERMARK_WIDTH = 64
WATERMARK_HEIGHT = 64
FREQUENCY_FACTOR = 8
INDEX_X = 0
INDEX_Y = 0
BLOCK_CUT = 50

# Helper lists
original_values = []
extracted_values = []

def calculate_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def calculate_ncc(image1, image2):
    return abs(np.mean(np.multiply((image1 - np.mean(image1)), (image2 - np.mean(image2)))) / (np.std(image1) * np.std(image2)))

def dct_transform(matrix):
    return cv.dct(matrix)

def idct_transform(matrix):
    return cv.idct(matrix)

def embed_watermark(image, watermark):
    rows, cols = np.size(image, 0), np.size(image, 1)
    image_rows, image_cols = rows, cols
    rows -= BLOCK_CUT * 2
    cols -= BLOCK_CUT * 2
    watermark_rows, watermark_cols = np.size(watermark, 0), np.size(watermark, 1)

    print(rows, cols, watermark_rows, watermark_cols)

    if (rows * cols // (BLOCK_SIZE * BLOCK_SIZE)) < watermark_rows * watermark_cols:
        print("Watermark too large.")
        return image

    selected_blocks = set()
    total_blocks = (rows // BLOCK_SIZE) * (cols // BLOCK_SIZE)
    print("Available blocks", total_blocks)
    required_blocks = watermark_rows * watermark_cols

    i, j = 0, 0
    image_float = np.float32(image)
    while i < image_rows:
        while j < image_cols:
            dct_block = cv.dct(image_float[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE] / 1.0)
            image_float[i:i + BLOCK_SIZE, j:j + BLOCK_SIZE] = cv.idct(dct_block)
            j += BLOCK_SIZE
        j = 0
        i += BLOCK_SIZE

    final_image = image
    random.seed(KEY)
    i = 0
    print("Blocks needed", required_blocks)
    counter = 0
    while i < required_blocks:
        watermark_pixel = watermark[i // watermark_cols][i % watermark_cols]
        bit_value = 0
        if watermark_pixel >= 127:
            watermark_pixel = 1
            bit_value = 255
        else:
            watermark_pixel = 0

        watermark[i // watermark_cols][i % watermark_cols] = bit_value
        block_index = random.randint(1, total_blocks)
        if block_index in selected_blocks:
            continue
        selected_blocks.add(block_index)
        rows_blocks = rows // BLOCK_SIZE
        cols_blocks = cols // BLOCK_SIZE
        ind_row = (block_index // cols_blocks) * BLOCK_SIZE + BLOCK_CUT
        ind_col = (block_index % cols_blocks) * BLOCK_SIZE + BLOCK_CUT

        dct_block = cv.dct(image_float[ind_row:ind_row + BLOCK_SIZE, ind_col:ind_col + BLOCK_SIZE] / 1.0)
        coeff = dct_block[INDEX_X][INDEX_Y]
        coeff /= FREQUENCY_FACTOR
        rounded_coeff = coeff
        if watermark_pixel % 2 == 1:
            if math.ceil(coeff) % 2 == 1:
                coeff = math.ceil(coeff)
            else:
                coeff = math.ceil(coeff) - 1
        else:
            if math.ceil(coeff) % 2 == 0:
                coeff = math.ceil(coeff)
            else:
                coeff = math.ceil(coeff) - 1

        dct_block[INDEX_X][INDEX_Y] = coeff * FREQUENCY_FACTOR
        original_values.append((coeff * FREQUENCY_FACTOR, watermark_pixel))

        final_image[ind_row:ind_row + BLOCK_SIZE, ind_col:ind_col + BLOCK_SIZE] = cv.idct(dct_block)
        image_float[ind_row:ind_row + BLOCK_SIZE, ind_col:ind_col + BLOCK_SIZE] = cv.idct(dct_block)
        i += 1

    final_image = np.uint8(final_image)
    print("PSNR is:", calculate_psnr(image_float, image))
    cv.imshow("Final Image", final_image)
    cv.imwrite(WATERMARKED_IMAGE_PATH, final_image)
    return image_float

def retrieve_watermark(image, output_name):
    rows, cols = np.size(image, 0), np.size(image, 1)

    if rows != 1000 or cols != 1000:
        image = cv.resize(image, (1000, 1000))
        rows, cols = 1000, 1000
    rows -= BLOCK_CUT * 2
    cols -= BLOCK_CUT * 2
    total_blocks = (rows // BLOCK_SIZE) * (cols // BLOCK_SIZE)
    required_blocks = WATERMARK_WIDTH * WATERMARK_HEIGHT

    extracted_watermark = np.zeros((WATERMARK_WIDTH, WATERMARK_HEIGHT), dtype=np.uint8)
    selected_blocks = set()
    random.seed(KEY)
    i = 0
    while i < required_blocks:
        watermark_pixel = 0
        block_index = random.randint(1, total_blocks)
        if block_index in selected_blocks:
            continue
        selected_blocks.add(block_index)
        rows_blocks = rows // BLOCK_SIZE
        cols_blocks = cols // BLOCK_SIZE
        ind_row = (block_index // cols_blocks) * BLOCK_SIZE + BLOCK_CUT
        ind_col = (block_index % cols_blocks) * BLOCK_SIZE + BLOCK_CUT
        dct_block = cv.dct(image[ind_row:ind_row + BLOCK_SIZE, ind_col:ind_col + BLOCK_SIZE] / 1.0)

        coeff = dct_block[INDEX_X][INDEX_Y]
        coeff = math.floor(coeff + 0.5)
        coeff /= FREQUENCY_FACTOR

        if coeff % 2 == 0:
            watermark_pixel = 0
        else:
            watermark_pixel = 255
        extracted_watermark[i // WATERMARK_WIDTH][i % WATERMARK_HEIGHT] = watermark_pixel
        extracted_values.append((coeff, bool(watermark_pixel)))

        i += 1

    cv.imwrite(output_name, extracted_watermark)
    print("Watermark extracted and saved in", output_name)
    return extracted_watermark

# Geometric attacks
def scale_up(image):
    return cv.resize(image, (1100, 1100))

def scale_down(image):
    return cv.resize(image, (0, 0), fx=0.1, fy=0.1)

def cut_rows(image):
    new_image = np.zeros((900, 1000), dtype=np.float32)
    new_row, new_col = 0, 0
    for row in range(1000):
        for col in range(1000):
            if row < 400 or row >= 500:
                new_image[new_row, new_col] = image[row, col]
                new_col += 1
                if new_col == 1000:
                    new_col = 0
                    new_row += 1
    return new_image

# Signal attacks
def apply_average_filter(image):
    kernel = np.ones((5, 5), np.float32) / 25
    return cv.filter2D(image, -1, kernel)

def apply_median_filter(image):
    m, n = image.shape
    filtered_image = np.zeros([m, n])
    for i in range(1, m - 1):
        for j in range(1, n - 1):
            neighborhood = [image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                            image[i, j-1], image[i, j], image[i, j+1],
                            image[i+1, j-1], image[i+1, j], image[i+1, j+1]]
            neighborhood.sort()
            filtered_image[i, j] = neighborhood[4]
    return filtered_image.astype(np.uint8)

def add_noise(noise_type, image):
    if noise_type == "gauss":
        noise = np.random.normal(0.1, 0.01 ** 0.5, image.shape)
        return image + noise
    elif noise_type == "s&p":
        prob = 0.05
        noisy_image = np.zeros(image.shape, np.uint8)
        threshold = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                random_val = random.random()
                if random_val < prob:
                    noisy_image[i, j] = 0
                elif random_val > threshold:
                    noisy_image[i, j] = 255
                else:
                    noisy_image[i, j] = image[i, j]
        return noisy_image
    elif noise_type == "speckle":
        speckle = np.random.normal(0, 1, image.size).reshape(image.shape[0], image.shape[1]).astype('uint8')
        return image + image * speckle

if __name__ == "__main__":
    print("Main image:", IMAGE_PATH)
    print("Watermark:", WATERMARK_PATH)

    print("=================================== EMBEDDING WATERMARK ======================")
    cover_image = cv.imread(IMAGE_PATH, 0)
    watermark_image = cv.imread(WATERMARK_PATH, 0)
    watermark_image = cv.resize(watermark_image, dsize=(WATERMARK_WIDTH, WATERMARK_HEIGHT), interpolation=cv.INTER_CUBIC)
    cv.imshow('Watermark Image', watermark_image)

    watermarked_image = embed_watermark(cover_image, watermark_image)

    print("\nWatermarking Done!\n")

    print("=================================== EXTRACTING WATERMARK ======================")
    extracted_watermark = retrieve_watermark(watermarked_image, EXTRACTED_WATERMARK_PATH)
    print("NCC:", calculate_ncc(watermark_image, extracted_watermark))

    print("\n\n============================= ATTACKS ==============================\n")

    print("Checking scaling down image:")
    scaled_down_image = scale_down(watermarked_image)
    extracted_watermark = retrieve_watermark(scaled_down_image, "Extracted_GeoAtt_Half.jpg")
    print("NCC:", calculate_ncc(watermark_image, extracted_watermark))

    print("\nChecking scaling up image:")
    scaled_up_image = scale_up(watermarked_image)
    extracted_watermark = retrieve_watermark(scaled_up_image, "Extracted_GeoAtt_Bigger.jpg")
    print("NCC:", calculate_ncc(watermark_image, extracted_watermark))

    print("\nChecking when 100 rows of image are cut:")
    cut_image = cut_rows(watermarked_image)
    extracted_watermark = retrieve_watermark(cut_image, "Extracted_GeoAtt_Cut100Rows.jpg")
    print("NCC:", calculate_ncc(watermark_image, extracted_watermark))

    print("\nChecking average filter application:")
    avg_filtered_image = apply_average_filter(watermarked_image)
    extracted_watermark = retrieve_watermark(avg_filtered_image, "Extracted_SigAtt_AvgFilter.jpg")
    print("NCC:", calculate_ncc(watermark_image, extracted_watermark))

    print("\nChecking median filter application:")
    med_filtered_image = apply_median_filter(watermarked_image)
    extracted_watermark = retrieve_watermark(med_filtered_image, "Extracted_SigAtt_MedFilter.jpg")
    print("NCC:", calculate_ncc(watermark_image, extracted_watermark))

    print("\nChecking Gaussian noise addition:")
    noisy_image = add_noise("gauss", watermarked_image)
    extracted_watermark = cv.imread("Extracted_SigAtt_GaussNoise.jpg", 0)
    print("NCC:", calculate_ncc(watermark_image, extracted_watermark))

    print("\nChecking Salt & Pepper noise addition:")
    noisy_image = add_noise("s&p", watermarked_image)
    extracted_watermark = retrieve_watermark(noisy_image, "Extracted_SigAtt_s&pNoise.jpg")
    print("NCC:", calculate_ncc(watermark_image, extracted_watermark))

    print("\nChecking speckle noise addition:")
    noisy_image = add_noise("speckle", watermarked_image)
    extracted_watermark = retrieve_watermark(noisy_image, "Extracted_SigAtt_SpeckNoise.jpg")
    print("NCC:", calculate_ncc(watermark_image, extracted_watermark))
