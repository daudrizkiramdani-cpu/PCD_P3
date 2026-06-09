import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fungsi untuk menghitung distribusi probabilitas (normalisasi histogram)
def calculate_normalized_histogram(image):
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    total_pixels = image.size
    normalized_histogram = histogram / total_pixels
    return normalized_histogram

# 1. Baca citra asli (priasolo.png) dalam format Grayscale
img_original = cv2.imread('priasolo.png', cv2.IMREAD_GRAYSCALE)

# 2. Manipulasi gambar untuk membuat 4 fase pencahayaan
# GELAP: Kurangi kecerahan (pixel di-shift ke kiri)
img_gelap = cv2.subtract(img_original, 80)

# TERANG: Tambah kecerahan (pixel di-shift ke kanan)
img_terang = cv2.add(img_original, 80)

# LOW CONTRAST: Sempitkan rentang kontras (misal ke rentang 100-180)
img_low = np.uint8(img_original * 0.3 + 100)

# HIGH CONTRAST: Gunakan teknik Thresholding / Peregangan Kontras ekstrem
img_high = cv2.equalizeHist(img_original) 

# 3. Hitung normalisasi histogram untuk masing-masing fase
hist_gelap = calculate_normalized_histogram(img_gelap)
hist_terang = calculate_normalized_histogram(img_terang)
hist_low = calculate_normalized_histogram(img_low)
hist_high = calculate_normalized_histogram(img_high)

# 4. Tampilkan dalam bentuk Grid 4x2
plt.figure(figsize=(12, 16))

# --- BARIS 1: FASE GELAP ---
plt.subplot(4, 2, 1)
plt.imshow(img_gelap, cmap='gray', vmin=0, vmax=255)
plt.title('Citra Gelap (Under-exposed)')
plt.axis('off')

plt.subplot(4, 2, 2)
plt.bar(range(256), hist_gelap, color='black')
plt.title('Normalisasi Histogram - Gelap')
plt.ylabel('Probabilitas')

# --- BARIS 2: FASE TERANG ---
plt.subplot(4, 2, 3)
plt.imshow(img_terang, cmap='gray', vmin=0, vmax=255)
plt.title('Citra Terang (Over-exposed)')
plt.axis('off')

plt.subplot(4, 2, 4)
plt.bar(range(256), hist_terang, color='black')
plt.title('Normalisasi Histogram - Terang')
plt.ylabel('Probabilitas')

# --- BARIS 3: FASE LOW CONTRAST ---
plt.subplot(4, 2, 5)
plt.imshow(img_low, cmap='gray', vmin=0, vmax=255)
plt.title('Citra Low Contrast')
plt.axis('off')

plt.subplot(4, 2, 6)
plt.bar(range(256), hist_low, color='black')
plt.title('Normalisasi Histogram - Low Contrast')
plt.ylabel('Probabilitas')

# --- BARIS 4: FASE HIGH CONTRAST ---
plt.subplot(4, 2, 7)
plt.imshow(img_high, cmap='gray', vmin=0, vmax=255)
plt.title('Citra High Contrast')
plt.axis('off')

plt.subplot(4, 2, 8)
plt.bar(range(256), hist_high, color='black')
plt.title('Normalisasi Histogram - High Contrast')
plt.xlabel('Intensitas Piksel')
plt.ylabel('Probabilitas')

# Tampilkan semua plot secara rapi
plt.tight_layout()
plt.show()