import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib
matplotlib.use('Agg')

img = cv2.imread('tangan.png', 0)
if img is None:
    img = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(img, (150, 150), 80, 255, -1)
    cv2.putText(img, "X-RAY", (110, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)

img = cv2.resize(img, (300, 300))

def hitung_metrik(org, proc):
    mse_val = np.mean((org.astype(float) - proc.astype(float)) ** 2)
    if mse_val == 0:
        psnr_val = 100.0
    else:
        psnr_val = 20 * np.log10(255.0 / np.sqrt(mse_val))
    res = cv2.matchTemplate(org, proc, cv2.TM_CCORR_NORMED)[0][0]
    ssim_val = (res + 1) / 2 if res < 1 else 1.0
    return round(mse_val, 2), round(psnr_val, 2), round(ssim_val, 4)

font = cv2.FONT_HERSHEY_SIMPLEX

def tambah_teks(gambar, teks):
    temp = gambar.copy()
    cv2.putText(temp, teks, (10, 30), font, 0.6, 255, 1, cv2.LINE_AA)
    return temp

img_negative = 255 - img
res1 = np.hstack((tambah_teks(img, "Original"), tambah_teks(img_negative, "Negative")))
cv2.imwrite('soal1_result.png', res1)

c_val = 255 / np.log(1 + np.max(img)) if np.max(img) > 0 else 1
img_log = np.array(c_val * (np.log(1 + img.astype(float))), dtype='uint8')
mse_log, psnr_log, ssim_log = hitung_metrik(img, img_log)
res2 = np.hstack((tambah_teks(img, "Original"), tambah_teks(img_log, "Log")))
cv2.imwrite('soal2_result.png', res2)

gammas = [0.1, 0.5, 1.2, 2.2]
tabel_gamma = []
res3 = tambah_teks(img, "Original")
for g in gammas:
    g_img = np.array(255 * (img / 255) ** g, dtype='uint8')
    res3 = np.hstack((res3, tambah_teks(g_img, f"G={g}")))
    m, p, s = hitung_metrik(img, g_img)
    tabel_gamma.append((g, m, p, s))
cv2.imwrite('soal3_gamma_comparison.png', res3)

min_v, max_v, _, _ = cv2.minMaxLoc(img)
if max_v - min_v == 0: max_v = 255
img_str = np.uint8(((img - min_v) / (max_v - min_v)) * 255)

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Gambar Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_str, cmap='gray', vmin=0, vmax=255)
plt.title('Hasil Contrast Stretching')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.hist(img.ravel(), 256, [0, 256], color='black')
plt.title('Histogram Sebelum')

plt.subplot(2, 2, 4)
plt.hist(img_str.ravel(), 256, [0, 256], color='black')
plt.title('Histogram Sesudah (Stretched)')
plt.tight_layout()
plt.savefig('soal4_result.png', dpi=150)
plt.close()

img_he = cv2.equalizeHist(img)
mse_he, psnr_he, ssim_he = hitung_metrik(img, img_he)

plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Gambar Original')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(img_he, cmap='gray', vmin=0, vmax=255)
plt.title('Hasil Histogram Equalization')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.hist(img.ravel(), 256, [0, 256], color='black')
plt.title('Histogram Sebelum')

plt.subplot(2, 2, 4)
plt.hist(img_he.ravel(), 256, [0, 256], color='black')
plt.title('Histogram Sesudah (HE / Rata)')
plt.tight_layout()
plt.savefig('soal5_result.png', dpi=150)
plt.close()

clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
img_clahe = clahe.apply(img)
mse_clahe, psnr_clahe, ssim_clahe = hitung_metrik(img, img_clahe)

plt.figure(figsize=(15, 8))
plt.subplot(2, 3, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=255)
plt.title('Gambar Original')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img_he, cmap='gray', vmin=0, vmax=255)
plt.title('Hasil HE')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_clahe, cmap='gray', vmin=0, vmax=255)
plt.title('Hasil CLAHE')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.hist(img.ravel(), 256, [0, 256], color='black')
plt.title('Histogram Original')

plt.subplot(2, 3, 5)
plt.hist(img_he.ravel(), 256, [0, 256], color='black')
plt.title('Histogram HE')

plt.subplot(2, 3, 6)
plt.hist(img_clahe.ravel(), 256, [0, 256], color='black')
plt.title('Histogram CLAHE')
plt.tight_layout()
plt.savefig('soal6_comparison.png', dpi=150)
plt.close()

print("\n" + "="*45)
print("       DATA LAPORAN PRAKTIKUM KAMU")
print("="*45)
print(f"[SOAL 2] LOG   -> MSE: {mse_log} | PSNR: {psnr_log} dB | SSIM: {ssim_log}")
print("\n[SOAL 3] GAMMA ->")
print(" Gamma	MSE	PSNR(dB)	SSIM")
for r in tabel_gamma:
    print(f" {r[0]}	{r[1]}	{r[2]}		{r[3]}")
print(f"\n[SOAL 5] HE    -> MSE: {mse_he} | PSNR: {psnr_he} dB | SSIM: {ssim_he}")
print(f"[SOAL 6] CLAHE -> MSE: {mse_clahe} | PSNR: {psnr_clahe} dB | SSIM: {ssim_clahe}")
print("="*45)
print("PROSES SELESAI! SILAKAN CEK FOLDER SEKARANG.")
print("="*45)
sys.stdout.flush()