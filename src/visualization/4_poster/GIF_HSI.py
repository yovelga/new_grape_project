import spectral
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# ——— פרמטרים ———
hdr_path = r"/all_raw_data\1_14\25.09.24\HS\results\REFLECTANCE_2024-09-25_015.hdr"
gif_output = os.path.join(os.path.dirname(hdr_path), "HSI_sequence.gif")
frame_duration = 0.2  # שניות בין מסגרות
loop_forever = 0  # 0 = אינסוף, 1 = חזרה אחת בלבד

# ——— קריאת ה-HSI וטעינת אורכי גל ———
img = spectral.open_image(hdr_path)
cube = img.load().astype(np.float32)  # (rows, cols, bands)
wavelengths = [float(w) for w in img.metadata["wavelength"]]

# ——— הכנת גופן ———
try:
    font = ImageFont.truetype("arial.ttf", size=20)
except IOError:
    font = ImageFont.load_default()

frames = []
for b, wl in enumerate(wavelengths):
    # 1. בחר band ונרמל ל־0–255
    band = cube[:, :, b]
    mn, mx = band.min(), band.max()
    norm = ((band - mn) / (mx - mn) * 255).astype(np.uint8)

    # 2. סיבוב 90° עם כיוון השעון
    rotated = np.rot90(norm, k=-1)

    # 3. המרת ה-array ל־PIL Image, ואז ל־RGB כדי לאפשר צבע מלא
    im = Image.fromarray(rotated).convert("RGB")
    draw = ImageDraw.Draw(im)
    text = f"{wl:.1f} nm"

    # 4. חשב מימדי טקסט באמצעות textbbox
    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]

    # # 5. רקע שחור מלבן מאחורי הטקסט
    padding = 0
    x, y = 420, 250
    # draw.rectangle(
    #     (x, y, x + text_w + padding, y + text_h + padding),
    #     fill=(0, 0, 0)  # שחור
    # )

    # 6. כתיבת הטקסט – כאן בצבע לבן, או כחול אם תבחר
    draw.text((x + padding // 2, y + padding // 2), text, font=font, fill=(0, 0, 0))
    # להחליף ל־כחול: fill=(0, 0, 255)

    frames.append(im)

# ——— שמירת ה-GIF המונפש ———
imageio.mimsave(
    gif_output, frames, format="GIF", duration=frame_duration, loop=loop_forever
)

print(f"Animated GIF saved to: {gif_output}")
