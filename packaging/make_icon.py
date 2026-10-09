"""Draw the app icon (a rounded square with a rising line) and build an .icns."""
import os
import subprocess
import sys
from PIL import Image, ImageDraw

OUT = sys.argv[1]
S = 1024
img = Image.new('RGBA', (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)
d.rounded_rectangle((60, 60, S - 60, S - 60), radius=210, fill=(22, 26, 33, 255))
d.rounded_rectangle((60, 60, S - 60, S - 60), radius=210, outline=(58, 65, 76, 255), width=8)
pts = [(230, 660), (400, 500), (540, 590), (800, 300)]
d.line(pts, fill=(106, 163, 255, 255), width=64, joint='curve')
for x, y in pts:
    d.ellipse((x - 40, y - 40, x + 40, y + 40), fill=(106, 163, 255, 255))
for x in (330, 480, 640, 790):                                  # three result bars
    pass
d.rectangle((230, 760, 800, 790), fill=(58, 65, 76, 255))
png = os.path.join(OUT, 'icon_1024.png')
img.save(png)
iconset = os.path.join(OUT, 'icon.iconset')
os.makedirs(iconset, exist_ok=True)
for s in (16, 32, 64, 128, 256, 512, 1024):
    img.resize((s, s), Image.LANCZOS).save(os.path.join(iconset, f'icon_{s}x{s}.png'))
for s in (16, 32, 128, 256, 512):
    img.resize((s * 2, s * 2), Image.LANCZOS).save(os.path.join(iconset, f'icon_{s}x{s}@2x.png'))
subprocess.check_call(['iconutil', '-c', 'icns', iconset, '-o', os.path.join(OUT, 'icon.icns')])
print('icon ok')
