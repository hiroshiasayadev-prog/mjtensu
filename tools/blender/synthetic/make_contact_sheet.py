from pathlib import Path
import json
from PIL import Image, ImageDraw

root = Path(__import__("sys").argv[1])
count = int(__import__("sys").argv[2])
thumbs = []
for i in range(count):
    im = Image.open(root / "images" / f"synthetic_{i:06d}.png").convert("RGB")
    rec = json.loads((root / "records" / f"synthetic_{i:06d}.json").read_text())
    draw = ImageDraw.Draw(im)
    for a in rec["annotations"]:
        x, y, w, h = a["bbox"]
        draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=1)
    draw.text((2, 2), str(i), fill=(255, 255, 0))
    im.thumbnail((256, 256))
    thumbs.append(im.copy())
cols = 5
rows = (len(thumbs) + cols - 1) // cols
sheet = Image.new("RGB", (cols * 256, rows * 256), (32, 32, 32))
for j, im in enumerate(thumbs):
    sheet.paste(im, ((j % cols) * 256, (j // cols) * 256))
out = root / f"contact_{count:04d}_bbox.png"
sheet.save(out)
print(out)
