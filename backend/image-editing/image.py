import sys
from pathlib import Path

# repo src directory is on sys.path
# PROJECT_SRC = Path(__file__).resolve().parents[3]
# # if str(PROJECT_SRC) not in sys.path:
# #     sys.path.insert(0, str(PROJECT_SRC))

from PIL import Image, ImageChops

class ImageEditor:

    def __init__(self, image_1: Path, image_2: Path):
        self.image_1 = Image.open(image_1).convert("RGBA")
        self.image_2 = Image.open(image_2).convert("RGBA")
        if self.image_2.size != self.image_1.size:
            self.image_2 = self.image_2.resize(image_1.size)

    # 128 is 50% transparency. 0 is fully transparent, 255 is opaque
    def transparency(self, alpha: int = 192, output_name: str = "overlay.png"):
        copy_2 = self.image_2
        copy_2.putalpha(alpha)

        copy_1 = self.image_1

        overlay = Image.alpha_composite(copy_1, copy_2)

        script_dir = Path(__file__).resolve().parent
        output_path = script_dir / output_name

        overlay.save(output_path)

        print(f"Saved overlay to: {output_path}")

if __name__ == "__main__":
    # i1 = "image_name"
    # i2 = "image2_name"
    # imgEditor = ImageEditor(i2, i1)
    # imgEditor.diff()
    # imgEditor.transparency()
    print("Complete!")