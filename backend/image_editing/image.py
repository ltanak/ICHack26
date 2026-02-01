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
            self.image_2 = self.image_2.resize(self.image_1.size)

    # 128 is 50% transparency. 0 is fully transparent, 255 is opaque
    def transparency(self, alpha: int = 192, output_name: str = "overlay.png", output_dir: Path = None):
        copy_2 = self.image_2
        copy_2.putalpha(alpha)

        copy_1 = self.image_1

        overlay = Image.alpha_composite(copy_1, copy_2)

        if output_dir is None:
            script_dir = Path(__file__).resolve().parent
            output_path = script_dir / output_name
        else:
            output_path = output_dir / output_name

        overlay.save(output_path)

        print(f"Saved overlay to: {output_path}")

def apply_transparency(image_1_path: Path, image_2_path: Path, year: int, alpha: int = 192):
    """
    Apply transparency overlay to two images and save to display_data/overlays.
    
    Args:
        image_1_path: Path to the base image
        image_2_path: Path to the overlay image
        year: Year for the output filename
        alpha: Transparency level (0-255, 0=transparent, 255=opaque)
    """
    output_dir = Path("display_data/overlays")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = f"{year}_overlay.png"
    
    imgEditor = ImageEditor(image_1_path, image_2_path)
    imgEditor.transparency(alpha=alpha, output_name=output_name, output_dir=output_dir)

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    i1 = script_dir / "image_2.png"
    i2 = script_dir / "image_3.png"
    apply_transparency(i1, i2, year=2017)
    print("Complete!")