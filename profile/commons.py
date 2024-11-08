import os
from pdf2image import convert_from_bytes
from PIL.Image import Image


def load_pdf_bytes() -> bytes:
    dir_path = os.path.dirname(__file__)
    path = os.path.join(dir_path, "resources", "sample_pdf.pdf")
    with open(path, "rb") as fh:
        return fh.read()


def pdf_to_image(pdf_bytes: bytes, dpi: int) -> Image:
    images = convert_from_bytes(pdf_bytes, dpi=dpi, grayscale=False)
    return images[0]
