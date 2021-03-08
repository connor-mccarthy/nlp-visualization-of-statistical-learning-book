import io
import os
from typing import List

import pdf2image
import PIL
import pytesseract
import requests

from caching import cache


@cache
def get_pdf(url: str) -> bytes:
    r = requests.get(url)
    if r.status_code == 200:
        return r.content
    else:
        raise Exception("Request failed.")


def convert_pdf_to_png(bytes_: bytes) -> bytes:
    list_of_pil_images = pdf2image.convert_from_bytes(bytes_)
    io_buffer = save_list_of_pil_images(list_of_pil_images, "png")

    return io_buffer.getvalue()


def save_list_of_pil_images(list_of_pil_images: List, output_mime: str) -> io.BytesIO:
    img_byte_arr = io.BytesIO()
    if isinstance(list_of_pil_images, list):
        image_1 = list_of_pil_images[0]
        other_images = list_of_pil_images[1:]
        image_1.save(
            img_byte_arr, save_all=True, append_images=other_images, format=output_mime
        )
    else:
        list_of_pil_images.save(img_byte_arr, format=output_mime)
    return img_byte_arr


def convert_image_to_string(png_bytes: bytes) -> str:
    replacements = {"\n": " ", "\x0c": "", "  ": " "}

    pages = {}
    image = PIL.Image.open(io.BytesIO(png_bytes))
    for page in range(image.n_frames):
        text = pytesseract.image_to_string(image).replace("\n", " ")
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = text.strip()
        image.seek(page)
        pages[page] = text
    page_list = list(pages.values())
    return " ".join(page_list)


def write_text_to_txt(text: str, file: str) -> None:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_directory, file)
    with open(filepath, "w+") as f:
        f.write(text)
    print(f"Text written to {filepath}.")


def read_text_from_txt(file: str) -> str:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(current_directory, file)
    with open(filepath, "r+") as f:
        text = f.read()
    return text


def get_text_from_url(url: str) -> str:
    pdf_bytes = get_pdf(url)
    png = convert_pdf_to_png(pdf_bytes)
    return convert_image_to_string(png)
