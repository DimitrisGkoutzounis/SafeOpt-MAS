#convert an image to pdf
from PIL import Image
import os

def image_to_pdf(image_path, pdf_path):
    img = Image.open(image_path)
    img.save(pdf_path, "PDF" ,resolution=100.0)


if __name__ == '__main__':
    image_path = "print1.png"
    pdf_path = "print1.pdf"
    image_to_pdf(image_path, pdf_path)
    print("Image converted to pdf successfully")