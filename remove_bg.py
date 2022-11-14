from rembg import remove
from PIL import Image


def remove_bg(filename, outfilename):
    """Удаляем фон"""
    input_f = Image.open(filename)
    output = remove(input_f, alpha_matting=False)
    """Меняем фон в PNG и сохранияем в jpg"""
    fill_color = (255, 255, 255)  # your new background color

    output = output.convert("RGBA")  # it had mode P after DL it from OP
    if output.mode in ('RGBA', 'LA'):
        background = Image.new(output.mode[:-1], output.size, fill_color)
        background.paste(output, output.split()[-1])  # omit transparency
        output = background
    output.convert("RGB").save(outfilename)

remove_bg('Tu.jpg', 'out_Tu.jpg')