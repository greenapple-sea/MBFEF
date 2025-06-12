 
from PIL import Image, ImageDraw, ImageFont

txt_img = Image.new("RGBA", (320, 240), (255,255,255,0))
d = ImageDraw.Draw(txt_img)
text = "abcabc"
font_mono= "/home/guocunhan/.config/Ultralytics/Arial.ttf"
font_color_green = (0,255,0,255)
font = ImageFont.truetype(font_mono, 28)
txt_width, _ = d.textsize(text, font=font)
print(txt_width)