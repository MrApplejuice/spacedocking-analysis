#!/usr/bin/python

import sys
import os
from math import sqrt
from PIL import Image, ImageFont, ImageDraw

print \
"""Texturize Font  Copyright (C) 2012  European Space Agency
This program comes with ABSOLUTELY NO WARRANTY; This is free software, 
and you are welcome to redistribute it under certain conditions;
For details please refer to LINCENSE.txt supplied with this software.
""";

if len(sys.argv) < 3:
  print """
Please specify texture dimensions and true type font

Usage:
  textureize.py square-dimensions font-name 
  """;
  sys.exit(1);
  
try:
  dimensions = int(sys.argv[1]);
except ValueError:
  print sys.argv[1], "is not integer";
  sys.exit(1);
  
if not os.path.isfile(sys.argv[2]):
  print sys.argv[2], "is not a file";
  sys.exit(1);

letters="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,. !?-_+*/\\%$\"()=|'"; # 81 elements
rect_size = int(sqrt(len(letters)));
if rect_size * rect_size != len(letters):
  raise Exception("Assert failed: rect_size * rect_size != len(letters)");

try:
  font = ImageFont.truetype(sys.argv[2], dimensions / rect_size);
except IOError as e:
  print "Error while loading font:", e;
  sys.exit(1);

img = Image.new("RGBA", (dimensions, dimensions), (255, 255, 255, 0));

# Draw letter raster
draw = ImageDraw.Draw(img);
widths = [0] * len(letters);
for y in range(rect_size):
  for x in range(rect_size):
    draw.text((x * dimensions / rect_size, y * dimensions / rect_size), letters[x + y * rect_size], font=font, fill="#ffffff");
    w,h = font.getsize(letters[x + y * rect_size]);
    widths[x + y * rect_size] = w;
del draw

img.save('out.png', 'PNG');
f = open('out.desc', 'w');
for w in zip(list(letters), map(lambda x: str(x), widths)):
  f.write("\t".join(w) + "\n");
f.close();
