# cec2017.functions
# Author: Duncan Tilley
# Combines simple, hybrid and composition functions (f1 - f30) into a single
# module

from .simple import *
from .hybrid import *
from .composition import *

all_functions = [
    None,
    f1,  f2,  f3,  f4,  f5,  f6,  f7,  f8,  f9,  f10,
    f11, f12, f13, f14, f15, f16, f17, f18, f19, f20,
    f21, f22, f23, f24, f25, f26, f27, f28, f29, f30
]

best_fitness = {1:100, 2:200, 3:300, 4:400, 5:500, 6:600, 7:700, 8:800, 9:900, 10:1000,
                11:1100, 12:2200, 13:3300, 14:4400, 15:5500, 16:6600, 17:7700, 18:8800, 19:9900, 20:10000,
                21:2100, 22:2200, 23:3300, 24:4400, 25:5500, 26:6600, 27:7700, 28:8800, 29:9900, 30:10000}