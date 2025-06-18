# https://medialab.github.io/iwanthue/
# random list of 100 colors
# hue between 0 and 360
# c between 17.72 and 100
# luminosity between 30.77 and 95.28
# Avoids use of 255 rgb value

# fmt: off
import json

import pandas as pd

VERSION = 1.2
NOTES = {
    "1":
    "USDA symbols are used as species index key.",
    "2":
    "When difference between USDA and EPPO codes, scientific_names, or authority occured, USDA information was used.",
    "3":
    "Purple nutsedge (CYRO) is used by may also be yellow nutsedge (CYES).",
    "4":
    "North Carolina (MD), Texas (TX), Maryland (MD)",
    "5":
    "Brassicas class contains 4 species - Brassica napus, Brassica rapa, Brassica juncea, Brassica hirta"
}

with open("species_info.json", "r") as outfile:
    data = json.load(outfile)
spec = data["species"]
class_ids = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
    40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
    59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77,
    78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
    97, 98, 99, 100
]

usda_symbols = [
    'background', 'AMPA', 'AMAR2', 'SEOB4', 'XAST', 'DISA', 'ELIN3', 'URPL2',
    'CYRO', 'AMTU', 'ECCR', 'ECCO2', 'URTE2', 'BASC5', 'HEAN3', 'PAHY', 'SOHA',
    'GLMA4', 'AMHY', 'CHAL7', 'PADI', 'DAST', 'ABTH', 'SEPU8', 'SEFA',
    'ERCA20', 'ZEA', 'plant', 'colorchecker', 'VIVI', 'PISA6', 'TRIN3',
    'TRPR2', 'BRASS2', 'RASA2', 'SECE', 'TRITI2', 'TRAE', 'AVSA', 'HORDE',
    'AVST2', 'GOHI', 'URRE2', "SISP", "IPLA", "AMSP", "TRPO2"
]

hex = [
    "#000000", "#1d686e", "#e452f1", "#60e54e", "#903be2", "#9ee33f",
    "#5d4ce7", "#dbe93c", "#bd36d6", "#3fac38", "#912bba", "#57eb93",
    "#e342c6", "#8fe175", "#5a42c1", "#e3ca3b", "#6364e2", "#a0b837",
    "#b360e8", "#6ba33c", "#ab2a9f", "#cfea7c", "#8e42b0", "#40be7b",
    "#e636a0", "#5aedc3", "#e83c2b", "#55ece3", "#e73b58", "#9ee3a0",
    "#617ff7", "#eea626", "#3967c5", "#e86928", "#5e97e6", "#ba3a1c",
    "#5dcfe9", "#ea427e", "#44975e", "#e172d8", "#407018", "#bd7ae4",
    "#aa952b", "#7350af", "#d6a846", "#968de9", "#d88633", "#34649f",
    "#e7d37e", # pineland threeseed mercury
    "#9d4999", # yellow nutsedge
    "#7b852d", # great ragweed
    "#b5317f", # silverleaf nightshade
    "#6fc196", # field bindweed
    "#b73361", # Canada thistle
    "#50bbaf", # redroot amaranth
    "#bb353f", # ram's horn
    "#87d9dd", # Balansa clover
    "#9e3442", # Berseem clover
    "#a6e2cc", # white clover
    "#e067af", # turnip
    "#436f30", # alfalfa
    "#e2a1e5", # buckwheat
    "#6a6115", # camelina
    "#a376c2", # browntop millet
    "#aebd70", # venice mallow
    "#675395", # common pokeweed
    "#d9e2ad", # bermudagrass
    "#8f3d76", # ivyleaf morning glory
    "#8ea975", # wild oat
    "#e57fa7", "#436e42", "#e87159",
    "#44a7d9", "#a15921", "#9ac6ed", "#a5482f", "#3e98a4", "#e07179",
    "#306b51", "#b16fa0", "#8f6a21", "#8080c0", "#b8965b", "#4481ab",
    "#db8e6b", "#336b85", "#dab28e", "#586188", "#8c895d", "#c3b5e9",
    "#605e32", "#e6adc2", "#33675e", "#9c4d5f", "#58937e", "#7d5371",
    "#836542", "#7a9cc2", "#945848", "#9e85ac", "#bf828a"
]

rgb = [[0, 0, 0], [29, 104, 110], [228, 82, 241], [96, 229, 78],
       [144, 59, 226], [158, 227, 63], [93, 76, 231], [219, 233, 60],
       [189, 54, 214], [63, 172, 56], [145, 43, 186], [87, 235, 147],
       [227, 66, 198], [143, 225, 117], [90, 66, 193], [227, 202, 59],
       [99, 100, 226], [160, 184, 55], [179, 96, 232], [107, 163, 60],
       [171, 42, 159], [207, 234, 124], [142, 66, 176], [64, 190, 123],
       [230, 54, 160], [90, 237, 195], [232, 60, 43], [85, 236, 227],
       [231, 59, 88], [158, 227, 160], [97, 127, 247], [238, 166, 38],
       [57, 103, 197], [232, 105, 40], [94, 151, 230], [186, 58, 28],
       [93, 207, 233], [234, 66, 126], [68, 151, 94], [225, 114, 216],
       [64, 112, 24], [189, 122, 228], [170, 149, 43], [115, 80, 175],
       [214, 168, 70], [150, 141, 233], [216, 134, 51], [52, 100, 159],
       [231, 211, 126], # pineland threeseed mercury
       [157, 73, 153], # yellow nutsedge
       [123, 133, 45], # great ragweed
       [181, 49, 127], # silverleaf nightshade
       [111, 193, 150], # field bindweed
       [183, 51, 97], # Canada thistle
       [80, 187, 175], # redroot amaranth
       [187, 53, 63], # ram's horn
       [135, 217, 221], # Balansa clover
       [158, 52, 66], # Berseem clover
       [166, 226, 204], # white clover 
       [224, 103, 175], # turnip
       [67, 111, 48], # alfalfa
       [226, 161, 229], # buckwheat
       [106, 97, 21],  # camelina
       [163, 118, 194], # browntop millet
       [174, 189, 112], # venice mallow
       [103, 83, 149], # common pokeweed
       [217, 226, 173], # bermudagrass
       [143, 61, 118], # ivyleaf morning glory
       [142, 169, 117], # wild oat
       [229, 127, 167], [67, 110, 66], [232, 113, 89],
       [68, 167, 217], [161, 89, 33], [154, 198, 237], [165, 72, 47],
       [62, 152, 164], [224, 113, 121], [48, 107, 81], [177, 111, 160],
       [143, 106, 33], [128, 128, 192], [184, 150, 91], [68, 129, 171],
       [219, 142, 107], [51, 107, 133], [218, 178, 142], [88, 97, 136],
       [140, 137, 93], [195, 181, 233], [96, 94, 50], [230, 173, 194],
       [51, 103, 94], [156, 77, 95], [88, 147, 126], [125, 83, 113],
       [131, 101, 66], [122, 156, 194], [148, 88, 72], [158, 133, 172],
       [191, 130, 138]]
