from colors import hex, rgb

cultivar_hex = [
    "#03039e", # peanut - Bailey II
    "#05f217", # peanut - TifNV-HG
    "#2605f2", # peanut - Georgia-06G
    "#05f2b6", # peanut - Georgia-12Y
    "#05f267", # peanut - Emery
    "#039e05", # peanut - NC20
    "#f105f2", # peanut - EXP-OLEIC-001
    "#4ff205",
    "#0549f2",
    "#5c039e",
    "#0590f2",
    "#13d3f2",
    "#6d05f2",
    "#9e0305",
    "#c0d104",
    "#039e87",
    "#04b84e",
    "#7bb804",
    "#f20574",
    "#8df205",
    "#d104ba",
    "#033f9e",
    "#f2f205",
    "#d10425",
    "#047eb8",
    "#b705f2",
    "#0fb8b6",
    "#9e036a",
    "#32b804",
    "#2c28b8",
    "#f21a05",
    "#36f235",
    "#0417d1",
    "#11d18a",
    "#8d039e",
    "#35f271",
    "#b87c04",
    "#4904d1",
    "#3635f2",
    "#3570f2",
    "#d10453",
    "#2ed1cd",
    "#569e0d",
    "#79d12e",
    "#2ed154",
    "#f2c605",
    "#d11190",
    "#f24f05",
    "#9e033f",
    "#2e039e",
]

cultivar_rgb = [
    [3, 3, 158], # peanut - Bailey II
    [5, 242, 23], # peanut - TifNV-HG
    [38, 5, 242], # peanut - Georgia-06G
    [5, 242, 182], # peanut - Georgia-12Y
    [5, 242, 103], # peanut - Emery
    [3, 158, 5], # peanut - NC20
    [241, 5, 242], # peanut - EXP-OLEIC-001
    [79, 242, 5],
    [5, 73, 242],
    [92, 3, 158],
    [5, 144, 242],
    [19, 211, 242],
    [109, 5, 242],
    [158, 3, 5],
    [192, 209, 4],
    [3, 158, 135],
    [4, 184, 78],
    [123, 184, 4],
    [242, 5, 116],
    [141, 242, 5],
    [209, 4, 186],
    [3, 63, 158],
    [242, 242, 5],
    [209, 4, 37],
    [4, 126, 184],
    [183, 5, 242],
    [15, 184, 182],
    [158, 3, 106],
    [50, 184, 4],
    [44, 40, 184],
    [242, 26, 5],
    [54, 242, 53],
    [4, 23, 209],
    [17, 209, 138],
    [141, 3, 158],
    [53, 242, 113],
    [184, 124, 4],
    [73, 4, 209],
    [54, 53, 242],
    [53, 112, 242],
    [209, 4, 83],
    [46, 209, 205],
    [86, 158, 13],
    [121, 209, 46],
    [46, 209, 84],
    [242, 198, 5],
    [209, 17, 144],
    [242, 79, 5],
    [158, 3, 63],
    [46, 3, 158],
]

assert not set(cultivar_hex).intersection(set(hex))

old_rgb = {tuple(x) for x in rgb}
new_rgb_check = {tuple(x) for x in cultivar_rgb}

assert not old_rgb.intersection(new_rgb_check)
assert len(cultivar_hex) == len(set(cultivar_hex)) == 50
assert len(new_rgb_check) == 50