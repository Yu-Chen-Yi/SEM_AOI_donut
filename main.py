import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import *

# 全局變量，用於存儲參考點和選擇狀態
ref_points = []
selecting = True
test_image_path = './data/M1M4_Newfixed_oxide_8_89.png'
save_folder_path = './data/donut_subimages_CY'
pix2length = get_px2length(test_image_path)
segment_and_display_donuts(test_image_path, save_folder_path)

# 準備資料框架來儲存結果
results = []

# 設定影像資料夾路徑
image_folder = save_folder_path
# 迭代資料夾中的每一張影像
for filename in os.listdir(image_folder):
    if filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        print(f"Processing {image_path}")
        center_outer, outer_diameter, inner_diameter = get_donut_radius(image_path, lower=140, upper=255, pix2length=pix2length)
        results.append([filename, center_outer, outer_diameter * pix2length, inner_diameter * pix2length])

# 將結果寫入 Excel
df = pd.DataFrame(results, columns=['Filename', 'Center', 'Outer Diameter (micron)', 'Inner Diameter (micron)'])
output_excel = './data/donut_diameters.xlsx'
df.to_excel(output_excel, index=False)

print(f"Results have been written to {output_excel}")