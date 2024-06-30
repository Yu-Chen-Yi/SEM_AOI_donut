import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_donut_radius(image_path = './mnt/data/image3.png', lower = 127, upper = 255):
    # 函數的輸入：
    # image_path: 圖像的路徑 (默認值為 './mnt/data/image3.png')
    # lower: 二值化圖像的閾值下限 (默認值為 127)
    # upper: 二值化圖像的閾值上限 (默認值為 255)
    
    # 函數的輸出：
    # center_outer: 外圓的圓心 (座標形式，如 (x, y))
    # radius_outer: 外圓的半徑 (像素單位)
    # radius_inner: 內圓的半徑 (像素單位)
        
    # 讀取圖像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 二值化圖像
    _, binary_image = cv2.threshold(image, lower, upper, cv2.THRESH_BINARY)
    # 找到輪廓
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 使用subplot顯示圖像
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(binary_image, cmap='gray')
    plt.title("Binary Image")

    # 找到最大的輪廓，即甜甜圈的外輪廓
    largest_contour = max(contours, key=cv2.contourArea)

    # 獲取外接圓
    (x_outer, y_outer), radius_outer = cv2.minEnclosingCircle(largest_contour)

    # 圓心取整
    center_outer = (int(x_outer), int(y_outer))
    radius_outer = int(radius_outer)

    # 通過圓心的水平線上的灰階值變化
    row_values = binary_image[center_outer[1], :].astype(float)
    changes = np.diff(row_values)
    nonzero_indices = np.nonzero(changes)
    # 假設內圓的灰階值變化最明顯，因此內圓的邊界應該是變化最大的位置
    radius_inner = (nonzero_indices[0][2] - nonzero_indices[0][1])/2
    # 顯示結果
    result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.circle(result_image, center_outer, radius_outer, (0, 255, 0), 2)
    cv2.circle(result_image, center_outer, 2, (255, 0, 0), 3) # 圓心

    # 畫出內圓
    if radius_inner > 0:
        cv2.circle(result_image, center_outer, int(radius_inner), (0, 0, 255), 2)
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Donut")
    plt.show()

    # 輸出結果
    print(f"Outer Circle: Center = {center_outer}, Radius = {radius_outer}")
    print(f"Inner Circle: Center = {center_outer}, Radius = {radius_inner}")
    print(f"Outer Diameter: {int(2 * radius_outer)} pixels")
    print(f"Inner Diameter: {int(2 * radius_inner)} pixels")
    return center_outer, radius_outer, radius_inner

test_image_path = './mnt/data/image3.png'
center_outer, radius_outer, radius_inner = get_donut_radius(image_path = test_image_path, lower = 127, upper = 255)
