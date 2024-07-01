import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 全局變量，用於存儲參考點和選擇狀態
ref_points = []
selecting = True

def select_points(event, x, y, flags, param):
    """
    回調函數，用於在圖像上選擇兩個參考點。
    """
    global ref_points, selecting
    
    if event == cv2.EVENT_LBUTTONDOWN:
        ref_points.append((x, y))
        
        if len(ref_points) == 2:
            selecting = False

def is_donut(contour):
    """
    檢查輪廓是否為完整的甜甜圈。
    """
    area = cv2.contourArea(contour)
    (x, y), radius = cv2.minEnclosingCircle(contour)
    circle_area = np.pi * radius * radius
    circularity = area / circle_area
    
    return circularity > 0.6

def get_px2length(image_path):
    """
    根據圖像上的參考點計算像素與實際長度之間的比例。
    """
    global ref_points, selecting
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    clone = image.copy()
    cv2.namedWindow("Select Reference Points")
    cv2.setMouseCallback("Select Reference Points", select_points, clone)
    
    while selecting:
        cv2.imshow("Select Reference Points", clone)
        cv2.waitKey(1)
    
    cv2.destroyAllWindows()
    
    if len(ref_points) == 2:
        pixel_distance = np.sqrt((ref_points[0][0] - ref_points[1][0]) ** 2 + (ref_points[0][1] - ref_points[1][1]) ** 2)
        print(f"Pixel distance: {pixel_distance:.2f} pixels")
        
        while True:
            try:
                reference_length = float(input("Enter the actual reference length in units: "))
                break
            except ValueError:
                print("Invalid input. Please enter a valid number.")
    return reference_length / pixel_distance

def segment_and_display_donuts(image_path, save_path, lower=127, upper=255, dummy_px=10):
    """
    分割圖像中的甜甜圈並顯示分割結果。
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return
    
    clone = image.copy()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, lower, upper, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    os.makedirs(save_path, exist_ok=True)
    
    valid_donuts_count = 0
    for i, contour in enumerate(contours):
        if is_donut(contour):
            valid_donuts_count += 1
            x, y, w, h = cv2.boundingRect(contour)
            donut_diameter_pixels = max(w, h)
            
            ymin = y - dummy_px
            xmin = x - dummy_px
            new_h = h + 2 * dummy_px
            new_w = w + 2 * dummy_px
            ymax = ymin + new_h
            xmax = xmin + new_w
            subimage = clone[ymin:ymax, xmin:xmax]
            
            save_file = os.path.join(save_path, f"subimage_{valid_donuts_count}.png")
            cv2.imwrite(save_file, subimage)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    
    directory, filename = os.path.split(image_path)
    new_filename =  'Segmemted_'+ filename
    new_image_path = os.path.join(directory, new_filename)
    cv2.imwrite(new_image_path, image)

    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Segmented Donuts")
    plt.axis('off')
    plt.show()
    
    print(f"Saved {valid_donuts_count} valid subimages to {save_path}")

def get_donut_radius(image_path, lower=140, upper=255, pix2length=0.1):
    """
    獲取甜甜圈的內外圓半徑並標註在圖像上。
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, binary_image = cv2.threshold(image, lower, upper, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    (x_outer, y_outer), radius_outer = cv2.minEnclosingCircle(largest_contour)
    center_outer = (int(x_outer), int(y_outer))
    radius_outer = int(radius_outer)
    
    row_values = binary_image[center_outer[1], :].astype(float)
    changes = np.diff(row_values)
    nonzero_indices = np.nonzero(changes)
    radius_inner = (nonzero_indices[0][2] - nonzero_indices[0][1]) / 2
    
    cv2.circle(image, center_outer, radius_outer, (0, 255, 0), 2)
    cv2.circle(image, center_outer, 2, (255, 0, 0), 3)
    if radius_inner > 0:
        cv2.circle(image, center_outer, int(radius_inner), (0, 0, 255), 2)
    
    text_outer = f'D1={2*radius_outer * pix2length:.2f}'
    text_inner = f'D2={radius_inner * pix2length:.2f}'
    directory, filename = os.path.split(image_path)
    new_directory = directory + '_labeled'
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    
    new_filename = text_outer + ', ' + text_inner + ', ' + filename
    new_image_path = os.path.join(new_directory, new_filename)
    cv2.imwrite(new_image_path, image)
    
    outer_diameter = int(2 * radius_outer)
    inner_diameter = int(2 * radius_inner)
    print(f"Outer Circle: Center = {center_outer}, Radius = {radius_outer}")
    print(f"Inner Circle: Center = {center_outer}, Radius = {radius_inner}")
    print(f"Outer Diameter: {outer_diameter} pixels")
    print(f"Inner Diameter: {inner_diameter} pixels")
    
    return center_outer, outer_diameter, inner_diameter
