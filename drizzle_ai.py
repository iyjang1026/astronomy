import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
import glob
import os
from numba import njit
import sys
import numpy as np
from numba import njit

@njit
def calculate_overlap_area(rect1, rect2):
    x_overlap = max(0.0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0]))
    y_overlap = max(0.0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1]))
    return x_overlap * y_overlap

@njit
def rotate_point(x, y, theta_deg, cx, cy):
    theta = np.radians(theta_deg)
    x_rel = x - cx
    y_rel = y - cy
    x_rot = x_rel * np.cos(theta) - y_rel * np.sin(theta) + cx
    y_rot = x_rel * np.sin(theta) + y_rel * np.cos(theta) + cy
    return x_rot, y_rot

def calculate_output_bounds(w_in, h_in, scale, rotation_deg, center):
    # 입력 이미지 네 꼭짓점 좌표
    corners = np.array([[0, 0], [w_in, 0], [0, h_in], [w_in, h_in]], dtype=np.float64)
    cx, cy = center
    theta = np.radians(rotation_deg)
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta),  np.cos(theta)]])
    transformed = []
    for x, y in corners:
        x_rel, y_rel = x - cx, y - cy
        xy_rot = rot_matrix @ np.array([x_rel, y_rel])
        x_rot, y_rot = xy_rot[0] + cx, xy_rot[1] + cy
        # 출력좌표: 회전 후 확대
        x_out = x_rot * scale
        y_out = y_rot * scale
        transformed.append((x_out, y_out))
    transformed = np.array(transformed)
    x_min, y_min = np.floor(transformed.min(axis=0))
    x_max, y_max = np.ceil(transformed.max(axis=0))
    # 필요한 출력 shape 계산
    w_out = int(x_max - x_min)
    h_out = int(y_max - y_min)
    offset = (x_min, y_min)
    return h_out, w_out, offset

@njit
def drizzle(input_image, output_shape, scale, pixfrac, rotation_deg, center_x, center_y, offset_x, offset_y, output_image, weight_map):
    h_in, w_in = input_image.shape
    h_out, w_out = output_shape
    pixel_size_out = 1.0 / scale
    drop_size = pixfrac * pixel_size_out

    for y_in in range(h_in):
        for x_in in range(w_in):
            val = input_image[y_in, x_in]
            x_center_in = x_in + 0.5
            y_center_in = y_in + 0.5
            x_rot, y_rot = rotate_point(x_center_in, y_center_in, rotation_deg, center_x, center_y)
            x_out = x_rot * scale - offset_x
            y_out = y_rot * scale - offset_y

            drop_rect = (x_out - drop_size/2, y_out - drop_size/2,
                         x_out + drop_size/2, y_out + drop_size/2)

            x_start = int(np.floor(drop_rect[0]))
            x_end = int(np.ceil(drop_rect[2]))
            y_start = int(np.floor(drop_rect[1]))
            y_end = int(np.ceil(drop_rect[3]))

            for y_out_idx in range(y_start, y_end):
                if y_out_idx < 0 or y_out_idx >= h_out:
                    continue
                for x_out_idx in range(x_start, x_end):
                    if x_out_idx < 0 or x_out_idx >= w_out:
                        continue
                    pixel_rect = (x_out_idx, y_out_idx, x_out_idx + 1, y_out_idx + 1)
                    overlap_area = calculate_overlap_area(drop_rect, pixel_rect)
                    if overlap_area > 0.0:
                        output_image[y_out_idx, x_out_idx] += val * overlap_area
                        weight_map[y_out_idx, x_out_idx] += overlap_area

@njit
def normalize_output(output_image, weight_map):
    h, w = output_image.shape
    normalized_image = np.zeros_like(output_image)
    normalized_image[:] = np.nan
    for y in range(h):
        for x in range(w):
            if weight_map[y, x] > 0:
                normalized_image[y, x] = output_image[y, x] / weight_map[y, x]
    return normalized_image
"""
# 실행 예시
if __name__ == "__main__":
    # (입력 이미지: 10x10, 중심부에 1.0 사각형 패치)
    input_img = np.zeros((10, 10), dtype=np.float64)
    input_img[4:6, 4:6] = 1.0

    scale = 2
    pixfrac = 0.7
    rotation_deg = 30.0
    h_in, w_in = input_img.shape
    center = (w_in / 2, h_in / 2)

    # ---- (1) 출력 크기/offset 자동 계산 ----
    h_out, w_out, (offset_x, offset_y) = calculate_output_bounds(w_in, h_in, scale, rotation_deg, center)

    # ---- (2) 배열 생성 ----
    output_img = np.zeros((h_out, w_out), dtype=np.float64)
    weight_map = np.zeros((h_out, w_out), dtype=np.float64)

    # ---- (3) 드리즐 실행 ----
    drizzle(input_img, (h_out, w_out), scale, pixfrac, rotation_deg, center[0], center[1], offset_x, offset_y, output_img, weight_map)
    normalized_img = normalize_output(output_img, weight_map)

    print("Output shape after safe bounds:", normalized_img.shape)
    print(normalized_img)
"""

# 실행 예시 (numba 함수는 미리 배열 생성, 매개변수 분리 필요)
# 예시 사용법
if __name__ == "__main__":
    #input_img[4:6, 4:6] = 1.0  # 밝기 있는 작은 사각 영역
    path = '/volumes/ssd/intern/25_summer/M101_L/sky_subed'
    obj_name = 'M101'
    file = sorted(glob.glob(path+'/pp*.fits'))
    if not os.path.exists(path+'/drz'):
        os.mkdir(path+'/drz')
    import random
    # 드리즐 적용 (2배 확대, 70% 픽스프랙, 30도 회전)
    for i in range(len(file)):
        n = format(i, '04')
        input_img = fits.open(file[i])[0].data #np.zeros((10, 10))
        hdr = fits.open(file[i])[0].header
        theta = random.uniform(0,30)
        scale = 1
        rotation_deg = theta
        
        #output_shape = (input_img.shape[0]*scale, input_img.shape[0]*scale) #
        pixfrac = 0.7
        h_in, w_in = input_img.shape
        center = (w_in / 2, h_in / 2)

        # ---- (1) 출력 크기/offset 자동 계산 ----
        h_out, w_out, (offset_x, offset_y) = calculate_output_bounds(w_in, h_in, scale, rotation_deg, center)

        # ---- (2) 배열 생성 ----
        output_img = np.zeros((h_out, w_out), dtype=np.float32)
        weight_map = np.zeros((h_out, w_out), dtype=np.float32)

        # ---- (3) 드리즐 실행 ----
        drizzle(input_img.astype(np.float32), (h_out, w_out), scale, pixfrac, rotation_deg, center[0], center[1], offset_x, offset_y, output_img, weight_map)
        normalized_img = normalize_output(output_img, weight_map)
        
        fits.writeto(path+'/drz/drz_'+obj_name+str(n)+'.fits', normalized_img.astype(np.float32),header=hdr, overwrite=True)
        print(f'drizzled {n}')
        #mean, median, std = sigma_clipped_stats(normalized_img, cenfunc='median', stdfunc='mad_std', sigma=3.0)
    #print("Output image shape:", output_img.shape)
        #plt.imshow(normalized_img, origin='lower', vmax=median+3*std, vmin=median-3*std)
        #plt.show(); sys.exit()
    #print(output_img)
