import numpy as np
import cv2 as cv
from dataclasses import dataclass
from typing import Tuple, List, Optional

default_config = {
    "focal_lenght_mm": 45, 
    "camera_distance": 10,
    "lim_min_xy": np.array([200, 200]),
    "lim_max_xy": np.array([200, 200]),
    "shape_px": np.array([255, 255]),
    "point": {
        "to_draw": True,
        "radius": 2,
        "color": (255, 255, 255), 
        "thickness": 1,
        "time_mapping": {
            "color_indicies": [1, 2],
            "use": True
        },
    },
    "axis": {
        "colors": [(255, 50, 50),(50, 255, 50),(50, 50, 255)],
        "to_draw": True,
        "thickness": 2
    },
    "trace": {
        "to_draw": False,
        "color": (70, 70, 70),
        "thickness": 2,
    },
    "bg_rgb": (50, 50, 50)
}

def lerp(v, fmin, fmax, tmin, tmax):
    return (v - fmin) / (fmax - fmin) * (tmax - tmin) + tmin

def rotate(v, vec_rad):
    roll, pitch, yaw = vec_rad
    rollmat = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, 1],
    ])

    pitchmat = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])

    yawmat = np.array([
        [1, 0, 0],
        [0, np.cos(yaw), -np.sin(yaw)],
        [0, np.sin(yaw), np.cos(yaw)],
    ])

    return v @ yawmat @ rollmat @ pitchmat


def render(canvas: Optional, points: np.ndarray, rotation_rad, cfg):
    """ 
        rotation_rad: roll pitch yaw 
        points: [[x, y, z], . . .]
    """
    if canvas is None:
        canvas = np.ones((*cfg['shape_px'].tolist(), 3), dtype=np.uint8) * np.array(cfg["bg_rgb"], dtype=np.uint8)

    rpoints = rotate(points, rotation_rad)

    projected = (cfg['focal_lenght_mm'] * rpoints[:, :2]) / (rpoints[:, 2] - cfg['camera_distance']).reshape(-1, 1)

    projected_mean = np.mean(projected, axis=0)
    
    pixels = lerp(projected, projected_mean-cfg['lim_min_xy'], projected_mean+cfg['lim_max_xy'], [0, 0], cfg['shape_px'])

    if cfg['axis']['to_draw']:
        axis_points = np.array([
            [0, 0, 0], # origin,
            [0, 0, 1], # z
            [0, 1, 0], # y
            [1, 0, 0], # x
        ]) * 3
        axis_points = rotate(axis_points, rotation_rad)
        proj_axis_points = (cfg['focal_lenght_mm'] * axis_points[:, :2]) / (axis_points[:, 2] - cfg['camera_distance']).reshape(-1, 1)
        axis_pixels = lerp(proj_axis_points, projected_mean-cfg['lim_min_xy'], projected_mean+cfg['lim_max_xy'], [0, 0], cfg['shape_px'])

        axis_pixels = axis_pixels.astype(int)
        origin = axis_pixels[0]
        try:
            for axis_pixel, color in zip(axis_pixels[1:4], cfg['axis']['colors']):
                print(axis_pixel)
                cv.line(canvas, origin, axis_pixel, color, cfg['axis']['thickness'])
        except: pass
    
    prev_pixel = None
    for i, pixel in enumerate(pixels):
        c =  (i+1) / len(pixels) * 255
        pixel = tuple(pixel.astype(int))
        base_color = list(cfg['point']['color'])
        if cfg['point']['time_mapping']['use']:
            for idx in cfg['point']['time_mapping']['color_indicies']:
                base_color[idx] = c

        try: 
            if cfg['trace']['to_draw'] and prev_pixel is not None:
                cv.line(canvas, prev_pixel, pixel, cfg['trace']['color'], cfg['trace']['thickness'])

            if cfg['point']['to_draw']:
                cv.circle(canvas, pixel, cfg['point']['radius'],base_color, cfg['point']['thickness'])
            prev_pixel = pixel
        except:pass 
    return canvas


if __name__ == "__main__":
    cfg = default_config
    roll, pitch, yaw = 0.0, 0.0, 0.0
    camera_distance = 10.0
    focal_lenght = 33.3

    def grid_pattern(n, y, spacing=1):
        points = []
        
        for i in range(n):
            for j in range(n):
                points.append([i * spacing, y, j * spacing])
        
        for i in range(n):
            for j in range(n-1):
                for k in range(1, 4):
                    t = k/4
                    x = i * spacing
                    z = (j + t) * spacing
                    points.append([x, y, z])
        
        for i in range(n-1):
            for j in range(n):
                for k in range(1, 4):
                    t = k/4
                    x = (i + t) * spacing
                    z = j * spacing
                    points.append([x, y, z])
        
        points = np.array(points)
        center = (n * spacing) / 2
        points[:, 0] -= center
        points[:, 2] -= center
        
        return points

    def generate_sphere_points(radius=1, num_points=100):
        num_phi = int(np.sqrt(num_points))  
        num_theta = num_phi * 2
        phi = np.linspace(0, np.pi, num_phi)
        theta = np.linspace(0, 2*np.pi, num_theta)
        phi, theta = np.meshgrid(phi, theta)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.sin(phi) * np.sin(theta)
        z = radius * np.cos(phi)
        points = np.column_stack((x.flatten(), y.flatten(), z.flatten()))
        return points


    points = list(generate_sphere_points(1, 1000))

    center = np.mean(points)
    def rollslider_callback(val):
        global roll
        roll = np.radians(val/10)
        print(f"roll+ rad: {roll}")

    def pitchslider_callback(val):
        global pitch
        pitch = np.radians(val/10)
        print(f"pitch rad: {pitch}")

    def yawslider_callback(val):
        global yaw
        yaw = np.radians(val/10)
        print(f"yaw rad: {yaw}")

    def cameradistance_callback(val):
        global camera_distance
        camera_distance = val/20

    def focallenght_callback(val):
        global focal_lenght
        focal_lenght = val/100

    def capture_movement(event, x, y, flags, param):
        global mouse_pos
        mouse_pos = np.array([x, y], dtype=int)

    cv.namedWindow("plot", cv.WINDOW_GUI_NORMAL)
    cv.setWindowProperty("plot", cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
    cv.setMouseCallback("plot", capture_movement)

    cv.createTrackbar(
        "roll", "plot", 0, 10*180, rollslider_callback
    ) 
    cv.createTrackbar(
        "pitch", "plot", 0, 10*180, pitchslider_callback
    ) 
    cv.createTrackbar(
        "yaw", "plot", 0, 10*180, yawslider_callback
    ) 
    cv.createTrackbar(
        "camera", "plot", int(camera_distance*10), 200, cameradistance_callback
    )
    cv.createTrackbar(
        "focal_length", "plot", int(focal_lenght*10), 300000, focallenght_callback
    )

    import time
    while True:     
        rotation_rad = roll, pitch, yaw
        cfg['camera_distance'] = camera_distance
        cfg['focal_lenght_mm'] = focal_lenght
        tstart = time.time()
        cv.imshow("plot", render(None, points, rotation_rad, cfg))
        tend = time.time()
        past_ms = (tend - tstart)*1000
        print(f"{past_ms:3.2f}ms. / render at {len(points)} points")
        cv.waitKey(1)
