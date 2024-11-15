
import cv2 as cv
import numpy as np 



distance_to_camera = 0.01 # cm 
width, height = 200, 300 # px  
# rad
roll, pitch, yaw = 1.15, 1.15, 0


def lerp(v, from_min, from_max, to_min, to_max):
        return (v - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

points = [
    # [0, 0, 0],
    [1, 1, 1],
    # [0, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    
    [0, 0, 2],
    [1, 1, 2],
    [0, 1, 2],
    [1, 0, 2],
    [0, 1, 2],
    
    [0, 0, 4],
    [1, 1, 4],
    [0, 1, 4],
    [1, 0, 4],
    [0, 1, 4],
    
    [0, 0, 7],
    [1, 1, 7],
    [0, 1, 7],
    [1, 0, 7],
    [0, 1, 7],
]

def render_points(roll, pitch, yaw, points, distance):
    rollmat = np.array([
        [np.cos(roll), -np.sin(roll), 0],
        [np.sin(roll), np.cos(roll), 0],
        [0, 0, distance],
    ])

    pitchmat = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, distance, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])

    yawmat = np.array([
        [distance, 0, 0],
        [0, np.cos(yaw), -np.sin(yaw)],
        [0, np.sin(yaw), np.cos(yaw)],
    ])
    
    points = np.array(points)

    projected_points = []
    for point in points: 
        point = np.array(point) 
        point = point @ rollmat  @ pitchmat @ yawmat
        
        ratio = distance / (point[2]+1e-5) 
        proj = ratio * (point[:2])
        projected_points.append(proj)

    pixels = lerp(projected_points, np.array([0, 0]), np.array([10, 10]), np.array([0, 0]), np.array([height, width]))
    canvas = np.zeros((width, height,3 ), dtype=np.uint8)  

    for pixel, point in zip(pixels, points):
        distance = point[2]
        color = int(distance / np.max(points[:, 2]) * 255)
        cv.circle(canvas, tuple(pixel.astype(int)), 2, (color, 255, color), 2)

    return canvas


def rollslider_callback(val):
    global roll
    roll = np.radians(-val)
    print(f"roll+ rad: {roll}")


def pitchslider_callback(val):
    global pitch
    pitch = np.radians(-val)
    print(f"pitch rad: {pitch}")

def yawslider_callback(val):
    global yaw
    yaw = np.radians(-val)
    print(f"yaw rad: {yaw}")



def capture_movement(event, x, y, flags, param):
    global mouse_pos
    mouse_pos = np.array([x, y], dtype=int)

cv.namedWindow("plot", cv.WINDOW_GUI_NORMAL)
cv.setWindowProperty("plot", cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
cv.setMouseCallback("plot", capture_movement)

cv.createTrackbar(
    "roll", "plot", 0, 180, rollslider_callback
) 
cv.createTrackbar(
    "pitch", "plot", 0, 180, pitchslider_callback
) 
cv.createTrackbar(
    "yaw", "plot", 0, 180, yawslider_callback
) 

while True: 
    
    
    canvas = render_points(roll, pitch, yaw, points, 20.01)
    cv.imshow("plot", canvas)
    cv.waitKey(1)