
import cv2 as cv
import numpy as np 



width, height = 200, 300 # px  
camera_distance = 10
focal_lenght = 30 # mm 
# rad
roll, pitch, yaw = 1.15, 1.15, 0


def lerp(v, from_min, from_max, to_min, to_max):
        return (v - from_min) / (from_max - from_min) * (to_max - to_min) + to_min

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

points = np.array(list(grid_pattern(20, 0))+list(generate_sphere_points(1, 1000)))


def render_points(roll, pitch, yaw, points, distance):
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
    
    points = np.array(points)


    z_axis = []
    projected_points = []
    max_z = float('-inf')  # Initialize with negative infinity
    min_z = float('inf')   # Initialize with positive infinity
    max_x = float('-inf')
    min_x = float('inf')
    max_y = float('-inf')
    min_y = float('inf')
    center_x = 0
    center_y = 0
    center_z = 0
    for point in points: 
        point = np.array(point) 
        # rpoint = point @ yawmat @ pitchmat @ rollmat
        rpoint = point @ yawmat @ rollmat @ pitchmat
        
        z = rpoint[2]
 
        if abs(z) < 1e-5:
            z = 1e-5
 
        max_z = max(max_z, z)
        min_z = min(min_z, z)
        z_axis.append(z)

        x = (focal_lenght * (rpoint[0])) / (z - distance) 
        y = (focal_lenght * (rpoint[1])) / (z - distance) 

        center_x += rpoint[0]
        center_y += rpoint[1]
        center_z += rpoint[2]

        max_x = max(max_x, x)
        min_x = min(min_x, x)
        max_y = max(max_y, y)
        min_y = min(min_y, y)

        projected_points.append([x, y])
        
    num_points = len(points)
    center_x /= num_points
    center_y /= num_points
    center_z /= num_points
    np.array(projected_points)        

    pixels = lerp(projected_points, np.array([-30, -30]), np.array([30, 30]), np.array([0, 0]), np.array([height, width]))
    canvas = np.zeros((width, height,3 ), dtype=np.uint8)  

    for pixel, point, point_z in zip(pixels, points, z_axis):

        if point_z < 1e-5 and point_z > -1e-5:
            point_z = 1e-5
        color = int((1 - point_z/max_z) * 255)
        try:
            cv.circle(canvas, tuple(pixel.astype(int)), 2, (color, color, color), 2)
        except: pass
    return canvas


def rollslider_callback(val):
    global roll
    roll = np.radians(-val/10)
    print(f"roll+ rad: {roll}")


def pitchslider_callback(val):
    global pitch
    pitch = np.radians(-val/10)
    print(f"pitch rad: {pitch}")

def yawslider_callback(val):
    global yaw
    yaw = np.radians(-val/10)
    print(f"yaw rad: {yaw}")

def cameradistance_callback(val):
    global camera_distance
    camera_distance = val/20

def focallenght_callback(val):
    global focal_lenght
    focal_lenght = val/10

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
    "camera", "plot", 0, 200, cameradistance_callback
)
cv.createTrackbar(
    "focal_length", "plot", 0, 1500, focallenght_callback
)

while True: 
    
    
    canvas = render_points(roll, pitch, yaw, points, camera_distance)
    cv.imshow("plot", canvas)
    cv.waitKey(1)


