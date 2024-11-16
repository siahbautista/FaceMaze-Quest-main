import cv2
import numpy as np
import pyautogui as gui
import time

# Set keypress delay to 0.
gui.PAUSE = 0

# Loading the pre-trained face model.
model_path = './model/face_model.caffemodel'
prototxt_path = './model/face_model_config.prototxt'

# Define the maze layout: 1 is wall, 0 is path.
# This is a simple maze for demonstration. You can make it larger or more complex.
maze = [
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # Start at [0, 0]
    [0, 1, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 1, 1],
    [1, 0, 1, 0, 0, 0, 1, 1, 1, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
    [1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0]   # Finish at [9, 9]
]

player_position = [1, 1]  # Starting position of the player

# Load assets
person_sprite = cv2.imread('player.png')  # 50x50 sprite of the player
tile_sprite = cv2.imread('tile.png')  # Wall tileset (one square tile)
floor_sprite = cv2.imread('floor.png')  # Floor tileset (one square tile)

# Resize sprites to fit the maze grid (50x50 pixels per tile)
tile_sprite = cv2.resize(tile_sprite, (50, 50))
floor_sprite = cv2.resize(floor_sprite, (50, 50))
person_sprite = cv2.resize(person_sprite, (50, 50))  # Resize player sprite to fit maze cell

def detect(net, frame):
    detected_faces = []
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            detected_faces.append({'start': (startX, startY), 'end': (endX, endY), 'confidence': confidence})
    return detected_faces

def drawFace(frame, detected_faces):
    for face in detected_faces:
        cv2.rectangle(frame, face['start'], face['end'], (0, 255, 0), 10)
    return frame

def checkRect(detected_faces, bbox):
    for face in detected_faces:
        x1, y1 = face['start']
        x2, y2 = face['end']
        if x1 > bbox[0] and x2 < bbox[1]:
            if y1 > bbox[3] and y2 < bbox[2]:
                return True
    return False

def move_player(direction):
    global player_position
    x, y = player_position
    if direction == 'up' and maze[x - 1][y] == 0:
        player_position[0] -= 1
    elif direction == 'down' and maze[x + 1][y] == 0:
        player_position[0] += 1
    elif direction == 'left' and maze[x][y - 1] == 0:
        player_position[1] -= 1
    elif direction == 'right' and maze[x][y + 1] == 0:
        player_position[1] += 1

def move(detected_faces, bbox):
    global last_mov
    for face in detected_faces:
        x1, y1 = face['start']
        x2, y2 = face['end']
        if checkRect(detected_faces, bbox):
            last_mov = 'center'
            return
        elif last_mov == 'center':
            if x1 < bbox[0]:
                move_player('left')
                last_mov = 'left'
            elif x2 > bbox[1]:
                move_player('right')
                last_mov = 'right'
            if y2 > bbox[2]:
                move_player('down')
                last_mov = 'down'
            elif y1 < bbox[3]:
                move_player('up')
                last_mov = 'up'

def draw_maze():
    maze_frame = np.zeros((len(maze) * 50, len(maze[0]) * 50, 3), dtype=np.uint8)

    for i in range(len(maze)):
        for j in range(len(maze[0])):
            # Place the tiles based on the maze structure
            if maze[i][j] == 1:
                maze_frame[i*50:(i+1)*50, j*50:(j+1)*50] = tile_sprite  # Wall tile
            else:
                maze_frame[i*50:(i+1)*50, j*50:(j+1)*50] = floor_sprite  # Floor tile

    # Draw the player sprite at the current position
    px, py = player_position
    maze_frame[px*50:(px+1)*50, py*50:(py+1)*50] = person_sprite

    return maze_frame

def play(prototxt_path, model_path):
    global last_mov
    last_mov = ''
    net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
    cap = cv2.VideoCapture(0)

    while not cap.isOpened():
        cap = cv2.VideoCapture(0)

    # Co-ordinates of the bounding box on frame
    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))
    box_width, box_height = 250, 250  # Set smaller bounding box dimensions
    left_x, top_y = frame_width // 2 - box_width // 2, frame_height // 2 - box_height // 2
    right_x, bottom_y = frame_width // 2 + box_width // 2, frame_height // 2 + box_height // 2
    bbox = [left_x, right_x, bottom_y, top_y]

    while True:
        ret, frame = cap.read()
        if not ret:
            return 0

        frame = cv2.flip(frame, 1)
        detected_faces = detect(net, frame)
        frame = drawFace(frame, detected_faces)
        frame = cv2.rectangle(frame, (left_x, top_y), (right_x, bottom_y), (0, 0, 255), 5)

        move(detected_faces, bbox)

        # Update and display the maze view with the new assets
        maze_view = draw_maze()
        cv2.imshow('maze_view', maze_view)

        # Display the camera feed
        cv2.imshow('camera_feed', frame)
        if cv2.waitKey(5) == 27:  # Exit on 'esc'
            break

if __name__ == "__main__":
    play(prototxt_path, model_path)
