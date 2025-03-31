import pygame
import sys
import numpy as np
import cv2
from pygame.locals import *
from keras.models import load_model

# Constants
WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDRYINC = 5

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False
MODEL = load_model("bestmodel.h5")

LABELS = {0: "Zero", 1: "One", 2: "Two", 3: "Three", 4: "Four",
          5: "Five", 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine"}

# Initialize pygame
pygame.init()

FONT = pygame.font.Font(None, 36)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Recognition Board")

iswriting = False
number_xcord = []
number_ycord = []
image_cnt = 1
PREDICT = True

DISPLAYSURF.fill(BLACK)  # Fill screen with black initially

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        
        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            if len(number_xcord) > 0:
                pygame.draw.line(DISPLAYSURF, WHITE, (number_xcord[-1], number_ycord[-1]), (xcord, ycord), 8)
            number_xcord.append(xcord)
            number_ycord.append(ycord)
        
        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        
        if event.type == MOUSEBUTTONUP:
            iswriting = False
            if len(number_xcord) > 0 and len(number_ycord) > 0:
                # Get bounding box
                rec_min_x = max(min(number_xcord) - BOUNDRYINC, 0)
                rec_max_x = min(max(number_xcord) + BOUNDRYINC, WINDOWSIZEX)
                rec_min_y = max(min(number_ycord) - BOUNDRYINC, 0)
                rec_max_y = min(max(number_ycord) + BOUNDRYINC, WINDOWSIZEY)

                # Extract drawn image
                img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rec_min_x:rec_max_x, rec_min_y:rec_max_y].T.astype(np.float32)

                # Save image if needed
                if IMAGESAVE:
                    cv2.imwrite(f"image_{image_cnt}.png", img_arr)
                    image_cnt += 1

                # Predict the digit
                if PREDICT:
                    image = cv2.resize(img_arr, (28, 28))
                    image = np.pad(image, (10, 10), mode='constant', constant_values=0)
                    image = cv2.resize(image, (28, 28)) / 255.0

                    # Model prediction
                    prediction = MODEL.predict(image.reshape(1, 28, 28, 1))
                    label = str(LABELS[np.argmax(prediction)])

                    # Draw bounding box
                    pygame.draw.rect(DISPLAYSURF, RED, (rec_min_x, rec_min_y, rec_max_x - rec_min_x, rec_max_y - rec_min_y), 2)

                    # Display the predicted label
                    textSurface = FONT.render(label, True, RED, WHITE)
                    DISPLAYSURF.blit(textSurface, (rec_min_x, rec_max_y))

                # Reset the coordinates
                number_xcord = []
                number_ycord = []

        # Clear screen on 'N' key press
        if event.type == KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
