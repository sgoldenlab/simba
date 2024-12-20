import cv2
import numpy as np

# Initial values
center = None
radius = 0
drawing = False
img = np.zeros((512, 512, 3), np.uint8)
overlay = img.copy()


def draw_circle(event, x, y, flags, param):
    global center, radius, drawing, img, overlay

    if event == cv2.EVENT_LBUTTONDOWN:
        center = (x, y)
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            overlay = img.copy()
            radius = int(((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5)
            angle1, angle2 = 0, 0
            if x >= center[0] and y <= center[1]:  # Top-right quadrant
                angle1, angle2 = 0, 90
                # Draw the straight lines
                cv2.line(overlay, center, (center[0] + radius, center[1]), (0, 255, 0), 2)
                cv2.line(overlay, center, (center[0], center[1] - radius), (0, 255, 0), 2)
            elif x <= center[0] and y <= center[1]:  # Top-left quadrant
                angle1, angle2 = 90, 180
                # Draw the straight lines
                cv2.line(overlay, center, (center[0] - radius, center[1]), (0, 255, 0), 2)
                cv2.line(overlay, center, (center[0], center[1] - radius), (0, 255, 0), 2)
            elif x <= center[0] and y >= center[1]:  # Bottom-left quadrant
                angle1, angle2 = 180, 270
                # Draw the straight lines
                cv2.line(overlay, center, (center[0] - radius, center[1]), (0, 255, 0), 2)
                cv2.line(overlay, center, (center[0], center[1] + radius), (0, 255, 0), 2)
            elif x >= center[0] and y >= center[1]:  # Bottom-right quadrant
                angle1, angle2 = 270, 360
                # Draw the straight lines
                cv2.line(overlay, center, (center[0] + radius, center[1]), (0, 255, 0), 2)
                cv2.line(overlay, center, (center[0], center[1] + radius), (0, 255, 0), 2)

            # Draw the quarter circle arc
            cv2.ellipse(overlay, center, (radius, radius), 0, angle1, angle2, (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        radius = int(((x - center[0]) ** 2 + (y - center[1]) ** 2) ** 0.5)
        angle1, angle2 = 0, 0
        if x >= center[0] and y <= center[1]:  # Top-right quadrant
            angle1, angle2 = 0, 90
            # Draw the straight lines
            cv2.line(img, center, (center[0] + radius, center[1]), (0, 255, 0), 2)
            cv2.line(img, center, (center[0], center[1] - radius), (0, 255, 0), 2)
        elif x <= center[0] and y <= center[1]:  # Top-left quadrant
            angle1, angle2 = 90, 180
            # Draw the straight lines
            cv2.line(img, center, (center[0] - radius, center[1]), (0, 255, 0), 2)
            cv2.line(img, center, (center[0], center[1] - radius), (0, 255, 0), 2)
        elif x <= center[0] and y >= center[1]:  # Bottom-left quadrant
            angle1, angle2 = 180, 270
            # Draw the straight lines
            cv2.line(img, center, (center[0] - radius, center[1]), (0, 255, 0), 2)
            cv2.line(img, center, (center[0], center[1] + radius), (0, 255, 0), 2)
        elif x >= center[0] and y >= center[1]:  # Bottom-right quadrant
            angle1, angle2 = 270, 360
            # Draw the straight lines
            cv2.line(img, center, (center[0] + radius, center[1]), (0, 255, 0), 2)
            cv2.line(img, center, (center[0], center[1] + radius), (0, 255, 0), 2)

        # Draw the filled quarter circle
        cv2.ellipse(img, center, (radius, radius), 0, angle1, angle2, (0, 255, 0), -1)
        overlay = img.copy()


# Create a black image
cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

while True:
    cv2.imshow('image', overlay if drawing else img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cv2.destroyAllWindows()
