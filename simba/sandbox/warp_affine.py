import cv2
import numpy as np

# Original coordinates of the body parts (7 coordinates)
original_coords = np.array([
    [182.94406, 277.93008],  # Body part 1
    [345.5105, 301.44055],  # Body part 2 (fixed)
    [309.65036, 257.01398],  # Body part 3
    [215, 260],  # Body part 4
    [266.0979, 307.34265],  # Body part 5
    [288.1888, 244.2098],  # Body part 6
    [273.5944, 276.4965],  # Body part 7
    [413.6084, 317.83917],  # Body part 8
    [474.93707, 311.13287]  # Body part 9
])

# Load the image
image = cv2.imread(r"C:\troubleshooting\mitra\project_folder\csv\outlier_corrected_movement_location\test\bg_temp\FL_gq_Saline_0626_frames\0.png")

# Specify fixed locations
fixed_point1 = np.array([100, original_coords[0, 1]])  # Fix body part 1
fixed_point2 = np.array([300, original_coords[1, 1]])  # Fix body part 2

# Update the fixed points
original_coords[0] = fixed_point1
original_coords[1] = fixed_point2

# Calculate the original distances
original_distance_x = original_coords[1, 0] - original_coords[0, 0]
original_distance_y = original_coords[1, 1] - original_coords[0, 1]

# Calculate new coordinates for the other points
for i in range(2, len(original_coords)):
    scale_x = (fixed_point2[0] - fixed_point1[0]) / original_distance_x
    scale_y = (original_coords[1, 1] - original_coords[0, 1]) / original_distance_y

    new_x = fixed_point1[0] + (original_coords[i, 0] - original_coords[0, 0]) * scale_x
    new_y = fixed_point1[1] + (original_coords[i, 1] - original_coords[0, 1]) * scale_y

    original_coords[i] = [new_x, new_y]

# Calculate the center of the body parts for rotation
center = np.mean(original_coords, axis=0)

# Calculate the angle of the original line connecting body part 1 and body part 2
dx = original_coords[1, 0] - original_coords[0, 0]
dy = original_coords[1, 1] - original_coords[0, 1]
original_angle = np.arctan2(dy, dx)  # Angle in radians

# Target angle for facing right (0 radians)
target_angle = 0  # 0 degrees or facing right

# Calculate the angle of rotation needed
angle_of_rotation = np.degrees(target_angle - original_angle)  # Convert to degrees

# Create the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle_of_rotation, 1)

# Apply the rotation transformation to the original coordinates
for i in range(len(original_coords)):
    # Convert to homogeneous coordinates
    coord_homogeneous = np.append(original_coords[i], 1)
    # Apply the rotation
    new_coord = rotation_matrix @ coord_homogeneous
    original_coords[i] = new_coord[:2]

# Debug: Print out the new coordinates
print("New coordinates:")
print(original_coords)

# Draw circles at the new body part locations
for coord in original_coords:
    cv2.circle(image, (int(coord[0]), int(coord[1])), 10, (0, 0, 255), -1)  # Larger red circles

# Display the modified image without saving
cv2.imshow('Modified Image', image)
cv2.waitKey(0)  # Wait for a key press
cv2.destroyAllWindows()  # Close the image window
