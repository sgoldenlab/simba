import numpy as np


# Function to rotate points around a given origin
def rotate_points(points, angle, origin=(0, 0)):
    """
    Rotate points by a given angle around the origin.
    points: array of shape (N, 2), where N is the number of points.
    angle: rotation angle in radians.
    origin: (x, y) coordinates of the origin of rotation.
    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])
    translated_points = points - origin  # Translate to the origin
    rotated_points = np.dot(translated_points, rotation_matrix.T)  # Apply rotation
    return rotated_points + origin  # Translate back

# Sample body parts data (Body1 to Body7)
body_parts = np.array([
    [x1, y1],  # Body1
    [x2, y2],  # Body2
    [x3, y3],  # Body3
    [x4, y4],  # Body4
    [x5, y5],  # Body5
    [x6, y6],  # Body6
    [x7, y7]   # Body7
])

# Choose 2 body parts for alignment (e.g., Body1, Body2)
body_part_indices = [0, 1]  # Indices for Body1 and Body2
chosen_body_parts = body_parts[body_part_indices]

# Center the data around the chosen reference body part (e.g., Body2)
reference_body_part = chosen_body_parts[1]  # Body2
centered_body_parts = chosen_body_parts - reference_body_part

# Vector from Body2 to Body1 (for alignment)
line_body1_body2 = centered_body_parts[0] - centered_body_parts[1]

# Calculate the angle to rotate the vector so it points north (along positive y-axis)
north_vector = np.array([0, 1])  # North is along the positive y-axis
angle_to_north = np.arctan2(line_body1_body2[1], line_body1_body2[0]) - np.arctan2(north_vector[1], north_vector[0])

# Rotate all body parts around the reference body part (Body2)
rotated_body_parts = rotate_points(body_parts, -angle_to_north, origin=reference_body_part)

# Output the rotated coordinates relative to the egocentric view
print("Rotated Body Parts:")
for i, part in enumerate(rotated_body_parts):
    print(f"Body{i+1}: {part}")
