import numpy as np


def egocentric_align_keypoints_fixed_tail(keypoints, nose, tail, target_angle_degrees, tail_target=(250, 250)):
    # Convert target angle to radians
    target_angle = np.deg2rad(target_angle_degrees)

    # Calculate the current angle of the nose-tail vector
    delta_x = nose[0] - tail[0]
    delta_y = nose[1] - tail[1]
    current_angle = np.arctan2(delta_y, delta_x)

    # Calculate the required rotation angle
    rotate_angle = target_angle - current_angle

    # Create the rotation matrix
    cos_theta = np.cos(rotate_angle)
    sin_theta = np.sin(rotate_angle)
    R = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Translate keypoints so that the tail is at the origin
    keypoints_translated = keypoints - tail

    # Apply rotation
    keypoints_rotated = np.dot(keypoints_translated, R.T)

    # Translate keypoints so that the tail is at the target position (250, 250)
    tail_position_after_rotation = keypoints_rotated[-1]  # Assuming tail is the last point in the array
    translation_to_target = np.array(tail_target) - tail_position_after_rotation
    keypoints_aligned = keypoints_rotated + translation_to_target

    return keypoints_aligned


# Example usage
keypoints = np.array([[x1, y1], [x2, y2], ..., [xn, yn]])  # Replace with actual keypoints
nose = keypoints[0]  # Assuming nose is at index 0
tail = keypoints[-1]  # Assuming tail is at the last index
target_angle_degrees = 90  # Target angle

aligned_keypoints = egocentric_align_keypoints_fixed_tail(keypoints, nose, tail, target_angle_degrees)
