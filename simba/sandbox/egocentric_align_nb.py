# In this notebook, we will "egocentrically" align pose estimation and pose-estimated video data.

# This means that we will rotate the data, so that the animal, in every frame, is always "anchored" in the same location and directing to the same location.
# (i) One body-part (e.g., the center or the tail-base of the animal is always located in the same pixel location of the video.
# (ii) A second body-part (e.g., the nose, head, or nape) is always directing N degrees from the anchor point.

# In short - we rotate the data so that the animal is always facing to the right, and the animal is always located at
# the center of the image.




