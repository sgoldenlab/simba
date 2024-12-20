


#In this notebook, we will egocentrically align pose estimation and associated video data.

# Egocentric alignment is a crucial preprocessing step with various applications across different domains.

# One primary use case is in unsupervised deep learning scenarios, where images of animals are used as inputs to train algorithms.
# Standardizing the orientation of the subject (e.g., an animal) ensures that the model and subsequent analyses do not incorrectly interpret the same behavior in different orientations
# # (e.g., grooming while facing north versus south) as distinct behaviors. By aligning the animal to a consistent frame of reference, we eliminate orientation as a confounding variable.

# While egocentric alignment is an essential first step, it is often insufficient by itself for comprehensive analyses. Additional preprocessing steps are typically required, such as:
#
# * Background subtraction to isolate the animal from its surroundings (see relevant methods and notebooks).
# *  Geometric segmentation to slice out and focus on the subject's body parts (see associated code and notebooks).

# In this notebook, we will focus exclusively on performing egocentric alignment. Further preprocessing steps are outlined in related materials.

from simba.data_processors.egocentric_aligner import EgocentricalAligner

#SETTINGS
ANCHOR_POINT_1 = 'center'       # Name of the body-part which is the "primary" anchor point around which the alignment centers. In rodents, this is often the center of the tail-base of the animal.
ANCHOR_POINT_2 = 'nose'         # The name of the secondary anchor point defining the alignment direction. This is often the anterior body-part, in rodents it can be the nose or nape.
DIRECTION = 0                   # The egocentric alignment angle, in degrees. For example, `0` and the animals `ANCHOR_POINT_2` is directly to the east (right) of `ANCHOR_POINT_1`. `180` and the animals `ANCHOR_POINT_2` is directly to the west (left) of `ANCHOR_POINT_1`.
ANCHOR_LOCATION = (250, 250)    # The pixel location in the video where `ANCHOR_POINT_1` should be placed. For example, if the videos are 500x500, 250x250 will place the anchor right in the middle.
GPU = True                      # If we have an NVIDEA GPU availalable, we can use it to speed up processing. Otherwise set this to `False`.
FILL_COLOR = (0, 0, 0)          # We are rotating videos, while at the same time retaining the original video size. Therefore, there will be some "new" areas exposed in the video (see below for more info). This is what color to color these new areas.
VERBOSE = False                 # If True, prints progress (like which frame and video is being processed etc). However, this information will flood this notebook is I am going to turn it off.


#DIRECTORY WHICH IS HOLDING POSE-ESTIMATION DATA
DATA_DIRECTORY = r'C:\Users\sroni\OneDrive\Desktop\rotate_ex\data'

#DIRECTORY WHICH IS VIDEOS, ONE FOR EACH FILE IN THE DATA_DIRECTORY
VIDEOS_DIRECTORY = r'C:\Users\sroni\OneDrive\Desktop\rotate_ex\videos'

#DIRECTORY WHERE WE SHOULD SAVE THE ROTATED POSE-ESTIMATION AND ROTATED VIDEOS.
SAVE_DIRECTORY = r"C:\Users\sroni\OneDrive\Desktop\rotate_ex\rotated"


# Now we are good to go, using the information above, we define an instance of an SimBA EgocentriAligner and run it
aligner = EgocentricalAligner(anchor_1=ANCHOR_POINT_1,
                              anchor_2=ANCHOR_POINT_2,
                              data_dir=DATA_DIRECTORY,
                              videos_dir=VIDEOS_DIRECTORY,
                              save_dir=SAVE_DIRECTORY,
                              direction=DIRECTION,
                              gpu=GPU,
                              anchor_location=ANCHOR_LOCATION,
                              fill_clr=FILL_COLOR,
                              verbose=VERBOSE)
aligner.run()



#EXAMPLE VIDEO EXPECTED RESULTS
####

#Now, let's change a few settings, to get a feeling for how it behaves..
ANCHOR_LOCATION = (500, 100)
FILL_COLOR = (255, 0, 0)
DIRECTION = 180


# ... we create a new instance based on the updated information above, and run it.
aligner = EgocentricalAligner(anchor_1=ANCHOR_POINT_1,
                              anchor_2=ANCHOR_POINT_2,
                              data_dir=DATA_DIRECTORY,
                              videos_dir=VIDEOS_DIRECTORY,
                              save_dir=SAVE_DIRECTORY,
                              direction=DIRECTION,
                              gpu=GPU,
                              anchor_location=ANCHOR_LOCATION,
                              fill_clr=FILL_COLOR,
                              verbose=VERBOSE)
aligner.run()

#EXAMPLE VIDEO EXPECTED RESULTS
####



