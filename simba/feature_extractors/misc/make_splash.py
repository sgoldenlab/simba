from pygifsicle import optimize
pt = optimize("/Users/simon/Downloads/Studio_Project (3).gif")



# # import cv2
# # import numpy as np
# # import os
# #
# # IMG_PATH = '/Users/simon/Desktop/splash.png'
# # OUT_PATH = '/Users/simon/Desktop/envs/simba_dev/splash'
# # alpha_lst = np.linspace(0, 1, 25).tolist()
# # for i in range(25):
# #     alpha_lst.append(1)
# #
# # #alpha_lst = alpha_lst[0:1]
# #
# # for cnt, i in enumerate(alpha_lst):
# #     img = cv2.imread(IMG_PATH)
# #     bgra = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
# #     bgra[..., 3] = int(254 * i)
# #     cv2.imwrite(os.path.join(OUT_PATH, 'image_' + str(cnt) + '.png'), bgra)
# #
#
#
#
# import glob
# import contextlib
# from PIL import Image
#
# # filepaths
# fp_in = "/Users/simon/Desktop/envs/simba_dev/splash/image_*.png"
# fp_out = "/Users/simon/Desktop/envs/simba_dev/splash_gif.gif"
#
# # use exit stack to automatically close opened images
# with contextlib.ExitStack() as stack:
#
#     # lazily load images
#     imgs = (stack.enter_context(Image.open(f))
#             for f in sorted(glob.glob(fp_in)))
#     print(imgs)
#
#     # extract  first image from iterator
#     img = next(imgs)
#
#     # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
#     img.save(fp=fp_out, format='GIF', append_images=imgs,
#              save_all=True, duration=2500, loop=1)
#
#
#
#
# frames = [Image.open(image) for image in glob.glob(f"{'/Users/simon/Desktop/envs/simba_dev/splash/'}/*")]
# frame_one = frames[0]
# frame_one.save("download.gif", format="GIF", append_images=frames, save_all=True, duration=1000, loop=1)
