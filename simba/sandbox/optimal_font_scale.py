import cv2
from typing import Optional, Tuple, Union, List
from simba.utils.checks import check_int, check_valid_tuple

def get_optimal_font_scale(text: Union[str, List[str]],
                           accepted_px_width: int,
                           accepted_px_height: int,
                           text_thickness: Optional[int] = 2,
                           font: Optional[int] = cv2.FONT_HERSHEY_TRIPLEX) -> Tuple[float, int, int]:

    """
    Get the optimal font size, column-wise and row-wise text distance of printed text for printing on images.

    :param str text: The text to be printed. Either a string or a list of strings. If a list, then the longest string will be used to evaluate spacings/font.
    :param int accepted_px_width: The widest allowed string in pixels. E.g., 1/4th of the image width.
    :param int accepted_px_height: The highest allowed string in pixels. E.g., 1/10th of the image size.
    :param Optional[int] text_thickness: The thickness of the font. Default: 2.
    :param Optional[int] font: The font integer representation 0-7. See ``simba.utils.enums.Options.CV2_FONTS.values
    :returns Tuple[int, int, int]: The font size, the shift on x between successive columns, the shift in y between successive rows.

    :example:
    >>> img = cv2.imread('/Users/simon/Desktop/Screenshot 2024-07-08 at 4.46.03 PM.png')
    >>> accepted_px_width = int(img.shape[1] / 4)
    >>> accepted_px_height = int(img.shape[0] / 10)
    >>>>text = 'HELLO MY FELLOW'
    >>> get_optimal_font_scale(text=text, accepted_px_width=accepted_px_width, accepted_px_height=accepted_px_height, text_thickness=2)
    """

    check_int(name='accepted_px_width', value=accepted_px_width, min_value=1)
    check_int(name='accepted_px_height', value=accepted_px_height, min_value=1)
    check_int(name='text_thickness', value=text_thickness, min_value=1)
    check_int(name='font', value=font, min_value=0, max_value=7)
    for scale in reversed(range(0, 100, 1)):
        text_size = cv2.getTextSize(text, fontFace=font, fontScale=scale/10, thickness=text_thickness)
        print(text_size)
        new_width, new_height = text_size[0][0], text_size[0][1]
        if (new_width <= accepted_px_width) and (new_height <= accepted_px_height):
            font_scale = scale / 10
            x_shift = new_width + text_size[1]
            y_shift = new_height + text_size[1]
            return (font_scale, x_shift, y_shift)
    return None, None, None


def get_optimal_circle_size(frame_size: Tuple[int, int],
                            circle_frame_ratio: Optional[int] = 100):

    check_int(name='accepted_circle_size', value=circle_frame_ratio, min_value=1)
    check_valid_tuple(x=frame_size, source='frame_size', accepted_lengths=(2,), valid_dtypes=(int,))
    for i in frame_size:
        check_int(name='frame_size', value=i, min_value=1)
    return int(max(frame_size[0], frame_size[1]) / circle_frame_ratio)




#
# img = cv2.imread('/Users/simon/Desktop/Screenshot 2024-07-08 at 4.46.03 PM.png')
# text_width = int(img.shape[1] / 4)
# text_height = int(img.shape[0] / 10)
# text = 'HELLO MY FELLOW'
# get_optimal_font_scale(text=text, accepted_px_width=text_width, accepted_px_height=text_height, text_thickness=2)
#
# #img = cv2.putText(img=img, text=text, org=(10, 10), fontScale=fontScale, fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0), thickness=2)
# img = cv2.putText(img=img, text=text, org=(10, 10+y_shift), fontScale=fontScale, fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0), thickness=2)
# img = cv2.putText(img=img, text=text, org=(10, 10+(y_shift*2)), fontScale=fontScale, fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0), thickness=2)
# img = cv2.putText(img=img, text=text, org=(10+x_shift, 10+y_shift), fontScale=fontScale, fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0, 255, 0), thickness=2)
#
#
# cv2.imshow('sadasd', img)
# cv2.waitKey(5000)

