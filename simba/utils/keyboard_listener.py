# from pynput import keyboard
#
# class KeyboardListener(object):
#     """ Helper class to check status of keyboard buttons"""
#     def __init__(self):
#         self.shift = False
#         self.space = False
#
#     def press(self, key):
#         if key == keyboard.Key.space:
#             self.space = True
#         elif key == keyboard.Key.shift:
#             self.shift = True
#
#     def release(self, key):
#         if key == keyboard.Key.space:
#             self.space = False
#         elif key == keyboard.Key.shift:
#             self.shift = False
#
#     def start(self):
#         with keyboard.Listener(on_press=self.press, on_release=self.release) as listener:
#             listener.join()
#
