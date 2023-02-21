import warnings

class FileNotFoundWarning(Warning):
    def __init__(self, message):
        self.message = message