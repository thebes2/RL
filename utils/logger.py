from utils.constants import Color


class Logger:

    """
    Basic logging class
    TODO: allow for writing the stream to file as well
    """

    def __init__(self):
        pass

    def info(self, *args):
        print(Color.INFO.value, *args, Color.CLEAR.value)

    def success(self, *args):
        print(Color.SUCCESS.value, *args, Color.CLEAR.value)

    def warning(self, *args):
        print(Color.WARNING.value, *args, Color.CLEAR.value)

    def error(self, *args):
        print(Color.ERROR.value, *args, Color.CLEAR.value)

    def log(self, *args):
        print(Color.CLEAR.value, *args)


logger = Logger()
