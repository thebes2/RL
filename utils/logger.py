from utils.constants import Color


class Logger:

    """
    Basic logging class
    TODO: allow for writing the stream to file as well
    """

    def __init__(self):
        pass

    def info(self, *args):
        print(Color.INFO, *args, Color.CLEAR)

    def success(self, *args):
        print(Color.SUCCESS, *args, Color.CLEAR)

    def warning(self, *args):
        print(Color.WARNING, *args, Color.CLEAR)

    def error(self, *args):
        print(Color.ERROR, *args, Color.CLEAR)

    def log(self, *args):
        print(Color.CLEAR, *args)


logger = Logger()
