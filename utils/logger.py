from utils.constants import Color


class Logger:

    """
    Basic logging class
    TODO: allow for writing the stream to file as well
    """

    def __init__(self, file=None):
        self.out_file = open(file, "a") if file is not None else None

    def __del__(self):
        if self.out_file is not None:
            self.out_file.close()

    def _concat(*args):
        return " ".join(list(map(repr, list(args))))

    def info(self, *args):
        print(Color.INFO.value, *args, Color.CLEAR.value)
        if self.out_file is not None:
            self.out_file.write(Logger._concat("[INFO]", *args))
            self.out_file.write("\n")

    def success(self, *args):
        print(Color.SUCCESS.value, *args, Color.CLEAR.value)
        if self.out_file is not None:
            self.out_file.write(Logger._concat("[SUCCESS]", *args))
            self.out_file.write("\n")

    def warning(self, *args):
        print(Color.WARNING.value, *args, Color.CLEAR.value)
        if self.out_file is not None:
            self.out_file.write(Logger._concat("[WARNING]", *args))
            self.out_file.write("\n")

    def error(self, *args):
        print(Color.ERROR.value, *args, Color.CLEAR.value)
        if self.out_file is not None:
            self.out_file.write(Logger._concat("[ERROR]", *args))
            self.out_file.write("\n")

    def log(self, *args):
        print(Color.CLEAR.value, *args)
        if self.out_file is not None:
            self.out_file.write(Logger._concat(*args))
            self.out_file.write("\n")


logger = Logger()
