from enum import Enum


class Color(Enum):
    ERROR = "\033[91m"
    WARNING = "\033[93m"
    INFO = "\033[92m"
    SUCCESS = "\033[96m"
    CLEAR = "\033[0m"
