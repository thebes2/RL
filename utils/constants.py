from enum import Enum


class Color(Enum):
    ERROR = "\033[91m"
    WARNING = "\033[93m"
    INFO = "\033[96m"
    SUCCESS = "\033[92m"
    CLEAR = "\033[0m"
