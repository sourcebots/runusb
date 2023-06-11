BOARD = 10

LOW = 0
HIGH = 1

OUTPUT = OUT = 0
INPUT = 1

def setmode(mode: int) -> None: ...

def setwarnings(warnings: bool) -> None: ...

def setup(pins: list[int], mode: int, *, initial: int) -> None: ...

def output(pin: int, mode: int) -> None: ...
