import logging
from typing import Any, Mapping


class RelativeTimeFilter(logging.Filter):
    def __init__(self, *args, **kwargs):
        ...

    def filter(self, record: logging.LogRecord) -> bool:
        ...

    def reset_time_reference(self) -> None:
        ...


class TieredFormatter(logging.Formatter):
    def __init__(
        self,
        fmt: str | None = None,
        level_fmts: Mapping[int, str] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        ...

    def format(self, record: logging.LogRecord) -> str:
        ...
