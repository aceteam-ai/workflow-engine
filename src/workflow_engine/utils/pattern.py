import re

MODULE_PATTERN = re.compile(r"^\w+(\.\w+)*$")
NAME_PATTERN = re.compile(r"^\w+$")
MODULE_NAME_PATTERN = re.compile(r"^(\w+(\.\w+)*)\.(\w+)$")


__all__ = [
    "MODULE_NAME_PATTERN",
    "MODULE_PATTERN",
    "NAME_PATTERN",
]
