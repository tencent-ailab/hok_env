__all__ = ["__version__", "get_version", "GAMECORE_VERSION"]

version_info = (45, 1, 5)
GAMECORE_VERSION = "v45_1450123"


def get_version():
    """Returns the version as a human-format string."""
    return "%d.%d.%d" % version_info


__version__ = get_version()
