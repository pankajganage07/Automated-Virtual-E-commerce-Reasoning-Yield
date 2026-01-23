import logging
from logging.config import dictConfig

_LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {"format": _LOG_FORMAT},
        "uvicorn": {"format": "%(levelprefix)s %(asctime)s %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "DEBUG",
        },
    },
    "loggers": {
        "": {"handlers": ["console"], "level": "INFO"},
        "uvicorn.error": {"level": "INFO"},
        "uvicorn.access": {"handlers": ["console"], "propagate": False},
    },
}


def setup_logging(level: str = "INFO") -> None:
    config = LOGGING_CONFIG.copy()
    config["loggers"][""]["level"] = level.upper()
    dictConfig(config)
    logging.getLogger(__name__).debug("Logging configured with level %s", level)
