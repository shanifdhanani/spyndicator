import logging
import logging.config
import sys

from datetime import datetime
import os

class Logger:
    """
    This is a utility class that makes it easy to return a built in Python logger.  This class is needed
    to standardize the way we initialize the logger.
    """

    LoggingDirectory = os.path.abspath("/tmp/spyndicator/logs")
    LogName = 'spyndicator' + "{:_%Y-%m-%d_%H_%M_%S}".format(datetime.now()) + '.log'

    os.makedirs(LoggingDirectory, exist_ok = True)
    logging.getLogger("botocore.vendored.requests.packages.urllib3.connectionpool").setLevel(logging.WARNING)

    DefaultLogLevel = "INFO"

    logging.config.dictConfig({
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d]: %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': DefaultLogLevel,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                "stream": sys.stdout
            },
            'file_handler': {
                'class': 'logging.FileHandler',
                "formatter": "standard",
                "filename": os.path.join(LoggingDirectory, LogName),
                "encoding": "utf8"
            }
        },
        'loggers': {
            '': {
                'handlers': ['default', 'file_handler'],
                'level': DefaultLogLevel,
                'propagate': True
            }
        }
    })

    @staticmethod
    def get_logger(name):
        """
        This method returns a logger with our standard configuration

        :return: Logger
        """

        return logging.getLogger(name)