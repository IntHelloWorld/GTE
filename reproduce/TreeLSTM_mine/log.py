import logging
import sys
from asyncio.log import logger


class Logger(logging.Logger):
    def __init__(self, name, file=None):
        super(Logger, self).__init__(name, logging.INFO)

        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        handler.flush = sys.stdout.flush
        super(Logger, self).addHandler(handler)
        if file is not None:
            file_handler = logging.FileHandler(filename=file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            super(Logger, self).addHandler(file_handler)
