import sys
import logging
import time
from datetime import timedelta

class LogFormatter():
    def __init__(self):
        self.start_time = time.time()

    def format(self, record):
        elapsed_seconds = round(record.created - self.start_time)

        prefix = "%s - %s - %s" % (record.levelname, time.strftime('%x %X'),
                                   timedelta(seconds=elapsed_seconds))
        message = record.getMessage()
        message = message.replace('\n', '\n' + ' ' * (len(prefix) + 3))
        return "%s - %s" % (prefix, message) if message else ''

def get_logger():
    '''
    create logger and output to file and stdout
    '''
    log_formatter = LogFormatter()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(log_formatter)
    logger.addHandler(stream)
    return logger