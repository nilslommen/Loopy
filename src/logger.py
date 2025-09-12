import logging

logger = logging.getLogger("ConstantRuntime")
logger.setLevel(logging.INFO)

stream = logging.StreamHandler()
stream.setLevel(logging.DEBUG)
logger.addHandler(stream)

def log_string(str):
  logger.info(str)
