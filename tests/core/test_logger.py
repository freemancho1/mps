from mps.core.libs import logger 

def test_logger():
    logger.debug("logger debug....")
    logger.info("logger debug....")
    logger.point("logger debug....")
    logger.warning("logger debug....")
    logger.error("logger debug....")
    logger.critical("logger debug....")