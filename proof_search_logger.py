import logging
logger = logging.getLogger(__name__)
logging.basicConfig(filename='proof_search.log', encoding='utf-8', level=logging.DEBUG)

if __name__ == "__main__":
    logger.debug("test log")
    logger.info("test log")
    logger.error("test log")