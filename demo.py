import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s-%(levelname)s-%(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)
logging.debug("Dataset shape:(1000,12)")
logging.info("Model Training started successfully")
logging.warning("Many values are missing")
logging.error("Failed to load dataset")
logging.critical("Databases connection can not be established")