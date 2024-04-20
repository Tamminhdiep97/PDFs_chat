import time
from loguru import logger

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .api_handler import handler
from .utils import setup_logger


def app_setting():
    # log setting
    time_stamp = time.strftime('%Y_%m_%d_%H%M%S', time.localtime())
    log_file_path = 'backend_logs/app_{}.log'.format(time_stamp)
    logger = setup_logger(
        log_system = config.LOG_SYSTEM,
        log_level_system = config.LOG_LEVEL_SYSTEM,
        log_file_path=log_file_path,
        level=config.LOG_LEVEL,
        rotation=config.ROTATION,
        retention=config.RETENTION,
    )

    logger.info('Init the engine')

    app = FastAPI(
        title=config.APP_NAME,
        version=config.API_VERSION,
        description=config.API_DESCRIPTION,
        debug=config.DEBUG,
    )

    logger.info('Started the engine')

    # setting logger
    app.logger = logger

    # add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=['*'],
        allow_credentials=True,
        allow_methods=['*'],
        allow_headers=['*'],
    )
    # setting router
    app.include_router(handler.api_router, prefix=config.API_PREFIX)
    return app


with logger.catch():
    app = app_setting()
