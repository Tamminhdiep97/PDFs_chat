from loguru import logger

import utils
import config as conf


class FunctionWrapper(object):
    def __init__(self, conf):
        self.llm = utils.LLM(conf)
        self.emb = utils.EMB(conf)
        self.vectorDb = utils.VectorDB(conf)


if __name__ == '__main__':
    function_runtime = FunctionWrapper(conf)
    logger.info('load 3 model done')
