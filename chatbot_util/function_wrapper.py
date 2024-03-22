import utils
import config as conf


class FunctionWrapper(object):
    def __init__(self, conf):
        self.llm = utils.LLM(conf)
        self.emb = utils.EMB(conf)


if __name__ == '__main__':
    function_runtime = FunctionWrapper(conf)
    print('load 2 model done')
