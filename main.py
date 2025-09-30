import torch
from utils import *
from sampler import Sampler
import random
import numpy as np

def same_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def main(args):
    logger = get_logger(args)
    sampler = Sampler(args)

    output1 = sampler.autoregressive()
    logger.info("model answer:{}".format(sampler.tokenizer.decode(output1[0])))
    inference_speed(sampler.autoregressive, 1, logger, "autoregressive")

    output2 = sampler.speculative()
    logger.info("model answer:{}".format(sampler.tokenizer.decode(output2[0])))
    inference_speed(sampler.speculative, 1, logger, "speculative")

if __name__ == "__main__":
    args = args_parser()
    same_seed(args.seed)
    main(args)