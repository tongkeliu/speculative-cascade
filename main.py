import torch
from utils import *
from sampler import Sampler
import contexttimer

def main(args):
    logger = get_logger(args)
    sampler = Sampler(args)
    output1 = sampler.autoregressive()
    logger.debug("model answer:{}".format(sampler.tokenizer.decode(output1)))
    inference_speed(sampler.autoregressive, 1, logger, "autoregressive")

if __name__ == "__main__":
    args = args_parser()
    main(args)