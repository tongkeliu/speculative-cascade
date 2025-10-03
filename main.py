import torch
from utils import *
from sampler import Sampler
import random
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    logger = get_logger(args)
    sampler = Sampler(args)

    # set_seed(args.seed)
    # output = sampler.autoregressive()
    # logger.info("model answer:{}".format(sampler.tokenizer.decode(output[0])))
    # set_seed(args.seed)
    # inference_speed(sampler.autoregressive, 1, logger, "autoregressive")

    # set_seed(args.seed)
    # output = sampler.speculative()
    # logger.info("model answer:{}".format(sampler.tokenizer.decode(output[0])))
    # set_seed(args.seed)
    # inference_speed(sampler.speculative, 1, logger, "speculative")

    set_seed(args.seed)
    output = sampler.cascade()
    logger.info("model answer:{}".format(sampler.tokenizer.decode(output[0])))
    set_seed(args.seed)
    inference_speed(sampler.cascade, 1, logger, f"cascade-{args.deferral_rule}")

if __name__ == "__main__":
    args = args_parser()
    set_seed(args.seed)
    main(args)