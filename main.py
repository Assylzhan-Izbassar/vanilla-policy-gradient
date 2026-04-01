import torch
import random
import numpy as np
from config import args
from train import train

if __name__ == "__main__":
    # TRY NOT TO MODIFY: seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train(
        env_name=args.env_name, hidden_size=[32], lr=args.lr, epochs=50, batch_size=5000
    )
