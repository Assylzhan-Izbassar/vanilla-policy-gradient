from config import args
from train import train

if __name__ == "__main__":
    train(
        env_name=args.env_name, hidden_size=[32], lr=args.lr, epochs=50, batch_size=5000
    )
