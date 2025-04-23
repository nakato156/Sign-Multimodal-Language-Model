import argparse
from mslm.scripts import train as train_script, make_study

def main():
    p = argparse.ArgumentParser(prog="mslm")
    subs = p.add_subparsers(dest="cmd", required=True)

    # train
    t = subs.add_parser("train")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--batch_size", type=int, default=32)
    t.add_argument("--checkpoint_interval", type=int, default=5)
    t.add_argument("--log_interval", type=int, default=2)

    # study
    s = subs.add_parser("study")
    s.add_argument("--n_trials", type=int, default=30)
    s.add_argument("--batch_size", type=int, default=32)

    args = p.parse_args()

    if args.cmd == "train":
        train_script.run(
            epochs=args.epochs,
            batch_size=args.batch_size,
            checkpoint_interval=args.checkpoint_interval,
            log_interval=args.log_interval,
        )
    elif args.cmd == "study":
        make_study.run(
            n_trials=args.n_trials,
            batch_size=args.batch_size,
        )

if __name__ == "__main__":
    main()
