import argparse
from typing import Optional

from utils.logger import logger
from utils.utils import load


def main(run_name: str, t_max: Optional[int]):
    agent = load(run_name=run_name, env=None)
    reward = agent.collect_rollout(t_max=t_max, display=True)

    logger.success("Total reward: ", reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str)
    parser.add_argument(
        "--t_max",
        type=int,
        default=None,
        help="override default time horizon used during training",
    )

    args = parser.parse_args()
    main(args.run_name, args.t_max)
