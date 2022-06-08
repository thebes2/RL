import os
import subprocess
import tempfile

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from utils.Loader import load_agent


def load(run_name, env=None, config=None, override=False):
    try:
        return load_agent(
            os.path.join("configs", (env or "") + ".yaml"),
            run_name=run_name,
            ckpt_folder=os.path.join("checkpoints"),
            config=config,
            override=override,
        )
    except FileNotFoundError:  # TODO: fallback to json during migration
        return load_agent(
            os.path.join("configs", (env or "") + ".json"),
            run_name=run_name,
            ckpt_folder=os.path.join("checkpoints"),
            config=config,
            override=override,
        )


def get_runs(base_name, runs, epochs, env, cfg):
    hist = []
    for i in range(runs):
        print("Run {}:".format(i + 1))
        agent = load(base_name + "-" + str(i), config=cfg)
        hist.append(agent.train(epochs, t_max=1000, logging=False, display=False))


def compare_algos(base_name, runs, epochs, env, diff):
    histA, histB = [], []
    for i in range(runs):
        print("Run {}a:".format(i + 1))
        agent = load(base_name + "-a-" + str(i), env)
        histA.append(agent.train(epochs, t_max=1000, logging=False, display=False))

        print("Run {}b:".format(i + 1))
        agent = load(base_name + "-b-" + str(i), env, diff)
        histB.append(agent.train(epochs, t_max=1000, logging=False, display=False))
    return histA, histB


def plot_series(*args):
    n = len(args[0])
    for a in args:
        plt.plot(range(n), a)
    plt.show()
    # plt.savefig('uniform-vs-proportional.png')


def plot_runs(*args):
    """
    Accepts a list of lists for each run indicating the results of each run
    and plots median and IQR ranges for each collection of runs
    """
    medians = []
    for c in args:
        a = np.array(c)
        n = a.shape[1]
        med = []
        p25 = []
        p75 = []
        for i in range(n):
            aa = a[:, i]
            med.append(np.percentile(aa, 50))
            p25.append(np.percentile(aa, 25))
            p75.append(np.percentile(aa, 75))
        plt.fill_between(range(n), p25, p75, alpha=0.5)
        medians.append(med)
    plot_series(*medians)


def draw_matrix(mat):
    img = Image.fromarray(mat, "RGB")
    img.show()


def generate_video_from_rollout(agent, env, t_max=10000):
    # only works for matplotlib based displays
    with tempfile.TemporaryDirectory() as tmp:
        obs = agent.preprocess(env.reset())
        for i in range(t_max):
            a = agent.get_action(obs, mode="greedy")[0][0]
            obs, r, dn, info = env.step(agent.action_wrapper(a))
            obs = agent.preprocess(obs)
            env.render()
            plt.savefig(os.path.join(tmp, "frame-{}.png".format(i)))
            if dn:
                break

        cur = os.getcwd()
        os.chdir(tmp)
        subprocess.call(
            [
                "ffmpeg",
                "-framerate",
                "8",
                "-i",
                "frame-%d.png",
                "-r",
                "30",
                "-pix_fmt",
                "yuv420p",
                "export.mp4",
            ]
        )

        os.rename(os.path.join(tmp, "export.mp4"), os.path.join(cur, "export.mp4"))
        os.chdir(cur)
