"""Record one deterministic rollout of the trained balance policy."""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))

from envs.env_builder import build_env
from learning.agent_builder import build_agent
from learning.base_agent import AgentMode
from util.display import ensure_virtual_display
from util.video import Video


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="output/balance_humanoid_continue/model.pt")
parser.add_argument("--env", default="data/envs/balance_humanoid_env.yaml")
parser.add_argument("--agent", default="data/agents/balance_humanoid_ppo_agent.yaml")
parser.add_argument("--out-dir", default="output/balance_humanoid_continue")
parser.add_argument("--seed", type=int, default=11)
args = parser.parse_args()

torch.manual_seed(args.seed)
ensure_virtual_display()
env = build_env(args.env,
                "data/engines/newton_engine.yaml", 1, "cuda:0", True)
agent = build_agent(args.agent, env, "cuda:0")
agent.load(args.model)
agent.set_mode(AgentMode.TEST)
obs, info = env.reset()
frames = []
for step in range(90):
    with torch.no_grad():
        action, _ = agent._decide_action(obs, info)
    obs, reward, done, info = env.step(action)
    if step % 3 == 0:
        frames.append(env._engine._viewer.get_frame(render_ui=False).numpy().copy())
    if done[0]:
        break

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
video = Video(10)
for frame in frames:
    video.add_frame(frame)
video.save(str(out_dir / "policy_rollout.mp4"))
print("rollout_steps={}".format(step + 1))
print("done={}".format(int(done[0])))
print("root_up_z={:.4f}".format(float(info["root_up_z"][0])))
