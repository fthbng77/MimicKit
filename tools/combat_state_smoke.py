"""Run from repo root: .venv-py310/bin/python tools/combat_state_smoke.py [--visualize]."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'mimickit'))
import torch
from envs.env_builder import build_env
from envs.combat_env import CombatState as S

parser = argparse.ArgumentParser()
parser.add_argument('--visualize', action='store_true')
args = parser.parse_args()
if args.visualize:
    from util.display import ensure_virtual_display
    ensure_virtual_display()
torch.manual_seed(7)
env = build_env('data/envs/combat_humanoid_env.yaml',
                'data/engines/newton_engine.yaml', 1, 'cuda:0', args.visualize)
env.reset()
action = torch.zeros((1, env.get_action_space().shape[0]), device='cuda:0')
seen = [set(), set()]
transitions = [set(), set()]
previous = [S.IDLE, S.IDLE]
hits = [0, 0]
frames = []
combo_seen = [False, False]
max_support_slip = 0.0
max_raw_slip = 0.0
support_samples = 0
for step in range(900):
    obs, reward, done, info = env.step(action)
    assert torch.isfinite(obs).all()
    states = env._combat_state[0].tolist()
    assert states != [S.ATTACK, S.ATTACK]
    for side, state in enumerate(states):
        seen[side].add(state)
        if state != previous[side]:
            transitions[side].add((previous[side], state))
            print(f'{step}: {side}: {S(previous[side]).name} -> {S(state).name}', flush=True)
        previous[side] = state
    hits[0] += int(info['combat_hit'][0])
    hits[1] += int(info['fighter_hit'][0])
    for side in range(2):
        combo_seen[side] |= bool(env._combo_second[0, side])
    max_support_slip = max(max_support_slip, float(info['support_foot_slip'].max()))
    max_raw_slip = max(max_raw_slip, float(info['support_raw_slip'].max()))
    support_samples += int((env._support_foot >= 0).sum())
    for obj in (env._get_char_id(), env._get_opponent_id()):
        pos = env._engine.get_root_pos(obj)[0]
        xmin, xmax, ymin, ymax = env._ring_bounds
        assert xmin <= pos[0] <= xmax and ymin <= pos[1] <= ymax
    if args.visualize and step % 3 == 0:
        from PIL import Image, ImageDraw
        import numpy as np
        frame = Image.fromarray(env._engine._viewer.get_frame(render_ui=False).numpy().copy())
        draw = ImageDraw.Draw(frame)
        draw.rectangle((10, 10, 700, 82), fill=(18, 22, 30))
        for side, label in enumerate(('BLUE', 'RED')):
            hp = env._fighter_health if side == 0 else env._opponent_health
            draw.text((20, 20 + side * 28),
                      f'{label}  HP {float(hp[0]):.1f}  {S(states[side]).name}  '
                      f'Stamina {float(env._stamina[0, side]):.2f}  '
                      f'Combo {bool(env._combo[0, side])}',
                      fill=(120, 170, 255) if side == 0 else (255, 130, 130))
        frames.append(np.asarray(frame).copy())
for side in range(2):
    assert {S.IDLE, S.APPROACH, S.ATTACK, S.RECOVER} <= seen[side], seen
    assert (S.ATTACK, S.RECOVER) in transitions[side]
assert any(S.BLOCK in states for states in seen), seen
assert any(S.SLIP in states for states in seen), seen
assert any(S.BACKSTEP in states for states in seen), seen
assert min(hits) > 0, hits
assert all(combo_seen), combo_seen
assert support_samples > 50, support_samples
assert max_support_slip < 0.03, max_support_slip
print('900 steps passed; hits:', hits, flush=True)
print(f'Support footfall samples: {support_samples}; raw/corrected max slip: {max_raw_slip:.4f}/{max_support_slip:.4f} m', flush=True)
if frames:
    from util.video import Video
    out = Path('output/combat_humanoid/state_machine.mp4')
    out.parent.mkdir(parents=True, exist_ok=True)
    video = Video(10)
    for frame in frames:
        video.add_frame(frame)
    video.save(str(out))
    from PIL import Image
    for i in (1, 20, 40, 60, 80):
        Image.fromarray(frames[i]).save(str(out.with_name(f'state_{i:02d}.png')))
# Exact-boundary outward velocity, both axes and both characters.
ids = torch.tensor([0], device='cuda:0')
# Exercise stun independently of the defense decisions in the free fight.
for side in range(2):
    env.reset()
    env._receive_hit(ids, side)
    assert env._combat_state[0, side] == S.HIT_STUN
    for _ in range(12):
        env.step(action)
    assert env._combat_state[0, side] == S.RETREAT
assert ((env._stamina >= 0) & (env._stamina <= 1)).all()
for obj in (env._get_char_id(), env._get_opponent_id()):
    pos = env._engine.get_root_pos(obj).clone()
    pos[0, :2] = pos.new_tensor([xmin, ymin])
    env._engine.set_root_pos(ids, obj, pos)
    vel = torch.zeros_like(pos)
    vel[0, :2] = -2
    env._engine.set_root_vel(ids, obj, vel)
    env._enforce_ring_bounds()
    assert (env._engine.get_root_vel(obj)[0, :2] == 0).all()
# Each side can be knocked down; winner stops, episode ends, reset clears state.
for side in range(2):
    env.reset()
    health = env._fighter_health if side == 0 else env._opponent_health
    health[0] = 0
    env._receive_hit(ids, side)
    env.step(action)
    assert env._combat_state[0, side] == S.DOWN
    assert env._combat_state[0, 1-side] == S.IDLE
    assert not env._attack_active_buf.any() and not env._opponent_attack_active_buf.any()
    assert env._done_buf[0] != 0
    obj = env._get_char_id() if side == 0 else env._get_opponent_id()
    assert (env._engine.get_root_vel(obj) == 0).all()
env.reset()
assert (env._combat_state == S.IDLE).all()
print('Ring, bilateral knockdown, reset passed', flush=True)
