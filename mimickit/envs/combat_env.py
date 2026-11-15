import numpy as np
import torch

import engines.engine as engine
import envs.char_env as char_env
import envs.base_env as base_env
import anim.motion_lib as motion_lib
import util.torch_util as torch_util
from util.logger import Logger


class CombatEnv(char_env.CharEnv):
    """Two-character sandbox for building the combat task incrementally.

    The first character is the learning character and uses the normal
    CharEnv action space. The second character is currently passive, but is
    fully simulated so that opponent observations and contact logic can be
    added without changing the scene layout later.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        self._opponent_start_pos = np.asarray(
            env_config.get("opponent_start_pos", [2.0, 0.0, 0.882416]),
            dtype=np.float32)
        self._opponent_start_rot = np.asarray(
            env_config.get("opponent_start_rot", [0.0, 0.0, 1.0, 0.0]),
            dtype=np.float32)
        self._attack_body_names = env_config.get(
            "attack_bodies", ["right_hand", "left_hand", "right_foot", "left_foot"])
        self._target_body_names = env_config.get(
            "target_bodies", ["torso", "head", "pelvis"])
        self._hit_distance = float(env_config.get("hit_distance", 0.18))
        self._hit_cooldown = float(env_config.get("hit_cooldown", 0.25))
        self._hit_damage = float(env_config.get("hit_damage", 10.0))
        self._hit_reward = float(env_config.get("hit_reward", 1.0))
        self._max_health = float(env_config.get("max_health", 100.0))
        self._punch_demo_enabled = bool(env_config.get("punch_demo", False))
        self._punch_motion_file = env_config.get(
            "punch_motion_file", "data/motions/humanoid/humanoid_punch.pkl")
        self._punch_demo_period = float(env_config.get("punch_demo_period", 3.0))
        self._punch_demo_yaw = float(env_config.get("punch_demo_yaw", -0.638))
        self._punch_active_start = float(env_config.get("punch_active_start", 0.70))
        self._punch_active_end = float(env_config.get("punch_active_end", 1.30))
        self._stagger_duration = float(env_config.get("stagger_duration", 0.35))
        self._stagger_speed = float(env_config.get("stagger_speed", 1.5))
        self._opponent_ids = []

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device,
                         visualize=visualize, record_video=record_video)
        return

    def _build_env(self, env_id, env_config):
        super()._build_env(env_id, env_config)

        opponent_id = self._engine.create_obj(
            env_id=env_id,
            obj_type=engine.ObjType.articulated,
            asset_file=env_config["char_file"],
            name="opponent",
            start_pos=self._opponent_start_pos,
            start_rot=self._opponent_start_rot,
            color=self._get_opponent_color(),
        )

        if env_id == 0:
            self._opponent_ids.append(opponent_id)
        else:
            assert self._opponent_ids[0] == opponent_id
        return

    def _get_opponent_id(self):
        return self._opponent_ids[0]

    def _get_opponent_color(self):
        if self._engine.get_name() == "newton":
            return np.array([0.7, 0.25, 0.25])
        return np.array([0.9, 0.35, 0.35])

    def _build_sim_tensors(self, env_config):
        super()._build_sim_tensors(env_config)

        opponent_body_names = env_config.get(
            "opponent_key_bodies", env_config.get("key_bodies", []))
        opponent_body_ids = []
        opponent_id = self._get_opponent_id()
        for body_name in opponent_body_names:
            body_id = self._engine.find_obj_body_id(opponent_id, body_name)
            assert body_id != -1
            opponent_body_ids.append(body_id)
        self._opponent_key_body_ids = torch.tensor(
            opponent_body_ids, device=self._device, dtype=torch.long)

        fighter_id = self._get_char_id()
        attack_body_ids = []
        for body_name in self._attack_body_names:
            body_id = self._engine.find_obj_body_id(fighter_id, body_name)
            assert body_id != -1
            attack_body_ids.append(body_id)
        target_body_ids = []
        opponent_id = self._get_opponent_id()
        for body_name in self._target_body_names:
            body_id = self._engine.find_obj_body_id(opponent_id, body_name)
            assert body_id != -1
            target_body_ids.append(body_id)

        self._attack_body_ids = torch.tensor(
            attack_body_ids, device=self._device, dtype=torch.long)
        self._target_body_ids = torch.tensor(
            target_body_ids, device=self._device, dtype=torch.long)

        num_envs = self.get_num_envs()
        self._opponent_health = torch.full(
            [num_envs], self._max_health, device=self._device, dtype=torch.float)
        self._hit_cooldown_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._hit_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.bool)
        self._attack_active_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.bool)
        self._attack_hit_consumed = torch.zeros(
            [num_envs], device=self._device, dtype=torch.bool)
        self._stagger_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)

        if self._punch_demo_enabled:
            self._punch_motion_lib = motion_lib.MotionLib(
                motion_file=self._punch_motion_file,
                kin_char_model=self._kin_char_model,
                device=self._device)
            self._punch_motion_ids = torch.zeros(
                [num_envs], device=self._device, dtype=torch.long)
            self._punch_time_buf = torch.zeros(
                [num_envs], device=self._device, dtype=torch.float)
            self._punch_prev_time_buf = torch.zeros(
                [num_envs], device=self._device, dtype=torch.float)

            yaw_axis = torch.zeros([num_envs, 3], device=self._device)
            yaw_axis[:, 2] = 1.0
            yaw = torch.full(
                [num_envs], self._punch_demo_yaw, device=self._device)
            self._punch_yaw_quat = torch_util.axis_angle_to_quat(yaw_axis, yaw)

            first_time = torch.zeros([num_envs], device=self._device)
            first_frame = self._punch_motion_lib.calc_motion_frame(
                self._punch_motion_ids, first_time)
            first_root_pos = torch_util.quat_rotate(
                self._punch_yaw_quat, first_frame[0])
            self._punch_root_offset = self._init_root_pos.unsqueeze(0) - first_root_pos
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            self._reset_opponent(env_ids)
            self._opponent_health[env_ids] = self._max_health
            self._hit_cooldown_buf[env_ids] = 0.0
            self._hit_buf[env_ids] = False
            self._attack_active_buf[env_ids] = False
            self._attack_hit_consumed[env_ids] = False
            self._stagger_buf[env_ids] = 0.0
            if self._punch_demo_enabled:
                self._punch_time_buf[env_ids] = 0.0
                self._punch_prev_time_buf[env_ids] = 0.0
        return

    def _update_misc(self):
        super()._update_misc()
        was_staggered = self._stagger_buf > 0.0
        self._stagger_buf[:] = torch.clamp_min(
            self._stagger_buf - self._engine.get_timestep(), 0.0)
        if torch.any(torch.logical_and(was_staggered, self._stagger_buf <= 0.0)):
            self._set_opponent_color([0.7, 0.25, 0.25])
        if self._punch_demo_enabled:
            self._update_punch_demo()
        self._update_hits()
        return

    def _update_punch_demo(self):
        dt = self._engine.get_timestep()
        prev_time = self._punch_time_buf.clone()
        self._punch_time_buf[:] = torch.remainder(
            self._punch_time_buf + dt, self._punch_demo_period)
        wrapped = self._punch_time_buf < prev_time
        self._attack_hit_consumed[wrapped] = False
        self._punch_prev_time_buf[:] = prev_time
        self._attack_active_buf[:] = torch.logical_and(
            self._punch_time_buf >= self._punch_active_start,
            self._punch_time_buf <= self._punch_active_end)

        fighter_id = self._get_char_id()
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = \
            self._punch_motion_lib.calc_motion_frame(
                self._punch_motion_ids, self._punch_time_buf)

        root_pos = torch_util.quat_rotate(self._punch_yaw_quat, root_pos)
        root_pos += self._punch_root_offset
        root_rot = torch_util.quat_mul(self._punch_yaw_quat, root_rot)
        root_vel = torch_util.quat_rotate(self._punch_yaw_quat, root_vel)
        root_ang_vel = torch_util.quat_rotate(self._punch_yaw_quat, root_ang_vel)
        dof_pos = self._punch_motion_lib.joint_rot_to_dof(joint_rot)
        body_pos, body_rot = self._kin_char_model.forward_kinematics(
            root_pos, root_rot, joint_rot)

        self._engine.set_root_pos(None, fighter_id, root_pos)
        self._engine.set_root_rot(None, fighter_id, root_rot)
        self._engine.set_root_vel(None, fighter_id, root_vel)
        self._engine.set_root_ang_vel(None, fighter_id, root_ang_vel)
        self._engine.set_dof_pos(None, fighter_id, dof_pos)
        self._engine.set_dof_vel(None, fighter_id, dof_vel)
        self._engine.set_body_pos(None, fighter_id, body_pos)
        self._engine.set_body_rot(None, fighter_id, body_rot)
        return

    def _update_hits(self):
        dt = self._engine.get_timestep()
        self._hit_cooldown_buf[:] = torch.clamp_min(
            self._hit_cooldown_buf - dt, 0.0)

        fighter_pos = self._engine.get_body_pos(self._get_char_id())
        opponent_pos = self._engine.get_body_pos(self._get_opponent_id())
        attack_pos = fighter_pos[:, self._attack_body_ids, :]
        target_pos = opponent_pos[:, self._target_body_ids, :]

        pair_dist = attack_pos.unsqueeze(-2) - target_pos.unsqueeze(-3)
        pair_dist = torch.linalg.norm(pair_dist, dim=-1)
        close_enough = torch.amin(pair_dist, dim=(-2, -1)) <= self._hit_distance
        can_hit = torch.logical_and(
            self._hit_cooldown_buf <= 0.0, self._attack_active_buf)
        can_hit = torch.logical_and(can_hit, ~self._attack_hit_consumed)
        self._hit_buf[:] = torch.logical_and(close_enough, can_hit)

        hit_ids = self._hit_buf.nonzero(as_tuple=False).flatten()
        if len(hit_ids) > 0:
            self._opponent_health[hit_ids] -= self._hit_damage
            self._hit_cooldown_buf[hit_ids] = self._hit_cooldown
            self._attack_hit_consumed[hit_ids] = True
            self._stagger_buf[hit_ids] = self._stagger_duration

            opponent_id = self._get_opponent_id()
            opponent_vel = self._engine.get_root_vel(opponent_id)[hit_ids].clone()
            opponent_vel[:, 0] += self._stagger_speed
            self._engine.set_root_vel(hit_ids, opponent_id, opponent_vel)
            self._set_opponent_color([1.0, 0.85, 0.05])
            Logger.print("HIT! opponent health: {:.1f}".format(
                float(self._opponent_health[hit_ids[0]].item())))
        return

    def _set_opponent_color(self, color):
        if not self._visualize:
            return
        if self._engine.get_name() != "newton":
            return

        opponent_id = self._get_opponent_id()
        num_bodies = self._engine.get_obj_num_bodies(opponent_id)
        for body_id in range(num_bodies):
            self._engine.set_body_color(
                0, opponent_id, body_id, np.asarray(color, dtype=np.float32))
        return

    def _reset_opponent(self, env_ids):
        opponent_id = self._get_opponent_id()
        n = len(env_ids)
        root_pos = torch.as_tensor(self._opponent_start_pos,
                                   device=self._device).expand(n, -1)
        root_rot = torch.as_tensor(self._opponent_start_rot,
                                   device=self._device).expand(n, -1)
        dof_pos = self._init_dof_pos.expand(n, -1)

        self._engine.set_root_pos(env_ids, opponent_id, root_pos)
        self._engine.set_root_rot(env_ids, opponent_id, root_rot)
        self._engine.set_root_vel(env_ids, opponent_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, opponent_id, 0.0)
        self._engine.set_dof_pos(env_ids, opponent_id, dof_pos)
        self._engine.set_dof_vel(env_ids, opponent_id, 0.0)
        self._engine.set_body_vel(env_ids, opponent_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, opponent_id, 0.0)

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        body_pos, body_rot = self._kin_char_model.forward_kinematics(
            root_pos, root_rot, joint_rot)
        self._engine.set_body_pos(env_ids, opponent_id, body_pos)
        self._engine.set_body_rot(env_ids, opponent_id, body_rot)
        return

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)

        fighter_id = self._get_char_id()
        opponent_id = self._get_opponent_id()
        fighter_pos = self._engine.get_root_pos(fighter_id)
        fighter_vel = self._engine.get_root_vel(fighter_id)
        opponent_pos = self._engine.get_root_pos(opponent_id)
        opponent_vel = self._engine.get_root_vel(opponent_id)

        if env_ids is not None:
            fighter_pos = fighter_pos[env_ids]
            fighter_vel = fighter_vel[env_ids]
            opponent_pos = opponent_pos[env_ids]
            opponent_vel = opponent_vel[env_ids]

        relative_pos = opponent_pos - fighter_pos
        relative_vel = opponent_vel - fighter_vel
        combat_obs = torch.cat([relative_pos, relative_vel], dim=-1)
        return torch.cat([obs, combat_obs], dim=-1)

    def _update_reward(self):
        super()._update_reward()
        self._reward_buf += self._hit_reward * self._hit_buf.float()
        return

    def _update_done(self):
        super()._update_done()
        self._done_buf[self._opponent_health <= 0.0] = base_env.DoneFlags.SUCC.value
        return

    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)
        if env_ids is None:
            self._info["combat_hit"] = self._hit_buf
            self._info["opponent_health"] = self._opponent_health
            self._info["attack_active"] = self._attack_active_buf
            self._info["opponent_stagger"] = self._stagger_buf
        else:
            self._info["combat_hit"] = self._hit_buf[env_ids]
            self._info["opponent_health"] = self._opponent_health[env_ids]
            self._info["attack_active"] = self._attack_active_buf[env_ids]
            self._info["opponent_stagger"] = self._stagger_buf[env_ids]
        return

