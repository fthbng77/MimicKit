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
        self._ring_bounds = np.asarray(
            env_config.get("ring_bounds", [-1.0, 3.0, -1.5, 1.5]),
            dtype=np.float32)
        self._ring_rail_height = float(env_config.get("ring_rail_height", 0.8))
        self._ring_rail_asset = env_config.get(
            "ring_rail_asset", "data/assets/objects/ring_rail.xml")
        self._attack_body_names = env_config.get(
            "attack_bodies", ["right_hand", "left_hand", "right_foot", "left_foot"])
        self._target_body_names = env_config.get(
            "target_bodies", ["torso", "head", "pelvis"])
        self._hit_distance = float(env_config.get("hit_distance", 0.18))
        self._hit_cooldown = float(env_config.get("hit_cooldown", 0.25))
        self._hit_damage = float(env_config.get("hit_damage", 10.0))
        self._hit_reward = float(env_config.get("hit_reward", 1.0))
        self._max_health = float(env_config.get("max_health", 100.0))
        self._min_attack_speed = float(env_config.get("min_attack_speed", 0.5))
        self._min_approach_speed = float(env_config.get("min_approach_speed", 0.1))
        self._contact_force_threshold = float(
            env_config.get("contact_force_threshold", 1.0))
        self._punch_demo_enabled = bool(env_config.get("punch_demo", False))
        self._punch_motion_file = env_config.get(
            "punch_motion_file", "data/motions/humanoid/humanoid_punch.pkl")
        self._punch_demo_period = float(env_config.get("punch_demo_period", 3.0))
        self._punch_demo_yaw = float(env_config.get("punch_demo_yaw", -0.638))
        self._punch_active_start = float(env_config.get("punch_active_start", 0.70))
        self._punch_active_end = float(env_config.get("punch_active_end", 1.30))
        self._opponent_attack_enabled = bool(
            env_config.get("opponent_attack", False))
        self._opponent_punch_period = float(
            env_config.get("opponent_punch_period", self._punch_demo_period))
        self._opponent_punch_phase = float(
            env_config.get("opponent_punch_phase", 0.0))
        self._opponent_punch_yaw = float(
            env_config.get("opponent_punch_yaw", self._punch_demo_yaw + np.pi))
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
        self._build_ring(env_id)
        return

    def _build_ring(self, env_id):
        xmin, xmax, ymin, ymax = self._ring_bounds
        z = self._ring_rail_height
        rails = [
            ([0.5 * (xmin + xmax), ymin, z], [0.0, 0.0, 0.0, 1.0]),
            ([0.5 * (xmin + xmax), ymax, z], [0.0, 0.0, 0.0, 1.0]),
            ([xmin, 0.5 * (ymin + ymax), z],
             [0.0, 0.0, 0.7071068, 0.7071068]),
            ([xmax, 0.5 * (ymin + ymax), z],
             [0.0, 0.0, 0.7071068, 0.7071068]),
        ]
        for rail_pos, rail_rot in rails:
            self._engine.create_obj(
                env_id=env_id,
                obj_type=engine.ObjType.rigid,
                asset_file=self._ring_rail_asset,
                name="ring_rail",
                fix_root=True,
                start_pos=np.asarray(rail_pos, dtype=np.float32),
                start_rot=np.asarray(rail_rot, dtype=np.float32),
                color=np.array([0.85, 0.1, 0.1], dtype=np.float32),
            )
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

        opponent_attack_body_ids = []
        for body_name in self._attack_body_names:
            body_id = self._engine.find_obj_body_id(opponent_id, body_name)
            assert body_id != -1
            opponent_attack_body_ids.append(body_id)
        fighter_target_body_ids = []
        for body_name in self._target_body_names:
            body_id = self._engine.find_obj_body_id(fighter_id, body_name)
            assert body_id != -1
            fighter_target_body_ids.append(body_id)

        self._opponent_attack_body_ids = torch.tensor(
            opponent_attack_body_ids, device=self._device, dtype=torch.long)
        self._opponent_target_body_ids = torch.tensor(
            fighter_target_body_ids, device=self._device, dtype=torch.long)

        num_envs = self.get_num_envs()
        self._opponent_health = torch.full(
            [num_envs], self._max_health, device=self._device, dtype=torch.float)
        self._fighter_health = torch.full(
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
        self._fighter_stagger_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._prev_attack_pos = torch.zeros(
            [num_envs, len(self._attack_body_ids), 3],
            device=self._device, dtype=torch.float)
        self._attack_speed_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._approach_speed_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._attack_contact_force_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._target_contact_force_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._knocked_down = torch.zeros(
            [num_envs], device=self._device, dtype=torch.bool)
        self._opponent_hit_cooldown_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._opponent_hit_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.bool)
        self._opponent_attack_active_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.bool)
        self._opponent_punch_hit_consumed = torch.zeros(
            [num_envs], device=self._device, dtype=torch.bool)
        self._opponent_prev_attack_pos = torch.zeros(
            [num_envs, len(self._opponent_attack_body_ids), 3],
            device=self._device, dtype=torch.float)
        self._opponent_attack_speed_buf = torch.zeros(
            [num_envs], device=self._device, dtype=torch.float)
        self._opponent_approach_speed_buf = torch.zeros(
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

        if self._opponent_attack_enabled:
            self._opponent_punch_motion_lib = motion_lib.MotionLib(
                motion_file=self._punch_motion_file,
                kin_char_model=self._kin_char_model,
                device=self._device)
            self._opponent_punch_motion_ids = torch.zeros(
                [num_envs], device=self._device, dtype=torch.long)
            self._opponent_punch_time_buf = torch.zeros(
                [num_envs], device=self._device, dtype=torch.float)
            self._opponent_punch_prev_time_buf = torch.zeros(
                [num_envs], device=self._device, dtype=torch.float)

            yaw_axis = torch.zeros([num_envs, 3], device=self._device)
            yaw_axis[:, 2] = 1.0
            yaw = torch.full([num_envs], self._opponent_punch_yaw,
                             device=self._device)
            self._opponent_punch_yaw_quat = torch_util.axis_angle_to_quat(
                yaw_axis, yaw)
            first_time = torch.zeros([num_envs], device=self._device)
            first_frame = self._opponent_punch_motion_lib.calc_motion_frame(
                self._opponent_punch_motion_ids, first_time)
            first_root_pos = torch_util.quat_rotate(
                self._opponent_punch_yaw_quat, first_frame[0])
            opponent_start = torch.as_tensor(
                self._opponent_start_pos, device=self._device)
            self._opponent_punch_root_offset = opponent_start.unsqueeze(0) - first_root_pos
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            self._reset_opponent(env_ids)
            self._opponent_health[env_ids] = self._max_health
            self._fighter_health[env_ids] = self._max_health
            self._hit_cooldown_buf[env_ids] = 0.0
            self._hit_buf[env_ids] = False
            self._attack_active_buf[env_ids] = False
            self._attack_hit_consumed[env_ids] = False
            self._stagger_buf[env_ids] = 0.0
            self._fighter_stagger_buf[env_ids] = 0.0
            self._attack_speed_buf[env_ids] = 0.0
            self._approach_speed_buf[env_ids] = 0.0
            self._attack_contact_force_buf[env_ids] = 0.0
            self._target_contact_force_buf[env_ids] = 0.0
            self._knocked_down[env_ids] = False
            fighter_body_pos = self._engine.get_body_pos(self._get_char_id())
            self._prev_attack_pos[env_ids] = fighter_body_pos[
                env_ids][:, self._attack_body_ids, :]
            opponent_body_pos = self._engine.get_body_pos(self._get_opponent_id())
            self._opponent_prev_attack_pos[env_ids] = opponent_body_pos[
                env_ids][:, self._opponent_attack_body_ids, :]
            self._opponent_hit_cooldown_buf[env_ids] = 0.0
            self._opponent_hit_buf[env_ids] = False
            self._opponent_attack_active_buf[env_ids] = False
            self._opponent_punch_hit_consumed[env_ids] = False
            self._opponent_attack_speed_buf[env_ids] = 0.0
            self._opponent_approach_speed_buf[env_ids] = 0.0
            if self._punch_demo_enabled:
                self._punch_time_buf[env_ids] = 0.0
                self._punch_prev_time_buf[env_ids] = 0.0
            if self._opponent_attack_enabled:
                self._opponent_punch_time_buf[env_ids] = self._opponent_punch_phase
                self._opponent_punch_prev_time_buf[env_ids] = 0.0
        return

    def _update_misc(self):
        super()._update_misc()
        was_staggered = self._stagger_buf > 0.0
        self._stagger_buf[:] = torch.clamp_min(
            self._stagger_buf - self._engine.get_timestep(), 0.0)
        was_fighter_staggered = self._fighter_stagger_buf > 0.0
        self._fighter_stagger_buf[:] = torch.clamp_min(
            self._fighter_stagger_buf - self._engine.get_timestep(), 0.0)
        if torch.any(torch.logical_and(was_staggered, self._stagger_buf <= 0.0)):
            self._set_character_color(self._get_opponent_id(),
                                      self._get_opponent_color())
        if torch.any(torch.logical_and(
                was_fighter_staggered, self._fighter_stagger_buf <= 0.0)):
            self._set_character_color(self._get_char_id(),
                                      self._get_char_color())
        if self._punch_demo_enabled:
            self._update_punch_demo()
        if self._opponent_attack_enabled:
            self._update_opponent_punch()
        self._enforce_ring_bounds()
        self._update_hits()
        self._update_opponent_hits()
        return

    def _enforce_ring_bounds(self):
        xmin, xmax, ymin, ymax = self._ring_bounds
        for obj_id in [self._get_char_id(), self._get_opponent_id()]:
            root_pos = self._engine.get_root_pos(obj_id)
            clamped_pos = root_pos.clone()
            clamped_pos[:, 0] = torch.clamp(clamped_pos[:, 0], xmin, xmax)
            clamped_pos[:, 1] = torch.clamp(clamped_pos[:, 1], ymin, ymax)
            changed = torch.any(torch.abs(clamped_pos - root_pos) > 1e-6, dim=-1)
            if not torch.any(changed):
                continue

            root_vel = self._engine.get_root_vel(obj_id)[changed].clone()
            root_vel[:, 0] = torch.where(
                ((clamped_pos[changed, 0] <= xmin + 1e-6) & (root_vel[:, 0] < 0.0)) |
                ((clamped_pos[changed, 0] >= xmax - 1e-6) & (root_vel[:, 0] > 0.0)),
                torch.zeros_like(root_vel[:, 0]), root_vel[:, 0])
            root_vel[:, 1] = torch.where(
                ((clamped_pos[changed, 1] <= ymin + 1e-6) & (root_vel[:, 1] < 0.0)) |
                ((clamped_pos[changed, 1] >= ymax - 1e-6) & (root_vel[:, 1] > 0.0)),
                torch.zeros_like(root_vel[:, 1]), root_vel[:, 1])
            changed_ids = changed.nonzero(as_tuple=False).flatten()
            dof_pos = self._engine.get_dof_pos(obj_id)[changed]
            root_rot = self._engine.get_root_rot(obj_id)[changed]
            joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
            body_pos, body_rot = self._kin_char_model.forward_kinematics(
                clamped_pos[changed], root_rot, joint_rot)
            self._engine.set_root_pos(changed_ids, obj_id, clamped_pos[changed])
            self._engine.set_root_vel(changed_ids, obj_id, root_vel)
            self._engine.set_body_pos(changed_ids, obj_id, body_pos)
            self._engine.set_body_rot(changed_ids, obj_id, body_rot)
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

    def _update_opponent_punch(self):
        dt = self._engine.get_timestep()
        prev_time = self._opponent_punch_time_buf.clone()
        self._opponent_punch_time_buf[:] = torch.remainder(
            self._opponent_punch_time_buf + dt, self._opponent_punch_period)
        wrapped = self._opponent_punch_time_buf < prev_time
        self._opponent_punch_hit_consumed[wrapped] = False
        self._opponent_punch_prev_time_buf[:] = prev_time
        self._opponent_attack_active_buf[:] = torch.logical_and(
            self._opponent_punch_time_buf >= self._punch_active_start,
            self._opponent_punch_time_buf <= self._punch_active_end)

        opponent_id = self._get_opponent_id()
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = \
            self._opponent_punch_motion_lib.calc_motion_frame(
                self._opponent_punch_motion_ids,
                self._opponent_punch_time_buf)
        root_pos = torch_util.quat_rotate(
            self._opponent_punch_yaw_quat, root_pos)
        root_pos += self._opponent_punch_root_offset
        root_rot = torch_util.quat_mul(
            self._opponent_punch_yaw_quat, root_rot)
        root_vel = torch_util.quat_rotate(self._opponent_punch_yaw_quat, root_vel)
        root_ang_vel = torch_util.quat_rotate(
            self._opponent_punch_yaw_quat, root_ang_vel)
        dof_pos = self._opponent_punch_motion_lib.joint_rot_to_dof(joint_rot)
        body_pos, body_rot = self._kin_char_model.forward_kinematics(
            root_pos, root_rot, joint_rot)

        self._engine.set_root_pos(None, opponent_id, root_pos)
        self._engine.set_root_rot(None, opponent_id, root_rot)
        self._engine.set_root_vel(None, opponent_id, root_vel)
        self._engine.set_root_ang_vel(None, opponent_id, root_ang_vel)
        self._engine.set_dof_pos(None, opponent_id, dof_pos)
        self._engine.set_dof_vel(None, opponent_id, dof_vel)
        self._engine.set_body_pos(None, opponent_id, body_pos)
        self._engine.set_body_rot(None, opponent_id, body_rot)
        return

    def _update_hits(self):
        dt = self._engine.get_timestep()
        self._hit_cooldown_buf[:] = torch.clamp_min(
            self._hit_cooldown_buf - dt, 0.0)

        fighter_pos = self._engine.get_body_pos(self._get_char_id())
        opponent_pos = self._engine.get_body_pos(self._get_opponent_id())
        attack_pos = fighter_pos[:, self._attack_body_ids, :]
        target_pos = opponent_pos[:, self._target_body_ids, :]

        attack_vel = (attack_pos - self._prev_attack_pos) / dt
        self._prev_attack_pos[:] = attack_pos
        attack_speed = torch.linalg.norm(attack_vel, dim=-1)
        self._attack_speed_buf[:] = torch.amax(attack_speed, dim=-1)

        delta = target_pos.unsqueeze(-3) - attack_pos.unsqueeze(-2)
        distance = torch.linalg.norm(delta, dim=-1).clamp_min(1e-6)
        direction = delta / distance.unsqueeze(-1)
        approach_speed = torch.sum(attack_vel.unsqueeze(-2) * direction, dim=-1)
        self._approach_speed_buf[:] = torch.amax(approach_speed, dim=(-2, -1))

        attack_contact = self._engine.get_contact_forces(self._get_char_id())
        target_contact = self._engine.get_contact_forces(self._get_opponent_id())
        attack_contact = torch.linalg.norm(
            attack_contact[:, self._attack_body_ids, :], dim=-1)
        target_contact = torch.linalg.norm(
            target_contact[:, self._target_body_ids, :], dim=-1)
        self._attack_contact_force_buf[:] = torch.amax(attack_contact, dim=-1)
        self._target_contact_force_buf[:] = torch.amax(target_contact, dim=-1)

        pair_dist = attack_pos.unsqueeze(-2) - target_pos.unsqueeze(-3)
        pair_dist = torch.linalg.norm(pair_dist, dim=-1)
        close_enough = torch.amin(pair_dist, dim=(-2, -1)) <= self._hit_distance
        can_hit = torch.logical_and(
            self._hit_cooldown_buf <= 0.0, self._attack_active_buf)
        can_hit = torch.logical_and(can_hit, ~self._attack_hit_consumed)
        moving_fast_enough = torch.logical_and(
            self._attack_speed_buf >= self._min_attack_speed,
            self._approach_speed_buf >= self._min_approach_speed)
        contact_confirmed = torch.logical_or(
            self._attack_contact_force_buf >= self._contact_force_threshold,
            self._target_contact_force_buf >= self._contact_force_threshold)
        impact_confirmed = torch.logical_or(moving_fast_enough, contact_confirmed)
        self._hit_buf[:] = torch.logical_and(
            torch.logical_and(close_enough, can_hit), impact_confirmed)

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
            self._set_character_color(self._get_opponent_id(),
                                      [1.0, 0.85, 0.05])
            knocked_down = self._opponent_health[hit_ids] <= 0.0
            knocked_down_ids = hit_ids[knocked_down]
            if len(knocked_down_ids) > 0:
                self._knocked_down[knocked_down_ids] = True
                self._apply_knockdown(knocked_down_ids)
            Logger.print("HIT! opponent health: {:.1f}".format(
                float(self._opponent_health[hit_ids[0]].item())))
        return

    def _update_opponent_hits(self):
        if not self._opponent_attack_enabled:
            self._opponent_hit_buf[:] = False
            return

        dt = self._engine.get_timestep()
        self._opponent_hit_cooldown_buf[:] = torch.clamp_min(
            self._opponent_hit_cooldown_buf - dt, 0.0)
        fighter_pos = self._engine.get_body_pos(self._get_char_id())
        opponent_pos = self._engine.get_body_pos(self._get_opponent_id())
        attack_pos = opponent_pos[:, self._opponent_attack_body_ids, :]
        target_pos = fighter_pos[:, self._opponent_target_body_ids, :]
        attack_vel = (attack_pos - self._opponent_prev_attack_pos) / dt
        self._opponent_prev_attack_pos[:] = attack_pos
        attack_speed = torch.linalg.norm(attack_vel, dim=-1)
        self._opponent_attack_speed_buf[:] = torch.amax(attack_speed, dim=-1)

        delta = target_pos.unsqueeze(-3) - attack_pos.unsqueeze(-2)
        distance = torch.linalg.norm(delta, dim=-1).clamp_min(1e-6)
        direction = delta / distance.unsqueeze(-1)
        approach_speed = torch.sum(attack_vel.unsqueeze(-2) * direction, dim=-1)
        self._opponent_approach_speed_buf[:] = torch.amax(
            approach_speed, dim=(-2, -1))
        pair_dist = attack_pos.unsqueeze(-2) - target_pos.unsqueeze(-3)
        pair_dist = torch.linalg.norm(pair_dist, dim=-1)
        close_enough = torch.amin(pair_dist, dim=(-2, -1)) <= self._hit_distance
        moving_fast_enough = torch.logical_and(
            self._opponent_attack_speed_buf >= self._min_attack_speed,
            self._opponent_approach_speed_buf >= self._min_approach_speed)
        can_hit = torch.logical_and(
            self._opponent_hit_cooldown_buf <= 0.0,
            self._opponent_attack_active_buf)
        can_hit = torch.logical_and(can_hit, ~self._opponent_punch_hit_consumed)
        self._opponent_hit_buf[:] = torch.logical_and(
            torch.logical_and(close_enough, can_hit), moving_fast_enough)
        hit_ids = self._opponent_hit_buf.nonzero(as_tuple=False).flatten()
        if len(hit_ids) > 0:
            self._fighter_health[hit_ids] -= self._hit_damage
            self._opponent_hit_cooldown_buf[hit_ids] = self._hit_cooldown
            self._opponent_punch_hit_consumed[hit_ids] = True
            self._fighter_stagger_buf[hit_ids] = self._stagger_duration
            self._set_character_color(self._get_char_id(),
                                      [1.0, 0.85, 0.05])
            Logger.print("FIGHTER HIT! fighter health: {:.1f}".format(
                float(self._fighter_health[hit_ids[0]].item())))
        return

    def _apply_knockdown(self, env_ids):
        opponent_id = self._get_opponent_id()
        root_pos = self._engine.get_root_pos(opponent_id)[env_ids].clone()
        root_pos[:, 2] = 0.35
        axis = torch.zeros([len(env_ids), 3], device=self._device)
        axis[:, 1] = 1.0
        angle = torch.full(
            [len(env_ids)], np.pi * 0.5, device=self._device)
        root_rot = torch_util.axis_angle_to_quat(axis, angle)
        dof_pos = self._engine.get_dof_pos(opponent_id)[env_ids]
        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        body_pos, body_rot = self._kin_char_model.forward_kinematics(
            root_pos, root_rot, joint_rot)

        self._engine.set_root_pos(env_ids, opponent_id, root_pos)
        self._engine.set_root_rot(env_ids, opponent_id, root_rot)
        self._engine.set_root_vel(env_ids, opponent_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, opponent_id, 0.0)
        self._engine.set_body_pos(env_ids, opponent_id, body_pos)
        self._engine.set_body_rot(env_ids, opponent_id, body_rot)
        self._set_character_color(self._get_opponent_id(), [0.35, 0.08, 0.08])
        return

    def _get_char_color(self):
        if self._engine.get_name() == "newton":
            return np.array([0.35, 0.45, 0.7])
        return np.array([0.5, 0.65, 0.95])

    def _set_character_color(self, obj_id, color):
        if not self._visualize:
            return
        if self._engine.get_name() != "newton":
            return

        num_bodies = self._engine.get_obj_num_bodies(obj_id)
        for body_id in range(num_bodies):
            self._engine.set_body_color(
                0, obj_id, body_id, np.asarray(color, dtype=np.float32))
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
        self._done_buf[self._fighter_health <= 0.0] = base_env.DoneFlags.FAIL.value
        return

    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)
        if env_ids is None:
            self._info["combat_hit"] = self._hit_buf
            self._info["opponent_health"] = self._opponent_health
            self._info["fighter_health"] = self._fighter_health
            self._info["attack_active"] = self._attack_active_buf
            self._info["opponent_stagger"] = self._stagger_buf
            self._info["attack_speed"] = self._attack_speed_buf
            self._info["approach_speed"] = self._approach_speed_buf
            self._info["attack_contact_force"] = self._attack_contact_force_buf
            self._info["target_contact_force"] = self._target_contact_force_buf
            self._info["opponent_knocked_down"] = self._knocked_down
            self._info["fighter_hit"] = self._opponent_hit_buf
            self._info["opponent_attack_active"] = self._opponent_attack_active_buf
            self._info["opponent_attack_speed"] = self._opponent_attack_speed_buf
        else:
            self._info["combat_hit"] = self._hit_buf[env_ids]
            self._info["opponent_health"] = self._opponent_health[env_ids]
            self._info["fighter_health"] = self._fighter_health[env_ids]
            self._info["attack_active"] = self._attack_active_buf[env_ids]
            self._info["opponent_stagger"] = self._stagger_buf[env_ids]
            self._info["attack_speed"] = self._attack_speed_buf[env_ids]
            self._info["approach_speed"] = self._approach_speed_buf[env_ids]
            self._info["attack_contact_force"] = self._attack_contact_force_buf[env_ids]
            self._info["target_contact_force"] = self._target_contact_force_buf[env_ids]
            self._info["opponent_knocked_down"] = self._knocked_down[env_ids]
            self._info["fighter_hit"] = self._opponent_hit_buf[env_ids]
            self._info["opponent_attack_active"] = self._opponent_attack_active_buf[env_ids]
            self._info["opponent_attack_speed"] = self._opponent_attack_speed_buf[env_ids]
        return

