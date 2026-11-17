from enum import IntEnum

import numpy as np
import torch

import engines.engine as engine
import envs.char_env as char_env
import envs.base_env as base_env
import anim.motion_lib as motion_lib
import util.torch_util as torch_util
import util.camera as camera
from envs.boxing_controller import BoxingReferenceController
from util.logger import Logger


class CombatState(IntEnum):
    IDLE = 0
    APPROACH = 1
    ATTACK = 2
    RECOVER = 3
    RETREAT = 4
    HIT_STUN = 5
    DOWN = 6


class CombatEnv(char_env.CharEnv):
    """Two-character combat sandbox with independently timed demo controllers."""

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        self._opponent_start_pos = np.asarray(
            env_config.get("opponent_start_pos", [2.0, 0.0, 0.882416]),
            dtype=np.float32)
        self._opponent_start_rot = np.asarray(
            env_config.get("opponent_start_rot", [0.0, 0.0, 1.0, 0.0]),
            dtype=np.float32)
        self._ring_bounds = np.asarray(
            env_config.get("ring_bounds", [-1.0, 3.0, -2.0, 2.0]),
            dtype=np.float32)
        self._ring_floor_height = float(env_config.get("ring_floor_height", 0.36))
        self._ring_asset = env_config.get("ring_asset", "data/assets/objects/boxing_ring.xml")
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
        self._punch_demo_yaw = float(env_config.get("punch_demo_yaw", -0.638))
        self._punch_active_start = float(env_config.get("punch_active_start", 0.70))
        self._punch_active_end = float(env_config.get("punch_active_end", 1.30))
        self._opponent_attack_enabled = bool(
            env_config.get("opponent_attack", False))
        self._stagger_duration = float(env_config.get("stagger_duration", 0.35))
        self._stagger_speed = float(env_config.get("stagger_speed", 1.5))
        self._attack_duration = float(env_config.get("attack_duration", 1.5))
        self._recover_range = env_config.get("recover_duration", [1.0, 1.5])
        self._attack_cooldown_range = env_config.get("attack_cooldown", [1.5, 3.0])
        self._attack_range = float(env_config.get("attack_range", 1.15))
        self._approach_speed = float(env_config.get("approach_speed", 0.65))
        self._retreat_duration = float(env_config.get("retreat_duration", 0.5))
        self._boxing_reference = BoxingReferenceController(
            attack_range=self._attack_range,
            preferred_distance=float(env_config.get("preferred_distance", 1.0)),
            retreat_distance=float(env_config.get("retreat_distance", 0.72)),
            guard_height=float(env_config.get("guard_height", 1.05)))
        self._reference_control_enabled = bool(env_config.get("reference_control", False))
        self._residual_scale = float(env_config.get("residual_scale", 0.25))
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
        # Asset has a 4 m square fighting area, centered on its local origin.
        if not np.allclose([xmax - xmin, ymax - ymin], [4.0, 4.0]):
            raise ValueError("Boxing ring asset requires 4 x 4 metre ring_bounds")
        self._engine.create_obj(
            env_id=env_id,
            obj_type=engine.ObjType.rigid,
            asset_file=self._ring_asset,
            name="boxing_ring",
            fix_root=True,
            start_pos=np.asarray([0.5 * (xmin + xmax), 0.5 * (ymin + ymax),
                                  self._ring_floor_height], dtype=np.float32),
        )
        return

    def _build_camera(self, env_config):
        xmin, xmax, ymin, ymax = self._ring_bounds
        center = np.array([0.5 * (xmin + xmax), 0.5 * (ymin + ymax),
                           self._ring_floor_height + 0.45])
        self._camera = camera.Camera(
            mode=camera.CameraMode[env_config.get("camera_mode", "still")],
            engine=self._engine,
            pos=center + np.array([4.8, -7.4, 4.8]),
            target=center,
            track_obj_id=self._get_char_id())

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

        def joint_dof(name):
            joint_id = next(i for i, joint in enumerate(self._kin_char_model._joints)
                            if joint is not None and joint.name == name)
            return self._kin_char_model.get_joint_dof_idx(joint_id)
        self._reference_joint_ids = (
            (joint_dof("right_shoulder"), joint_dof("right_shoulder") + 1,
             joint_dof("right_elbow")),
            (joint_dof("left_shoulder"), joint_dof("left_shoulder") + 1,
             joint_dof("left_elbow")))

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

        self._combat_state = torch.zeros((num_envs, 2), device=self._device,
                                         dtype=torch.long)
        self._state_timer = torch.zeros((num_envs, 2), device=self._device)
        self._attack_cooldown = torch.zeros_like(self._state_timer)
        self._recover_timer = torch.zeros_like(self._state_timer)
        self._attack_variant = torch.zeros((num_envs, 2), device=self._device, dtype=torch.long)
        self._combat_pos = torch.zeros((num_envs, 2, 3), device=self._device)

        if self._punch_demo_enabled:
            self._punch_motion_lib = motion_lib.MotionLib(
                motion_file=self._punch_motion_file,
                kin_char_model=self._kin_char_model,
                device=self._device)
            self._punch_motion_ids = torch.zeros(
                [num_envs], device=self._device, dtype=torch.long)
        if self._opponent_attack_enabled:
            self._opponent_punch_motion_lib = motion_lib.MotionLib(
                motion_file=self._punch_motion_file,
                kin_char_model=self._kin_char_model,
                device=self._device)
            self._opponent_punch_motion_ids = torch.zeros(
                [num_envs], device=self._device, dtype=torch.long)
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            self._reset_opponent(env_ids)
            self._combat_state[env_ids] = CombatState.IDLE
            self._state_timer[env_ids] = 0.0
            self._recover_timer[env_ids] = 0.0
            self._attack_variant[env_ids, 0] = 0
            self._attack_variant[env_ids, 1] = 1
            self._attack_cooldown[env_ids, 0] = 0.35
            self._attack_cooldown[env_ids, 1] = 1.05
            self._combat_pos[env_ids, 0] = self._engine.get_root_pos(self._get_char_id())[env_ids]
            self._combat_pos[env_ids, 1] = self._engine.get_root_pos(self._get_opponent_id())[env_ids]
            self._set_character_color(self._get_char_id(), self._get_char_color(), env_ids)
            self._set_character_color(self._get_opponent_id(), self._get_opponent_color(), env_ids)
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
        return

    def _apply_action(self, actions):
        if not self._reference_control_enabled:
            return super()._apply_action(actions)
        intent, _, _, _ = self._boxing_reference.compute(
            self._engine.get_root_pos(self._get_char_id()),
            self._engine.get_root_pos(self._get_opponent_id()),
            self._combat_state[:, 0], self._fighter_health, self._opponent_health)
        base = self._init_dof_pos.expand(actions.shape[0], -1)
        reference = self._boxing_reference.reference_action(
            base, intent, 0, self._reference_joint_ids[0])
        residual = actions - base
        target = reference + self._residual_scale * residual
        target = torch.minimum(torch.maximum(target, self._action_bound_low), self._action_bound_high)
        self._engine.set_cmd(self._get_char_id(), target)
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
                                      self._get_opponent_color(),
                                      ((was_staggered & (self._stagger_buf <= 0)) &
                                       (self._opponent_health > 0)).nonzero().flatten())
        if torch.any(torch.logical_and(
                was_fighter_staggered, self._fighter_stagger_buf <= 0.0)):
            self._set_character_color(self._get_char_id(),
                                      self._get_char_color(),
                                      ((was_fighter_staggered & (self._fighter_stagger_buf <= 0)) &
                                       (self._fighter_health > 0)).nonzero().flatten())
        self._update_combat_states()
        if self._punch_demo_enabled:
            self._update_punch_demo()
        if self._opponent_attack_enabled:
            self._update_opponent_punch()
        self._enforce_ring_bounds()
        self._update_hits()
        self._update_opponent_hits()
        self._enforce_ring_bounds()
        return

    def _enforce_ring_bounds(self):
        xmin, xmax, ymin, ymax = self._ring_bounds
        for obj_id in [self._get_char_id(), self._get_opponent_id()]:
            root_pos = self._engine.get_root_pos(obj_id)
            clamped_pos = root_pos.clone()
            clamped_pos[:, 0] = torch.clamp(clamped_pos[:, 0], xmin, xmax)
            clamped_pos[:, 1] = torch.clamp(clamped_pos[:, 1], ymin, ymax)
            changed = ((clamped_pos[:, 0] <= xmin + 1e-6) |
                       (clamped_pos[:, 0] >= xmax - 1e-6) |
                       (clamped_pos[:, 1] <= ymin + 1e-6) |
                       (clamped_pos[:, 1] >= ymax - 1e-6))
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

    def _enter_state(self, side, mask, state):
        self._combat_state[mask, side] = state
        self._state_timer[mask, side] = 0.0

    def _sample_duration(self, bounds, count):
        return torch.empty(count, device=self._device).uniform_(*bounds)

    def _update_combat_states(self):
        dt = self._engine.get_timestep()
        self._state_timer += dt
        self._attack_cooldown.sub_(dt).clamp_min_(0.0)
        self._recover_timer.sub_(dt).clamp_min_(0.0)
        distance = torch.linalg.vector_norm(
            self._combat_pos[:, 1, :2] - self._combat_pos[:, 0, :2], dim=-1)
        finished = (self._fighter_health <= 0) | (self._opponent_health <= 0)
        for side, enabled in enumerate((self._punch_demo_enabled,
                                        self._opponent_attack_enabled)):
            state = self._combat_state[:, side].clone()
            health = self._fighter_health if side == 0 else self._opponent_health
            self._enter_state(side, finished & (health <= 0), CombatState.DOWN)
            self._enter_state(side, finished & (health > 0), CombatState.IDLE)
            if not enabled:
                continue
            live = ~finished
            stun = (state == CombatState.HIT_STUN) & live
            self._enter_state(side, stun & (self._state_timer[:, side] >=
                                           self._stagger_duration), CombatState.RETREAT)
            retreat = (state == CombatState.RETREAT) & live
            self._enter_state(side, retreat & (self._state_timer[:, side] >=
                                              self._retreat_duration), CombatState.IDLE)
            ended = (state == CombatState.ATTACK) & live & (
                self._state_timer[:, side] >= self._attack_duration)
            self._enter_state(side, ended, CombatState.RECOVER)
            self._recover_timer[ended, side] = self._sample_duration(
                self._recover_range, int(ended.sum()))
            recovering = (state == CombatState.RECOVER) & live
            self._enter_state(side, recovering & (self._recover_timer[:, side] <= 0),
                              CombatState.IDLE)
            deciding = ((state == CombatState.IDLE) |
                        (state == CombatState.APPROACH)) & live
            far = distance > self._attack_range
            self._enter_state(side, deciding & far, CombatState.APPROACH)
            ready = deciding & ~far & (self._attack_cooldown[:, side] <= 0)
            # Sequential reservation avoids both controllers starting in one tick.
            ready &= self._combat_state[:, 1 - side] != CombatState.ATTACK
            self._enter_state(side, deciding & ~far & ~ready, CombatState.IDLE)
            self._enter_state(side, ready, CombatState.ATTACK)
            self._attack_variant[ready, side] = 1 - self._attack_variant[ready, side]
            self._attack_cooldown[ready, side] = self._attack_duration + self._sample_duration(
                self._attack_cooldown_range, int(ready.sum()))
            consumed = self._attack_hit_consumed if side == 0 else self._opponent_punch_hit_consumed
            consumed[ready] = False

    def _update_punch_demo(self):
        self._pose_combatant(0, self._get_char_id(), self._punch_motion_lib,
                             self._punch_motion_ids, self._attack_active_buf)

    def _update_opponent_punch(self):
        self._pose_combatant(1, self._get_opponent_id(), self._opponent_punch_motion_lib,
                             self._opponent_punch_motion_ids, self._opponent_attack_active_buf)

    def _pose_combatant(self, side, obj_id, lib, motion_ids, active):
        state = self._combat_state[:, side]
        attacking = state == CombatState.ATTACK
        variant = self._attack_variant[:, side]
        punch_time = self._state_timer[:, side] * torch.where(
            variant == 0, torch.full_like(variant, 0.78, dtype=torch.float32),
            torch.ones_like(variant, dtype=torch.float32))
        time = torch.where(attacking, punch_time, 0.0)
        active_start = torch.where(variant == 0, 0.45, self._punch_active_start)
        active_end = torch.where(variant == 0, 0.95, self._punch_active_end)
        active[:] = attacking & (time >= active_start) & (time <= active_end)
        pos = self._combat_pos[:, side].clone()
        delta = self._combat_pos[:, 1 - side, :2] - pos[:, :2]
        distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(1e-6)
        direction = delta / distance.unsqueeze(-1)
        speed = torch.zeros_like(distance)
        speed[state == CombatState.APPROACH] = self._approach_speed
        speed = torch.minimum(speed, (distance - self._attack_range).clamp_min(0) /
                              self._engine.get_timestep())
        speed[state == CombatState.RETREAT] = -self._approach_speed
        speed[state == CombatState.HIT_STUN] = -self._stagger_speed
        vel = torch.zeros_like(pos)
        vel[:, :2] = direction * speed.unsqueeze(-1)
        xmin, xmax, ymin, ymax = self._ring_bounds
        center = pos.new_tensor([(xmin + xmax) / 2, (ymin + ymax) / 2])
        near_edge = ((pos[:, 0] < xmin + 0.25) | (pos[:, 0] > xmax - 0.25) |
                     (pos[:, 1] < ymin + 0.25) | (pos[:, 1] > ymax - 0.25))
        mobile = (state == CombatState.APPROACH) | (state == CombatState.RETREAT)
        inward = center - pos[:, :2]
        inward /= torch.linalg.vector_norm(inward, dim=-1).clamp_min(1e-6).unsqueeze(-1)
        vel[near_edge & mobile, :2] = inward[near_edge & mobile] * self._approach_speed
        pos += vel * self._engine.get_timestep()
        pos[:, 0].clamp_(xmin, xmax)
        pos[:, 1].clamp_(ymin, ymax)
        self._combat_pos[:, side] = pos
        # Clip yaw calibration aligns recorded punch direction with opponent bearing.
        yaw = torch.atan2(delta[:, 1], delta[:, 0]) + self._punch_demo_yaw
        axis = torch.zeros_like(pos)
        axis[:, 2] = 1
        quat = torch_util.axis_angle_to_quat(axis, yaw)
        clip_pos, root_rot, clip_vel, root_ang_vel, joint_rot, dof_vel = lib.calc_motion_frame(motion_ids, time)
        first_pos = lib.calc_motion_frame(motion_ids, torch.zeros_like(time))[0]
        displacement = torch_util.quat_rotate(quat, clip_pos - first_pos)
        pos += displacement
        pos[:, 0].clamp_(xmin, xmax)
        pos[:, 1].clamp_(ymin, ymax)
        vel += torch.where(attacking.unsqueeze(-1),
                           torch_util.quat_rotate(quat, clip_vel), 0.0)
        root_rot = torch_util.quat_mul(quat, root_rot)
        root_ang_vel = torch_util.quat_rotate(quat, root_ang_vel)
        root_ang_vel[~attacking] = 0
        dof_vel[~attacking] = 0
        dof_pos = lib.joint_rot_to_dof(joint_rot)
        if self._reference_control_enabled:
            intent = torch.full_like(state, 0, dtype=torch.long)
            intent[state == CombatState.APPROACH] = 1
            intent[state == CombatState.ATTACK] = 2
            intent[state == CombatState.RECOVER] = 3
            intent[state == CombatState.RETREAT] = 4
            reference = self._boxing_reference.reference_action(
                self._init_dof_pos.expand_as(dof_pos), intent, side,
                self._reference_joint_ids[side])
            guard_mask = (state != CombatState.ATTACK) & (state != CombatState.DOWN)
            dof_pos[guard_mask] = reference[guard_mask]
            joint_rot = self._kin_char_model.dof_to_rot(dof_pos)
        alive_ids = (state != CombatState.DOWN).nonzero(as_tuple=False).flatten()
        body_pos, body_rot = self._kin_char_model.forward_kinematics(pos, root_rot, joint_rot)
        for setter, value in ((self._engine.set_root_pos, pos),
                              (self._engine.set_root_rot, root_rot),
                              (self._engine.set_root_vel, vel),
                              (self._engine.set_root_ang_vel, root_ang_vel),
                              (self._engine.set_dof_pos, dof_pos),
                              (self._engine.set_dof_vel, dof_vel),
                              (self._engine.set_body_pos, body_pos),
                              (self._engine.set_body_rot, body_rot)):
            setter(alive_ids, obj_id, value[alive_ids])
        down_ids = (state == CombatState.DOWN).nonzero(as_tuple=False).flatten()
        if len(down_ids):
            self._apply_knockdown(down_ids, obj_id)

    def _receive_hit(self, env_ids, side):
        self._enter_state(side, torch.isin(torch.arange(self.get_num_envs(), device=self._device),
                                         env_ids), CombatState.HIT_STUN)
        self._attack_cooldown[env_ids, side] = self._stagger_duration + self._retreat_duration + 1.0
        active = self._attack_active_buf if side == 0 else self._opponent_attack_active_buf
        active[env_ids] = False
        health = self._fighter_health if side == 0 else self._opponent_health
        down_ids = env_ids[health[env_ids] <= 0]
        if len(down_ids):
            self._combat_state[down_ids, side] = CombatState.DOWN
            self._combat_state[down_ids, 1 - side] = CombatState.IDLE
            self._attack_active_buf[down_ids] = False
            self._opponent_attack_active_buf[down_ids] = False
            self._apply_knockdown(down_ids, self._get_char_id() if side == 0 else self._get_opponent_id())

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

            self._receive_hit(hit_ids, 1)
            self._set_character_color(self._get_opponent_id(),
                                      [1.0, 0.85, 0.05],
                                      hit_ids[self._opponent_health[hit_ids] > 0])
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
            self._receive_hit(hit_ids, 0)
            self._set_character_color(self._get_char_id(),
                                      [1.0, 0.85, 0.05],
                                      hit_ids[self._fighter_health[hit_ids] > 0])
            Logger.print("FIGHTER HIT! fighter health: {:.1f}".format(
                float(self._fighter_health[hit_ids[0]].item())))
        return

    def _apply_knockdown(self, env_ids, opponent_id=None):
        if opponent_id is None:
            opponent_id = self._get_opponent_id()
        root_pos = self._engine.get_root_pos(opponent_id)[env_ids].clone()
        root_pos[:, 2] = self._ring_floor_height + 0.35
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
        self._engine.set_dof_vel(env_ids, opponent_id, 0.0)
        self._engine.set_body_vel(env_ids, opponent_id, 0.0)
        self._engine.set_body_ang_vel(env_ids, opponent_id, 0.0)
        self._engine.set_body_pos(env_ids, opponent_id, body_pos)
        self._engine.set_body_rot(env_ids, opponent_id, body_rot)
        self._set_character_color(opponent_id, [0.35, 0.08, 0.08], env_ids)
        return

    def _get_char_color(self):
        if self._engine.get_name() == "newton":
            return np.array([0.35, 0.45, 0.7])
        return np.array([0.5, 0.65, 0.95])

    def _set_character_color(self, obj_id, color, env_ids=None):
        if not self._visualize:
            return
        if self._engine.get_name() != "newton":
            return

        num_bodies = self._engine.get_obj_num_bodies(obj_id)
        if env_ids is None:
            env_ids = range(self.get_num_envs())
        for env_id in env_ids:
            for body_id in range(num_bodies):
                self._engine.set_body_color(
                    int(env_id), obj_id, body_id, np.asarray(color, dtype=np.float32))
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
        selection = slice(None) if env_ids is None else env_ids
        ref_intent, ref_direction, ref_distance, ref_guard = self._boxing_reference.compute(
            self._engine.get_root_pos(self._get_char_id()),
            self._engine.get_root_pos(self._get_opponent_id()),
            self._combat_state[:, 0], self._fighter_health, self._opponent_health)
        self._info["boxing_reference_intent"] = ref_intent[selection]
        self._info["boxing_reference_direction"] = ref_direction[selection]
        self._info["boxing_reference_distance"] = ref_distance[selection]
        self._info["boxing_reference_guard_target"] = ref_guard[selection]
        for side, prefix in enumerate(("fighter", "opponent")):
            self._info[prefix + "_state"] = self._combat_state[selection, side]
            self._info[prefix + "_attack_cooldown"] = self._attack_cooldown[selection, side]
            stun = self._fighter_stagger_buf if side == 0 else self._stagger_buf
            self._info[prefix + "_hit_stun"] = stun[selection]
        self._info["combat_state_names"] = tuple(state.name for state in CombatState)
        self._info["fighter_distance"] = torch.linalg.vector_norm(
            self._engine.get_root_pos(self._get_char_id())[:, :2] -
            self._engine.get_root_pos(self._get_opponent_id())[:, :2], dim=-1)[selection]
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
            self._info["fighter_attack_variant"] = self._attack_variant[:, 0]
            self._info["opponent_attack_variant"] = self._attack_variant[:, 1]
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
            self._info["fighter_attack_variant"] = self._attack_variant[env_ids, 0]
            self._info["opponent_attack_variant"] = self._attack_variant[env_ids, 1]
        return

