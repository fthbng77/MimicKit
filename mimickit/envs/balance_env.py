import numpy as np
import torch
import gymnasium.spaces as spaces

import envs.char_env as char_env
import engines.engine as engine
import util.torch_util as torch_util
import envs.base_env as base_env


class BalanceEnv(char_env.CharEnv):
    """Four-joint balance task with all other joints held at init pose."""

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        self._active_joint_names = env_config.get("active_joints", [
            "right_hip", "right_knee", "left_hip", "left_knee"])
        self._target_root_height = float(env_config.get("target_root_height", 0.882416))
        self._fall_height = float(env_config.get("fall_height", 0.55))
        self._fall_up = float(env_config.get("fall_up", 0.25))
        self._action_penalty = float(env_config.get("action_penalty", 0.002))
        self._upright_weight = float(env_config.get("upright_reward_weight", 0.5))
        self._reset_joint_noise = float(env_config.get("reset_joint_noise", 0.08))
        self._reset_root_tilt = float(env_config.get("reset_root_tilt", 0.10))
        self._push_interval = float(env_config.get("push_interval", 0.0))
        self._push_start_time = float(env_config.get("push_start_time", 0.75))
        self._push_speed = float(env_config.get("push_speed", 0.0))
        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)

    def _build_action_space(self):
        full_space = super()._build_action_space()
        self._full_action_low = full_space.low
        self._full_action_high = full_space.high
        dof_ids = []
        for name in self._active_joint_names:
            joint_id = next((i for i, joint in enumerate(self._kin_char_model._joints)
                             if joint is not None and joint.name == name), -1)
            assert joint_id >= 0, "Unknown active joint: {}".format(name)
            dof_dim = self._kin_char_model.get_joint_dof_dim(joint_id)
            dof_start = self._kin_char_model.get_joint_dof_idx(joint_id)
            dof_ids.extend(range(dof_start, dof_start + dof_dim))
        self._active_dof_ids = torch.tensor(dof_ids, device=self._device, dtype=torch.long)
        return spaces.Box(low=self._full_action_low[dof_ids],
                          high=self._full_action_high[dof_ids])

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        self._active_dof_ids = self._active_dof_ids.to(self._device)
        self._full_action_low = torch.tensor(self._full_action_low, device=self._device)
        self._full_action_high = torch.tensor(self._full_action_high, device=self._device)
        self._root_up = torch.zeros((self.get_num_envs(), 3), device=self._device)
        self._balance_fallen = torch.zeros(self.get_num_envs(), device=self._device, dtype=torch.bool)
        self._next_push_time = torch.full((self.get_num_envs(),), float("inf"),
                                          device=self._device)
        self._last_push = torch.zeros((self.get_num_envs(), 3), device=self._device)

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) == 0:
            return
        char_id = self._get_char_id()
        num_resets = len(env_ids)
        self._balance_fallen[env_ids] = False
        self._last_push[env_ids] = 0.0
        if self._push_interval > 0.0 and self._push_speed > 0.0:
            self._next_push_time[env_ids] = self._push_start_time
        else:
            self._next_push_time[env_ids] = float("inf")
        dof_pos = self._engine.get_dof_pos(char_id)[env_ids].clone()
        noise = torch.empty((num_resets, len(self._active_dof_ids)), device=self._device)
        noise.uniform_(-self._reset_joint_noise, self._reset_joint_noise)
        dof_pos[:, self._active_dof_ids] += noise
        self._engine.set_dof_pos(env_ids, char_id, dof_pos)

        axis = torch.zeros((num_resets, 3), device=self._device)
        axis[:, :2] = torch.randn((num_resets, 2), device=self._device)
        axis /= torch.linalg.vector_norm(axis, dim=-1, keepdim=True).clamp_min(1e-6)
        angle = torch.empty(num_resets, device=self._device)
        angle.uniform_(-self._reset_root_tilt, self._reset_root_tilt)
        root_rot = torch_util.quat_mul(
            torch_util.axis_angle_to_quat(axis, angle),
            self._init_root_rot.expand(num_resets, -1))
        self._engine.set_root_rot(env_ids, char_id, root_rot)
        self._reset_char_rigid_body_state(env_ids)

    def _update_misc(self):
        super()._update_misc()
        if self._push_interval <= 0.0 or self._push_speed <= 0.0:
            return
        push_ids = (self._time_buf >= self._next_push_time).nonzero(as_tuple=False).flatten()
        if len(push_ids) == 0:
            return
        direction = torch.randn((len(push_ids), 2), device=self._device)
        direction /= torch.linalg.vector_norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
        push = torch.zeros((len(push_ids), 3), device=self._device)
        push[:, :2] = direction * self._push_speed
        char_id = self._get_char_id()
        root_vel = self._engine.get_root_vel(char_id)[push_ids].clone()
        self._engine.set_root_vel(push_ids, char_id, root_vel + push)
        self._last_push[push_ids] = push
        self._next_push_time[push_ids] += self._push_interval

    def _apply_action(self, actions):
        char_id = self._get_char_id()
        full_action = self._init_dof_pos.expand(actions.shape[0], -1).clone()
        active = torch.minimum(torch.maximum(
            actions, self._action_bound_low), self._action_bound_high)
        full_action[:, self._active_dof_ids] = active
        self._engine.set_cmd(char_id, full_action)

    def _update_reward(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        up_axis = torch.zeros_like(root_pos)
        up_axis[:, 2] = 1.0
        self._root_up[:] = torch_util.quat_rotate(root_rot, up_axis)
        height_error = torch.abs(root_pos[:, 2] - self._target_root_height)
        upright = torch.clamp(self._root_up[:, 2], min=0.0, max=1.0)
        height_score = torch.exp(-4.0 * height_error)
        action = self._engine.get_dof_pos(char_id)[:, self._active_dof_ids]
        action_error = torch.mean(torch.square(action), dim=-1)
        self._reward_buf[:] = height_score + self._upright_weight * upright \
            - self._action_penalty * action_error
        self._reward_buf[self._balance_fallen] = 0.0

    def _update_done(self):
        super()._update_done()
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        self._balance_fallen[:] = (root_pos[:, 2] < self._fall_height) | \
            (self._root_up[:, 2] < self._fall_up)
        self._done_buf[self._balance_fallen] = base_env.DoneFlags.FAIL.value

    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)
        selection = slice(None) if env_ids is None else env_ids
        self._info["balance_fallen"] = self._balance_fallen[selection]
        self._info["root_up_z"] = self._root_up[selection, 2]
        self._info["active_dof_ids"] = self._active_dof_ids
        self._info["balance_last_push"] = self._last_push[selection]
