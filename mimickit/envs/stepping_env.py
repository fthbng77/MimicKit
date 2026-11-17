"""Forward stepping task built on balance control."""

import torch

from envs.balance_env import BalanceEnv


class SteppingEnv(BalanceEnv):
    """Balance humanoid while rewarding controlled forward motion."""

    def __init__(self, *args, **kwargs):
        config = kwargs.get("env_config", {})
        self._step_period = float(config.get("step_period", kwargs.pop("step_period", 0.9)))
        self._target_speed = float(config.get("target_speed", kwargs.pop("target_speed", 0.25)))
        self._progress_weight = float(config.get("progress_reward_weight", kwargs.pop("progress_reward_weight", 0.5)))
        self._step_length = float(config.get("step_length", kwargs.pop("step_length", 0.16)))
        self._foot_target_weight = float(config.get("foot_target_reward_weight", kwargs.pop("foot_target_reward_weight", 0.25)))
        self._foot_clearance = float(config.get("foot_clearance", kwargs.pop("foot_clearance", 0.04)))
        self._foot_ground_height = float(config.get("foot_ground_height", kwargs.pop("foot_ground_height", 0.02)))
        self._contact_threshold = float(config.get("foot_contact_threshold", kwargs.pop("foot_contact_threshold", 1.0)))
        self._contact_reward_weight = float(config.get("foot_contact_reward_weight", kwargs.pop("foot_contact_reward_weight", 0.2)))
        super().__init__(*args, **kwargs)

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        self._step_phase = torch.zeros(self.get_num_envs(), device=self._device)
        self._foot_ids = self._build_body_ids_tensor(["right_foot", "left_foot"])

    def _update_misc(self):
        super()._update_misc()
        self._step_phase = torch.remainder(self._time_buf / self._step_period, 1.0)

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        ids = torch.arange(self.get_num_envs(), device=self._device) if env_ids is None else env_ids
        phase = self._step_phase[ids]
        phase_obs = torch.stack((torch.sin(2.0 * torch.pi * phase), torch.cos(2.0 * torch.pi * phase)), dim=-1)
        foot_obs = self._foot_features(ids)
        contact_obs = self._foot_contact_strength(ids)
        return torch.cat((obs, phase_obs, foot_obs, contact_obs), dim=-1)

    def _foot_features(self, ids):
        char_id = self._get_char_id()
        root = self._engine.get_root_pos(char_id)[ids]
        feet = self._engine.get_body_pos(char_id)[ids][:, self._foot_ids, :]
        phase = self._step_phase[ids]
        right_swing = (phase < 0.5).to(feet.dtype)
        left_swing = 1.0 - right_swing
        target_x = root[:, 0:1] + torch.cat((right_swing[:, None], left_swing[:, None]), dim=1) * self._step_length
        target_z = self._foot_ground_height + torch.cat((right_swing[:, None], left_swing[:, None]), dim=1) * self._foot_clearance
        return torch.stack((feet[..., 0] - target_x, feet[..., 2] - target_z), dim=-1).reshape(feet.shape[0], 4)

    def _foot_contact_strength(self, ids):
        forces = self._engine.get_ground_contact_forces(self._get_char_id())[ids][:, self._foot_ids, :]
        return (torch.linalg.vector_norm(forces, dim=-1) / max(self._contact_threshold, 1e-6)).clamp(max=1.0)

    def _update_reward(self):
        super()._update_reward()
        forward_speed = self._engine.get_root_vel(self._get_char_id())[:, 0]
        progress = 1.0 - torch.abs(forward_speed - self._target_speed) / max(self._target_speed, 1e-6)
        self._reward_buf += self._progress_weight * torch.clamp(progress, -1.0, 1.0)
        features = self._foot_features(torch.arange(self.get_num_envs(), device=self._device))
        foot_error = torch.mean(torch.square(features), dim=-1)
        self._reward_buf -= self._foot_target_weight * foot_error
        ids = torch.arange(self.get_num_envs(), device=self._device)
        contact = self._foot_contact_strength(ids)
        phase = self._step_phase
        right_swing = (phase < 0.5).to(contact.dtype)
        expected = torch.stack((1.0 - right_swing, right_swing), dim=-1)
        contact_score = torch.sum(expected * contact, dim=-1) - torch.sum((1.0 - expected) * contact, dim=-1)
        self._reward_buf += self._contact_reward_weight * contact_score

    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)
        selection = slice(None) if env_ids is None else env_ids
        self._info["step_phase"] = self._step_phase[selection]
        self._info["forward_speed"] = self._engine.get_root_vel(self._get_char_id())[selection, 0]
        self._info["target_speed"] = self._target_speed
        feature_ids = torch.arange(self.get_num_envs(), device=self._device) if env_ids is None else env_ids
        self._info["foot_target_error"] = torch.mean(torch.square(self._foot_features(feature_ids)), dim=-1)
        self._info["foot_contact"] = self._foot_contact_strength(feature_ids)
