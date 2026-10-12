"""Sequential EEF/joint control for the single-environment harness protocol v3."""

import torch

from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

from robolab.robots.droid import DroidIKActionCfg, DroidJointPositionActionCfg


class HybridArmAction(ActionTerm):
    """Dispatch mode + seven values to exactly one existing arm controller."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if self.num_envs != 1:
            raise ValueError("The hybrid harness controller requires num_envs=1")
        self._pose = cfg.pose.class_type(cfg.pose, env)
        self._joints = cfg.joints.class_type(cfg.joints, env)
        self._raw_actions = torch.zeros((1, 8), device=self.device)
        self._active = None

    @property
    def action_dim(self):
        return 8

    @property
    def raw_actions(self):
        return self._raw_actions

    @property
    def processed_actions(self):
        if self._active is None:
            return self._raw_actions
        return torch.cat((self._raw_actions[:, :1], self._active.processed_actions), dim=1)

    def process_actions(self, actions):
        if actions.shape != (1, 8) or not torch.isfinite(actions).all():
            raise ValueError("Hybrid arm actions must be 1 x 8 finite values")
        mode = actions[0, 0].item()
        if mode not in (0, 1):
            raise ValueError("Hybrid mode must be 0 (EEF) or 1 (joints)")
        active = self._pose if mode == 0 else self._joints
        if active is not self._active:
            active.reset()
        active.process_actions(actions[:, 1:])
        self._raw_actions[:] = actions
        self._active = active

    def apply_actions(self):
        if self._active is not None:
            self._active.apply_actions()

    def reset(self, env_ids=None):
        self._pose.reset(env_ids)
        self._joints.reset(env_ids)
        self._raw_actions[:] = 0
        self._active = None


@configclass
class HybridArmActionCfg(ActionTermCfg):
    class_type = HybridArmAction
    pose = DroidIKActionCfg().arm_action
    joints = DroidJointPositionActionCfg().body


@configclass
class DroidHarnessActionCfg:
    # 8 arm values (mode + payload), followed by the existing binary gripper.
    arm_action = HybridArmActionCfg(asset_name="robot")
    finger_joint = DroidIKActionCfg().finger_joint
