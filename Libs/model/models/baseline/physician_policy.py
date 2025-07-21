import pandas as pd
import numpy as np
from collections import defaultdict
from typing import List, Tuple, Dict, Any, Optional, Union, Sequence


# 默认用于状态键精度
_KEY_DECIMALS: int = 6


# ================================
#       Physician Policy
# ================================

class PhysicianPolicy:
    """
    Physician policy baseline: replicates the empirical distribution of physician actions
    observed in the dataset for each state. Supports multi-dimensional actions and RL environment integration.
    """

    def __init__(
        self,
        trajectory_csv: str,
        state_cols: Optional[Sequence[str]] = None,
        action_cols: Optional[Sequence[str]] = None,
        *,
        id_col: str = "subject_id",
        time_col: str = "hours_from_onset",
        key_decimals: int = _KEY_DECIMALS,
    ) -> None:
        """基于数据集的医师策略基线。

        该策略通过统计历史数据中 ``state -> action`` 的经验分布，在给定状态时重现医生行为。

        Args:
            trajectory_csv: 轨迹 CSV 路径。
            state_cols: 状态特征列。若为 ``None``，则使用除 ``id``, ``time``, ``reward``, ``done`` 与动作列之外的所有列。
            action_cols: 动作列，默认 ['peep_bin', 'fio2_bin', 'tidal_volume_ibw_bin']。
            id_col: 患者唯一标识列名。
            time_col: 时间步列名。
            key_decimals: 转换状态到字典键时保留的小数位数，用于降低浮点舍入误差。
        """
        self.trajectory_csv = trajectory_csv
        self.id_col = id_col
        self.time_col = time_col
        self._key_decimals: int = key_decimals

        # Load data
        self.df = pd.read_csv(trajectory_csv)
        self.df = self.df.sort_values([id_col, time_col])

        # Default action columns - use TaskManager if available
        if action_cols is None:
            try:
                from Libs.utils.task_manager import get_current_task_config
                task_config = get_current_task_config()
                action_cols = task_config.action_cols
            except:
                # Fallback to vent task columns
                action_cols = ['peep_bin', 'fio2_bin', 'tidal_volume_ibw_bin']
        self.action_cols = action_cols

        # Default state columns: exclude id, time, reward, done, action columns
        if state_cols is None:
            exclude = set([id_col, time_col, 'reward', 'done'] + action_cols)
            state_cols = [c for c in self.df.columns if c not in exclude]
        self.state_cols = state_cols

        # Build transitions and state-action mapping
        self.transitions = self._build_transitions()
        self.state_action_dict = self._build_state_action_dict()

        # For random action fallback
        self.all_actions: np.ndarray = self.df[self.action_cols].values

        # 便于复现实验的随机数发生器
        self._rng: np.random.Generator = np.random.default_rng()

    def _build_transitions(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Build (state, action, next_state) transitions from the trajectory dataframe.

        Returns:
            List of (state, action, next_state) tuples.
        """
        transitions = []
        for subject_id, group in self.df.groupby(self.id_col):
            group = group.sort_values(self.time_col)
            states = group[self.state_cols].values
            actions = group[self.action_cols].values
            for i in range(len(group) - 1):
                st = states[i]
                at = actions[i]
                st1 = states[i + 1]
                transitions.append((st, at, st1))
        return transitions

    def _build_state_action_dict(self) -> Dict[Tuple, List[np.ndarray]]:
        """
        Build a mapping from state (as a string key) to list of actions taken by physicians.

        Returns:
            Dictionary mapping state string to list of actions.
        """
        state_action_dict: Dict[Tuple, List[np.ndarray]] = defaultdict(list)
        for st, at, _ in self.transitions:
            state_key = self._state_to_key(st)
            state_action_dict[state_key].append(at)
        return state_action_dict

    def _state_to_key(self, state: Union[np.ndarray, Sequence[float]]) -> Tuple:
        """将 ``state`` 转换为可哈希的键以用于字典查找。

        采用对浮点数进行四舍五入并转为 ``tuple`` 的方式，避免 ``np.array2string`` 带来的字符串解析和内存开销。

        Args:
            state: 状态向量。

        Returns:
            Tuple: 可用作 ``dict`` key 的元组。
        """
        arr = np.asarray(state, dtype=np.float64)
        return tuple(np.round(arr, self._key_decimals))

    def act(self, state: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """
        Given a state, sample an action according to the empirical physician policy.

        Args:
            state: State vector.

        Returns:
            Action vector (same shape as action_cols).
        """
        state_key = self._state_to_key(state)
        actions = self.state_action_dict.get(state_key, None)
        if not actions:
            # Unseen state: fallback to random action
            return self.random_action()
        idx = self._rng.integers(len(actions))
        return actions[idx]

    def random_action(self) -> np.ndarray:
        """
        Sample a random action from the empirical action space.

        Returns:
            Action vector.
        """
        idx = self._rng.integers(len(self.all_actions))
        return self.all_actions[idx]

    def get_transitions(self) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Get all (state, action, next_state) transitions.

        Returns:
            List of (state, action, next_state) tuples.
        """
        return self.transitions

    def evaluate_policy(
        self,
        env: Any,
        num_episodes: int = 100,
        *,
        render: bool = False,
        seed: Optional[int] = None,
    ) -> float:
        """
        Evaluate the physician policy in a given RL environment.

        Args:
            env: RL environment with reset() and step(action) methods.
            num_episodes: Number of episodes to run.
            render: Whether to render the environment.
            seed: Random seed for reproducibility.

        Returns:
            Average total reward.
        """
        if seed is not None:
            self.set_random_seed(seed)
            env.seed(seed)

        total_rewards: List[float] = []
        for ep in range(num_episodes):
            state = env.reset()
            done = False
            episode_reward = 0.0
            while not done:
                action = self.act(state)
                # If env expects int or list, convert accordingly
                if hasattr(env.action_space, "shape") and len(env.action_space.shape) == 1:
                    # Multi-discrete or continuous
                    action_to_env = action.astype(int) if np.issubdtype(
                        action.dtype, np.integer) else action
                else:
                    # Discrete
                    action_to_env = int(action[0]) if isinstance(
                        action, np.ndarray) and action.size == 1 else action
                state, reward, done, info = env.step(action_to_env)
                episode_reward += reward
                if render:
                    env.render()
            total_rewards.append(episode_reward)
        avg_reward = float(np.mean(total_rewards))
        print(
            f"Physician policy average reward over {num_episodes} episodes: {avg_reward:.2f}")
        return avg_reward

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def set_random_seed(self, seed: int) -> None:
        """设置随机种子以确保可复现性。"""
        self._rng = np.random.default_rng(seed)

    def __repr__(self) -> str:  # noqa: D401
        return (
            f"{self.__class__.__name__}(trajectory_csv='{self.trajectory_csv}', "
            f"state_cols={len(self.state_cols)}, action_cols={self.action_cols})"
        )
