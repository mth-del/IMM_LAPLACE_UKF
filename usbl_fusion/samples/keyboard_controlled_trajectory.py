"""
键盘控制“船”的轨迹，并动态生成预测轨迹（NE 平面）

坐标约定（与项目绘图一致）：
- 状态 x = [pN, pE, vN, vE]
- 绘图：横轴 E，纵轴 N

运行方式（在仓库根目录 IMM_LAPLACE_UKF 下）：
    python usbl_fusion/samples/keyboard_controlled_trajectory.py

按键说明：
- W / S：北向加速度 aN 增/减
- D / A：东向加速度 aE 增/减
- Space：加速度清零
- P：暂停/继续
- R：重置轨迹
- Q / ESC：退出

实现说明：
- “真实轨迹”：用当前 a_cmd 对状态进行离散积分（propagate_state）
- “预测轨迹”：从当前状态出发，假设 a_cmd 在未来保持不变，向前滚动 H 步得到预测轨迹（虚线）
"""

import sys
from pathlib import Path

# 让脚本可从 samples/ 子目录直接运行
sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from models.dynamics import propagate_state
from tools.logging_setup import setup_logger, logger
from tools.mpl_setup import configure_matplotlib


class KeyboardTrajectorySim:
    def __init__(
        self,
        *,
        dt: float = 0.1,
        horizon_steps: int = 200,
        acc_step: float = 0.01,
        acc_limit: float = 0.2,
        v_limit: float = 5.0,
        history_max_len: int = 5000,
    ):
        self.dt = float(dt)
        self.horizon_steps = int(horizon_steps)
        self.acc_step = float(acc_step)
        self.acc_limit = float(acc_limit)
        self.v_limit = float(v_limit)
        self.history_max_len = int(history_max_len)

        self.paused = False
        self.a_cmd = np.zeros(2, dtype=float)  # [aN, aE]
        self.x = np.zeros(4, dtype=float)      # [pN, pE, vN, vE]

        self.t = 0.0
        self.t_hist: list[float] = []
        self.x_hist: list[np.ndarray] = []

    def reset(self):
        self.paused = False
        self.a_cmd[:] = 0.0
        self.x[:] = 0.0
        self.t = 0.0
        self.t_hist.clear()
        self.x_hist.clear()
        logger.info("已重置状态/轨迹")

    def _clip(self):
        self.a_cmd[:] = np.clip(self.a_cmd, -self.acc_limit, self.acc_limit)
        self.x[2:4] = np.clip(self.x[2:4], -self.v_limit, self.v_limit)

    def step(self):
        if self.paused:
            return
        self.x = propagate_state(self.x, self.a_cmd, self.dt)
        self._clip()
        self.t += self.dt

        self.t_hist.append(self.t)
        self.x_hist.append(self.x.copy())
        if len(self.x_hist) > self.history_max_len:
            self.t_hist = self.t_hist[-self.history_max_len :]
            self.x_hist = self.x_hist[-self.history_max_len :]

    def predict_traj(self) -> np.ndarray:
        """
        从当前状态出发，用“未来 a_cmd 保持不变”的假设向前滚动，输出 (H+1, 4) 状态序列。
        """
        H = self.horizon_steps
        x_pred = np.zeros((H + 1, 4), dtype=float)
        x_pred[0, :] = self.x
        for i in range(H):
            x_pred[i + 1, :] = propagate_state(x_pred[i, :], self.a_cmd, self.dt)
        return x_pred

    def on_key(self, key: str):
        k = (key or "").lower()
        if k in ("q", "escape"):
            raise SystemExit
        if k == "p":
            self.paused = not self.paused
            logger.info("paused={}", self.paused)
            return
        if k == "r":
            self.reset()
            return
        if k == " ":
            self.a_cmd[:] = 0.0
            logger.info("加速度清零")
            return

        if k == "w":
            self.a_cmd[0] += self.acc_step
        elif k == "s":
            self.a_cmd[0] -= self.acc_step
        elif k == "d":
            self.a_cmd[1] += self.acc_step
        elif k == "a":
            self.a_cmd[1] -= self.acc_step
        else:
            return

        self._clip()
        logger.debug("a_cmd=[aN={:.3f}, aE={:.3f}]", self.a_cmd[0], self.a_cmd[1])


def main():
    setup_logger(level="INFO")
    backend = configure_matplotlib(gui=True)
    logger.info("matplotlib backend = {}", backend)

    import matplotlib.pyplot as plt

    sim = KeyboardTrajectorySim(dt=0.1, horizon_steps=250, acc_step=0.01, acc_limit=0.2)
    logger.info("键盘控制轨迹启动：dt={}s, horizon_steps={}", sim.dt, sim.horizon_steps)

    fig, ax = plt.subplots()
    ax.set_title("Keyboard-controlled trajectory (E-N) with predicted path")
    ax.set_xlabel("E (m)")
    ax.set_ylabel("N (m)")
    ax.grid(True)
    ax.axis("equal")

    (line_hist,) = ax.plot([], [], label="history")
    (pt_now,) = ax.plot([], [], "o", label="now")
    (line_pred,) = ax.plot([], [], "--", label="pred (const a_cmd)")

    info = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        family="monospace",
    )

    ax.legend(loc="lower right")

    def _redraw():
        # 历史轨迹
        if sim.x_hist:
            hist = np.vstack(sim.x_hist)  # (K,4)
            e = hist[:, 1]
            n = hist[:, 0]
            line_hist.set_data(e, n)
            pt_now.set_data([hist[-1, 1]], [hist[-1, 0]])
        else:
            line_hist.set_data([], [])
            pt_now.set_data([], [])

        # 预测轨迹
        pred = sim.predict_traj()
        line_pred.set_data(pred[:, 1], pred[:, 0])

        # 动态信息
        info.set_text(
            "t={:.1f}s  paused={}\n"
            "pN={:.2f} pE={:.2f}\n"
            "vN={:.2f} vE={:.2f}\n"
            "aN={:.3f} aE={:.3f}".format(
                sim.t,
                sim.paused,
                sim.x[0],
                sim.x[1],
                sim.x[2],
                sim.x[3],
                sim.a_cmd[0],
                sim.a_cmd[1],
            )
        )

        # 自适应视野（留一定边界）
        all_e = np.concatenate([pred[:, 1], line_hist.get_xdata() if sim.x_hist else np.array([])])
        all_n = np.concatenate([pred[:, 0], line_hist.get_ydata() if sim.x_hist else np.array([])])
        if all_e.size > 0:
            e_min, e_max = float(np.min(all_e)), float(np.max(all_e))
            n_min, n_max = float(np.min(all_n)), float(np.max(all_n))
            pad_e = max(5.0, 0.1 * (e_max - e_min + 1e-6))
            pad_n = max(5.0, 0.1 * (n_max - n_min + 1e-6))
            ax.set_xlim(e_min - pad_e, e_max + pad_e)
            ax.set_ylim(n_min - pad_n, n_max + pad_n)

        fig.canvas.draw_idle()

    def _on_key(event):
        try:
            sim.on_key(event.key)
            _redraw()
        except SystemExit:
            plt.close(fig)

    def _on_timer():
        sim.step()
        _redraw()

    fig.canvas.mpl_connect("key_press_event", _on_key)

    timer = fig.canvas.new_timer(interval=int(sim.dt * 1000))
    timer.add_callback(_on_timer)
    timer.start()

    _redraw()
    plt.show()


if __name__ == "__main__":
    main()


