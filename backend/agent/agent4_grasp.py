"""
Closed-loop grasp with RETRY for the SO-100 follower — the "smart" pick.

The imitation policy (SmolVLA / ACT) drives the motion. After the gripper closes
we VERIFY the grasp with two independent, sensor-free signals from the Feetech
gripper motor (id 6), the thresholds for which come from calibrate_grasp.py:

  1) Torque feedback  — Present_Current. Pressing on an object keeps current high;
                        an empty closed gripper settles low.
  2) Gripper width    — Present_Position. An empty gripper closes ALL the way; an
                        object holds the fingers open at its width.
  3) (optional) Vision — Fanar-Oryx confirms the target is no longer on the table
                        (i.e. it's now in the claw). Enabled with BASEER_GRASP_VISION=1.

If the grasp FAILED: open, return to the ready pose, and try again (up to N). That
is the behaviour you asked for — detect the miss, re-approach, regrasp — instead of
blindly continuing with an empty claw.

IMPORTANT — recovery is mostly LEARNED FROM DATA. If the dataset only contains
clean first-try grasps, the policy can freeze when it misses (an out-of-distribution
pose it never saw). Record a handful of episodes where you deliberately miss, then
re-approach and grab — the policy then learns to recover on its own, and this
controller becomes the success-gate + safety net on top.

Usage (on the machine with the arm, lerobot env):
    /opt/anaconda3/envs/lerobot/bin/python grasp.py \
        --policy ~/baseer/policy_vla/pretrained_model \
        --port /dev/tty.usbmodem5AB90677591 --id follower_so100 \
        --task "Pick up the hair serum and place it in the delivery zone" \
        --cam 0 --attempts 3

Or import GraspController and call .pick(task) from the agent's deliver().
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

# allow `python backend/agent/agent4_grasp.py ...` + importing sibling backend modules
_BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # backend/
sys.path.insert(0, _BACKEND)

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig
from lerobot.utils.constants import OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, hw_to_dataset_features

# calibration artifacts live at the backend/ root (shared with backend/robot/ scripts)
CFG_PATH = os.path.join(_BACKEND, "grasp_cfg.json")
DELIVERY_PATH = os.path.join(_BACKEND, "delivery_pose.json")
LOCALIZATION_PATH = os.path.join(_BACKEND, "localization_map.json")

# Conservative fallbacks if calibrate_grasp.py hasn't been run yet.
_DEFAULTS = {
    "gripper_open_cmd": 60.0,
    "gripper_close_cmd": 0.0,
    "empty_pos": 2.0,
    "pos_threshold": 8.0,
    "empty_current": 40.0,
    "current_threshold": 120.0,
}


def _load_cfg():
    cfg = dict(_DEFAULTS)
    if os.path.exists(CFG_PATH):
        cfg.update(json.load(open(CFG_PATH)))
        print(f"[grasp] thresholds from {CFG_PATH}: "
              f"pos>{cfg['pos_threshold']}  current>{cfg['current_threshold']}")
    else:
        print(f"[grasp] WARNING: {CFG_PATH} not found — using rough defaults. "
              f"Run calibrate_grasp.py for reliable detection.")
    return cfg


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GraspController:
    def __init__(self, policy_path, port, robot_id="follower_so100", cam_index=0,
                 fps=30, robot=None):
        self.cfg = _load_cfg()
        self.fps = fps
        self.device = _device()
        print(f"[grasp] device={self.device}")

        # --- robot (reuse an open one if the caller passes it) ---
        if robot is None:
            cams = {"front": OpenCVCameraConfig(index_or_path=cam_index,
                                                width=640, height=480, fps=fps)}
            self.robot = SO100Follower(SO100FollowerConfig(port=port, id=robot_id, cameras=cams))
            self.robot.connect()
            self._owns_robot = True
        else:
            self.robot = robot
            self._owns_robot = False
        self.action_keys = list(self.robot.action_features.keys())
        # Feature spec to aggregate the robot's per-motor floats into the
        # dataset-frame format the policy expects (observation.state + images).
        self.obs_features = hw_to_dataset_features(self.robot.observation_features, OBS_STR)

        # --- policy + processors loaded straight from the checkpoint ---
        # Policy-AGNOSTIC: the class is chosen from the checkpoint's config.type, so
        # the same controller deploys SmolVLA, ACT, pi0/pi05, or GR00T unchanged —
        # just point --policy at a different checkpoint.
        policy_path = os.path.expanduser(policy_path)
        pcfg = PreTrainedConfig.from_pretrained(policy_path)
        policy_cls = get_policy_class(pcfg.type)
        self.policy = policy_cls.from_pretrained(policy_path).to(self.device).eval()
        # The checkpoint's processors were saved with the TRAINING device (cuda).
        # Override to the local device so deploy works on a Mac (mps/cpu).
        dev = self.device.type
        self.pre, self.post = make_pre_post_processors(
            pcfg, pretrained_path=policy_path,
            preprocessor_overrides={"device_processor": {"device": dev}},
            postprocessor_overrides={"device_processor": {"device": dev}},
        )
        print(f"[grasp] policy loaded: type={pcfg.type} class={policy_cls.__name__} device={dev}")

        # capture the start pose as "home" so we can reset between attempts
        self.home_pose = {k: v for k, v in self.robot.get_observation().items()
                          if k.endswith(".pos")}

        # optional localization map (pixel -> hover joints) from calibrate_localization.py
        self.loc_map = None
        if os.path.exists(LOCALIZATION_PATH):
            self.loc_map = json.load(open(LOCALIZATION_PATH))
            print(f"[grasp] localization map loaded ({self.loc_map.get('n_samples')} samples) "
                  f"— will pre-position above the object via Oryx.")

    # -- low-level gripper feedback ----------------------------------------
    def _gripper(self, field):
        return float(self.robot.bus.sync_read(field)["gripper"])

    def grasp_ok(self):
        """True if both torque AND width say an object is held (sampled over ~0.3s)."""
        pos, cur = [], []
        for _ in range(max(3, self.fps // 3)):
            pos.append(self._gripper("Present_Position"))
            cur.append(abs(self._gripper("Present_Current")))
            time.sleep(1.0 / self.fps)
        p, c = np.mean(pos[-3:]), np.mean(cur[-3:])
        ok = (p > self.cfg["pos_threshold"]) and (c > self.cfg["current_threshold"])
        print(f"[grasp] check: pos={p:.1f}(>{self.cfg['pos_threshold']}) "
              f"current={c:.0f}(>{self.cfg['current_threshold']}) -> {'HELD' if ok else 'EMPTY'}")
        return ok

    # -- motion helpers -----------------------------------------------------
    def _send_full(self, pose):
        self.robot.send_action({k: float(pose[k]) for k in pose})

    def open_gripper(self):
        obs = {k: v for k, v in self.robot.get_observation().items() if k.endswith(".pos")}
        obs["gripper.pos"] = self.cfg["gripper_open_cmd"]
        self._send_full(obs)
        time.sleep(0.6)

    def _glide_to(self, target, duration_s=2.0):
        """Smoothly interpolate the arm from its current pose to `target` (dict of .pos)."""
        cur = {k: v for k, v in self.robot.get_observation().items() if k in target}
        steps = max(int(duration_s * self.fps), 1)
        for s in range(1, steps + 1):
            t = s / steps
            self._send_full({k: cur[k] * (1 - t) + target[k] * t for k in cur})
            time.sleep(1.0 / self.fps)

    def go_home(self, duration_s=2.5):
        """Smoothly interpolate back to the captured start pose (in-distribution restart)."""
        self._glide_to(self.home_pose, duration_s)

    def _arm_current(self):
        """Combined |current| on the lift+elbow motors (the joints that bear a downward push)."""
        c = self.robot.bus.sync_read("Present_Current")
        return abs(float(c["shoulder_lift"])) + abs(float(c["elbow_flex"]))

    def lower_until_contact(self, step_deg=1.5, max_drop_deg=35.0, settle_s=0.12):
        """Lower the arm in small steps until the held object meets the table — detected
        by a RISE in lift/elbow motor current (torque), not vision. Stops on contact and
        leaves the object resting; caller then opens the gripper. Returns True on contact.

        Direction: on this arm 'down' = INCREASING shoulder_lift. Override the signed step
        with BASEER_LOWER_STEP (e.g. -1.5) if your calibration's sign is reversed."""
        step = float(os.environ.get("BASEER_LOWER_STEP", str(abs(step_deg))))  # +ve = lower (this arm)
        margin = float(os.environ.get("BASEER_CONTACT_MARGIN", "120"))
        pose = {k: v for k, v in self.robot.get_observation().items() if k.endswith(".pos")}
        pose["gripper.pos"] = self.cfg["gripper_close_cmd"]      # keep holding while lowering
        base, dropped, readings = None, 0.0, []
        while dropped < max_drop_deg:
            pose["shoulder_lift.pos"] += step
            self._send_full(pose)
            time.sleep(settle_s)
            dropped += abs(step)
            cur = self._arm_current()
            readings.append(cur)
            if base is None and len(readings) >= 3:
                base = sum(readings[:3]) / 3.0                   # free-lowering baseline
            if base is not None and cur > base + margin:
                print(f"[grasp] set-down CONTACT (current {cur:.0f} > base {base:.0f}+{margin:.0f}) "
                      f"after {dropped:.0f}° drop")
                return True
        print(f"[grasp] no clear contact after {max_drop_deg:.0f}° drop — releasing anyway")
        return False

    def deliver_to_zone(self):
        """Replay the saved delivery waypoints (fixed zone), then release.

        Returns True if a scripted trajectory was available and executed, False if
        no delivery_pose.json exists (caller should fall back to policy-carry).
        While carrying, the gripper is held CLOSED so the object isn't dropped."""
        if not os.path.exists(DELIVERY_PATH):
            return False
        waypoints = json.load(open(DELIVERY_PATH))
        close = self.cfg["gripper_close_cmd"]
        # 1) AUTO up-clearance: lift the held object to the high 'travel' pose FIRST so the
        #    carry clears the other items (e.g. perfume) instead of dragging through them.
        travel = (self.loc_map or {}).get("travel_pose")
        if travel:
            print("[grasp] delivery: lifting to travel height to clear obstacles")
            self._glide_to({**travel, "gripper.pos": close}, duration_s=2.0)
        # 2) carry through the saved waypoints (these just bring the arm OVER the zone;
        #    the exact set-down height is found by feel, not by a precise recording)
        print(f"[grasp] scripted delivery: {len(waypoints)} waypoints -> zone")
        for i, wp in enumerate(waypoints, 1):
            self._glide_to({**wp, "gripper.pos": close}, duration_s=2.0)  # hold object
            print(f"[grasp]   waypoint {i}/{len(waypoints)} reached")
        # 3) lower until the object touches the table (torque feedback), then release
        if os.environ.get("BASEER_SETDOWN_CONTACT", "1") == "1":
            self.lower_until_contact()
        self.open_gripper()                                        # release once resting
        return True

    # -- one policy attempt -------------------------------------------------
    def _run_policy(self, task, max_seconds, grasp_check_after_close=True):
        """Roll the policy. Detect the close event; once the gripper has closed and
        settled, verify the grasp. Returns 'held' | 'empty' | 'timeout'."""
        self.policy.reset()
        debug = os.environ.get("BASEER_GRASP_DEBUG") == "1"
        opened_seen = False
        grip_hist = []          # recent (t, gripper_pos) while policy commands a close
        t0 = time.perf_counter()
        n = 0
        infer_t_sum = 0.0
        last_state = None
        last_log = t0
        while time.perf_counter() - t0 < max_seconds:
            loop_t = time.perf_counter()
            try:
                obs = self.robot.get_observation()    # skip transient camera hiccups, don't crash
            except Exception as e:
                print(f"[grasp] camera hiccup, skipping frame: {e}")
                time.sleep(0.03)
                continue
            # aggregate per-motor floats + camera into the dataset-frame the policy expects
            obs_frame = build_dataset_frame(self.obs_features, obs, prefix=OBS_STR)
            it = time.perf_counter()
            action = predict_action(
                obs_frame, self.policy, self.device, self.pre, self.post,
                use_amp=False, task=task, robot_type=self.robot.name,
            )
            infer_t_sum += time.perf_counter() - it
            act = action.squeeze(0).detach().cpu()      # drop batch dim -> (6,)
            action_dict = {k: act[i].item() for i, k in enumerate(self.action_keys)}
            self.robot.send_action(action_dict)
            n += 1

            g = float(obs.get("gripper.pos", self.cfg["gripper_open_cmd"]))
            g_cmd = action_dict["gripper.pos"]

            # --- diagnostics: ~1/s, show loop rate, arm motion, gripper, target ---
            if debug and time.perf_counter() - last_log >= 1.0:
                state = np.array([obs[k] for k in self.action_keys])
                moved = 0.0 if last_state is None else float(np.abs(state - last_state).max())
                last_state = state
                hz = n / (time.perf_counter() - t0)
                print(f"[dbg] t={time.perf_counter()-t0:4.1f}s hz={hz:4.1f} "
                      f"infer={infer_t_sum/n*1000:4.0f}ms armΔ={moved:5.1f}° "
                      f"grip_now={g:5.1f} grip_cmd={action_dict['gripper.pos']:5.1f} "
                      f"opened_seen={opened_seen}")
                last_log = time.perf_counter()

            # Detect the grasp moment. A HELD object keeps the gripper open at its width
            # (~held_pos) even while the policy commands it shut — so we must NOT wait for
            # the gripper to reach fully-closed. Instead: once the policy COMMANDS a close
            # and the gripper position SETTLES, judge with grasp_ok (pos + current).
            if grasp_check_after_close:
                if g > self.cfg["pos_threshold"] + 15:
                    opened_seen = True
                closing = g_cmd < self.cfg["pos_threshold"]        # policy wants it shut
                if opened_seen and closing:
                    now = time.perf_counter()
                    grip_hist.append((now, g))
                    grip_hist = [(t, v) for (t, v) in grip_hist if now - t <= 0.6]
                    span = now - grip_hist[0][0]
                    vals = [v for _, v in grip_hist]
                    if span >= 0.4 and (max(vals) - min(vals)) < 4:  # closed & settled
                        return "held" if self.grasp_ok() else "empty"
                else:
                    grip_hist = []        # policy re-opened mid-attempt -> reset

            dt = time.perf_counter() - loop_t
            if (sleep := 1.0 / self.fps - dt) > 0:
                time.sleep(sleep)
        if debug:
            print(f"[dbg] window ended: {n} steps, avg infer "
                  f"{infer_t_sum/max(n,1)*1000:.0f}ms, avg loop {n/(time.perf_counter()-t0):.1f}Hz")
        return "timeout"

    # -- optional vision confirmation (Method 2) ---------------------------
    def _object_gone(self, item_name):
        """Vision cross-check via Fanar-Oryx: True if `item_name` is no longer on the
        table (i.e. it's now in the claw). Off unless BASEER_GRASP_VISION=1."""
        if os.environ.get("BASEER_GRASP_VISION") != "1" or not item_name:
            return None  # unknown / disabled -> caller ignores it
        try:
            import cv2
            from agent.agent3_vision import describe_scene
            import tools as T
            frame = self.robot.get_observation().get("front")
            if frame is None:
                return None
            _ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            present = describe_scene(buf.tobytes(), T.PRODUCTS)
            gone = item_name not in present
            print(f"[grasp] vision: '{item_name}' on table={not gone} -> "
                  f"{'GONE (grasped)' if gone else 'still there'}")
            return gone
        except Exception as e:
            print(f"[grasp] vision check skipped: {e}")
            return None

    # -- localization: Oryx finds the object, map -> hover pose above it ----
    @staticmethod
    def _clean(label):
        return label.strip().strip("<>").strip()

    def _locate_pixel(self, item_name):
        """Use Fanar-Oryx to find `item_name` in the live frame. Robust: query the
        (good-recall) all-items detector `samples` times, MEDIAN the target's box center,
        tolerate misses. Returns (u, v) or None."""
        try:
            import statistics
            import cv2
            from agent.agent3_vision import locate_scene
            import tools as T
            frame = np.asarray(self.robot.get_observation()["front"])
            h, w = frame.shape[:2]
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            jpg = buf.tobytes()
            samples = int(os.environ.get("BASEER_LOCATE_SAMPLES", "3"))
            centers, last_seen = [], []
            for _ in range(samples):
                boxes = locate_scene(jpg, w, h, T.PRODUCTS)
                last_seen = [self._clean(b["label"]) for b in boxes]
                cand = [b for b in boxes if self._clean(b["label"]) == item_name] or \
                       [b for b in boxes if item_name in self._clean(b["label"])
                        or self._clean(b["label"]) in item_name]
                if cand:
                    x1, y1, x2, y2 = cand[0]["box"]
                    centers.append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
            if not centers:
                print(f"[grasp] Oryx could NOT locate '{item_name}' in {samples} tries "
                      f"(last saw: {last_seen})")
                return None
            u = statistics.median(c[0] for c in centers)
            v = statistics.median(c[1] for c in centers)
            print(f"[grasp] Oryx located '{item_name}' at ({round(u)},{round(v)}) "
                  f"[{len(centers)}/{samples} hits]")
            return (u, v)
        except Exception as e:
            print(f"[grasp] localization detect failed: {e}")
            return None

    def _hover_joints(self, u, v):
        """Map an object pixel (u,v) to the calibrated hover-pose joints."""
        W, H = self.loc_map["W"], self.loc_map["H"]
        # warn if the pixel is OUTSIDE the calibrated sample range -> the quadratic map
        # EXTRAPOLATES there and the hover pose can be badly wrong (recalibrate wider).
        rng = self.loc_map.get("px_range")
        if rng and not (rng["umin"] <= u <= rng["umax"] and rng["vmin"] <= v <= rng["vmax"]):
            print(f"[grasp] ⚠ object pixel ({round(u)},{round(v)}) is OUTSIDE the calibrated "
                  f"range u[{rng['umin']}-{rng['umax']}] v[{rng['vmin']}-{rng['vmax']}] — "
                  f"hover pose is extrapolated and may be off. Recalibrate covering this area.")
        un, vn = u / W, v / H
        feats = np.array([1.0, un, vn, un * un, un * vn, vn * vn])
        coeffs = np.array(self.loc_map["coeffs"])          # (6 feats, 6 joints)
        vals = feats @ coeffs
        return {k: float(vals[i]) for i, k in enumerate(self.loc_map["keys"])}

    def prereach(self, item_name):
        """Pre-position the arm above the Oryx-located object (gripper open), so the
        policy only has to do the short final descent+grasp. Returns True if it moved.

        Approaches FROM ABOVE — rotates over the target at the raised 'home' height first,
        then descends — so the gripper doesn't sweep across the table into other objects.
        OFF by default (policy runs alone — the behavior that grasped reliably); the
        pixel->hover localization is opt-in via BASEER_PREREACH=1 while it's being tuned."""
        if os.environ.get("BASEER_PREREACH", "0") != "1":
            print("[grasp] pre-reach OFF (default) — policy runs alone. "
                  "Set BASEER_PREREACH=1 to enable Oryx pre-positioning (still being tuned).")
            return False
        if not self.loc_map:
            print("[grasp] NO localization map — skipping pre-position (policy runs alone). "
                  "Run backend/robot/calibrate_localization.py to enable localization.")
            return False
        if not item_name:
            print("[grasp] no --item given — skipping localization (pass e.g. --item 'سيروم الشعر').")
            return False
        uv = self._locate_pixel(item_name)
        if uv is None:
            return False
        target = self._hover_joints(*uv)
        target["gripper.pos"] = self.cfg["gripper_open_cmd"]   # approach with gripper open
        # TOP-DOWN approach to avoid sweeping into neighbors:
        #   1) raise to the captured high 'travel' pose, 2) rotate over the target while
        #   staying high, 3) descend onto the hover pose. Falls back to home height if no
        #   travel pose was captured.
        high = dict(self.loc_map.get("travel_pose") or self.home_pose)
        over = dict(high)
        over["shoulder_pan.pos"] = target["shoulder_pan.pos"]   # rotate toward target, stay high
        for d in (high, over):
            d["gripper.pos"] = self.cfg["gripper_open_cmd"]
        print(f"[grasp] pre-reaching above '{item_name}' (raise -> over -> descend)")
        self._glide_to(high, duration_s=1.8)     # raise straight up to travel height
        self._glide_to(over, duration_s=1.5)     # rotate over the target, still high
        self._glide_to(target, duration_s=1.5)   # descend onto the hover pose
        return True

    def announce_scene(self):
        """Oryx perception readout: print what's on the table before grasping (the
        'Oryx announces' step). Off via BASEER_ANNOUNCE=0. Does NOT drive motion."""
        if os.environ.get("BASEER_ANNOUNCE", "1") != "1":
            return
        try:
            import cv2
            from agent.agent3_vision import describe_scene
            import tools as T
            frame = np.asarray(self.robot.get_observation()["front"])
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            items = describe_scene(buf.tobytes(), T.PRODUCTS)
            print(f"[oryx] sees on the table: {items if items else '(none recognized)'}")
        except Exception as e:
            print(f"[oryx] scene announce skipped: {e}")

    # -- public: pick with retries -----------------------------------------
    def pick(self, task, attempts=3, attempt_seconds=25, deliver_seconds=15, item_name=None):
        """Try to grasp (and deliver) up to `attempts` times. Returns True on success.

        Demo mode: Oryx ANNOUNCES the scene (perception), then the policy grasps on its
        own (reliable, no pushing). Pre-reach localization is opt-in (BASEER_PREREACH=1).
        """
        self.announce_scene()
        for i in range(1, attempts + 1):
            print(f"\n[grasp] === attempt {i}/{attempts} ===")
            if i > 1:
                self.open_gripper()
                self.go_home()
            self.prereach(item_name)        # Oryx-localize + move above object (no-op if unavailable)
            result = self._run_policy(task, attempt_seconds)
            held = result == "held"
            if held:
                gone = self._object_gone(item_name)        # None if disabled/unknown
                if gone is False:                          # claw closed but object still on table
                    print("[grasp] torque said held but vision says object still on table — retry.")
                    held = False
            if held:
                print("[grasp] object secured — completing delivery.")
                # Prefer the deterministic scripted delivery (fixed zone); if no
                # waypoints saved, fall back to letting the policy carry + place.
                if not self.deliver_to_zone():
                    self._run_policy(task, deliver_seconds, grasp_check_after_close=False)
                    self.open_gripper()                   # release at the zone
                return True
            print(f"[grasp] attempt {i} failed ({result}); retrying…" if i < attempts
                  else "[grasp] all attempts failed.")
        return False

    def close(self):
        if self._owns_robot and self.robot.is_connected:
            self.go_home()
            self.robot.disconnect()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", default="follower_so100")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--task", required=True)
    ap.add_argument("--item", default=None,
                    help="catalog name for Oryx localization, e.g. 'سيروم الشعر' "
                         "(enables pre-positioning above the object)")
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--attempt-seconds", type=int, default=25,
                    help="max seconds per grasp attempt (raise if the pick gets cut off)")
    ap.add_argument("--deliver-seconds", type=int, default=15)
    args = ap.parse_args()

    gc = GraspController(args.policy, args.port, args.id, args.cam)
    try:
        ok = gc.pick(args.task, attempts=args.attempts,
                     attempt_seconds=args.attempt_seconds,
                     deliver_seconds=args.deliver_seconds,
                     item_name=args.item)
        print("\nRESULT:", "DELIVERED ✅" if ok else "GAVE UP ❌")
    finally:
        gc.close()


if __name__ == "__main__":
    main()
