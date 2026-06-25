"""
Compare the robot's LIVE joint state to the TRAINING data distribution. If joints
read outside the training min/max, the current calibration differs from when the
data was recorded -> the policy sees an out-of-distribution state and flails.

Put the arm roughly where episodes STARTED (both serums on the table, arm at its
usual rest pose), then run:

    /opt/anaconda3/envs/lerobot/bin/python check_state_match.py \
        --port /dev/tty.usbmodem5AB90677591 --id follower_so100
"""
import argparse
import json
import os

from lerobot.robots.so_follower import SO100Follower, SO100FollowerConfig

L = os.path.expanduser("~/.cache/huggingface/lerobot/55CancriE/baseer_serums_20260623_204039")
KEYS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", default="follower_so100")
    args = ap.parse_args()

    stats = json.load(open(os.path.join(L, "meta/stats.json")))["observation.state"]
    mn, mx, mean = stats["min"], stats["max"], stats["mean"]

    robot = SO100Follower(SO100FollowerConfig(port=args.port, id=args.id, cameras={}))
    robot.connect()
    obs = robot.get_observation()
    robot.disconnect()
    live = [float(obs[f"{k}.pos"]) for k in KEYS]

    print(f"{'joint':14s} {'live':>8s} {'train_min':>10s} {'train_max':>10s} {'train_mean':>11s}  status")
    bad = []
    for i, k in enumerate(KEYS):
        lo, hi = mn[i], mx[i]
        margin = (hi - lo) * 0.15            # allow a little slack outside the range
        ok = (lo - margin) <= live[i] <= (hi + margin)
        if not ok:
            bad.append(k)
        print(f"{k:14s} {live[i]:8.1f} {lo:10.1f} {hi:10.1f} {mean[i]:11.1f}  "
              f"{'ok' if ok else 'OUT OF RANGE ⚠'}")
    if bad:
        print(f"\n⚠ {len(bad)} joint(s) out of training range: {bad}")
        print("  -> calibration likely differs from recording. Re-record with THIS")
        print("     calibration, or restore the calibration used when recording.")
    else:
        print("\n✅ all joints within training range — calibration matches the data.")
        print("   If it still fails, the issue is the CAMERA VIEW (run capture_policy_view.py).")


if __name__ == "__main__":
    main()
