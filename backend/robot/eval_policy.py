"""
Real-world success-rate eval for a grasp policy — used to compare SmolVLA vs pi0 vs GR00T
on the SAME arm/scene with the SAME protocol.

For each trial: you place the serum, press Enter; the policy attempts the localized grasp
(via GraspController); you confirm whether it actually delivered (y/n — ground truth).
At the end it prints the success rate and appends a row to results/<label>.json.

    python backend/robot/eval_policy.py \
        --policy ~/baseer/policy_vla/pretrained_model --label smolvla \
        --port /dev/tty.usbmodem5AB90677591 --id follower_so100 \
        --task "Pick up the hair serum and place it in the delivery zone" \
        --item "سيروم الشعر" --trials 10 --attempts 1

Run it once per policy (--label smolvla / pi0 / groot), then compare the JSON results.
Use --attempts 1 to measure the RAW policy; raise it to include the retry system.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on path
from agent.agent4_grasp import GraspController

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--label", required=True, help="smolvla | pi0 | groot")
    ap.add_argument("--port", required=True)
    ap.add_argument("--id", default="follower_so100")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--task", required=True)
    ap.add_argument("--item", default=None)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--attempts", type=int, default=1)
    args = ap.parse_args()

    gc = GraspController(args.policy, args.port, args.id, args.cam)
    trials = []
    try:
        for i in range(1, args.trials + 1):
            cmd = input(f"\n=== Trial {i}/{args.trials} ({args.label}) === "
                        "place the serum, Enter to run (s=skip, q=quit): ").strip().lower()
            if cmd == "q":
                break
            if cmd == "s":
                continue
            auto = gc.pick(args.task, attempts=args.attempts, item_name=args.item)
            human = input("  Did it actually deliver the serum? (y/n): ").strip().lower().startswith("y")
            trials.append({"trial": i, "auto_held": bool(auto), "delivered": human})
            print(f"  -> auto_held={auto}  delivered={human}")
    finally:
        gc.close()

    n = len(trials)
    succ = sum(1 for t in trials if t["delivered"])
    auto = sum(1 for t in trials if t["auto_held"])
    print("\n" + "=" * 50)
    print(f"POLICY: {args.label}   trials={n}")
    if n:
        print(f"  delivered (ground truth): {succ}/{n} = {100*succ/n:.0f}%")
        print(f"  auto-HELD (gripper sig):  {auto}/{n} = {100*auto/n:.0f}%")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, f"{args.label}.json")
    json.dump({"label": args.label, "policy": args.policy, "attempts": args.attempts,
               "trials": trials, "n": n, "delivered": succ, "auto_held": auto}, open(out, "w"), indent=2)
    print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
