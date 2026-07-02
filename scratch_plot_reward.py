"""Reward-trajectory chart with linear trend. Canonical per-step reward (last value per step
across resumes) + least-squares fit. Colors from the validated dataviz reference palette
(series-1 blue, series-8 orange); light surface. -> output/fugu_ultra_lcb/reward_trajectory.png"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("/tmp/reward_traj.json"))
steps, reward, slope, intercept = d["steps"], d["reward"], d["slope"], d["intercept"]
trend = [slope * s + intercept for s in steps]
mean = sum(reward) / len(reward)

# palette (validated reference, light mode)
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"

fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)

# 0.5 "valid-but-wrong" reference floor (context for the 0/0.5/1 reward)
ax.axhline(0.5, color=GRID, lw=1.0, zorder=1)
ax.text(steps[-1], 0.5, " valid-but-wrong floor (0.5)", color=MUTED, fontsize=7.5,
        va="center", ha="left")

# #2 intervention marker
ax.axvline(80, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
ax.text(80, 0.955, "redundancy penalty (#2)\nresume @ step 80", color=MUTED, fontsize=7.5,
        va="top", ha="center")

# reward series (thin line + faint per-step markers)
ax.plot(steps, reward, color=BLUE, lw=1.4, alpha=0.9, zorder=3, label="reward / mean (per step)")
ax.plot(steps, reward, color=BLUE, marker="o", ms=2.6, ls="none", alpha=0.55, zorder=3)

# linear trend
ax.plot(steps, trend, color=ORANGE, lw=2.2, ls=(0, (6, 3)), zorder=4,
        label=f"linear trend  (+{slope*len(steps):.3f} over {len(steps)} steps,  slope {slope:+.5f}/step)")

ax.set_xlim(-2, steps[-1] + 2)
ax.set_ylim(0.42, 0.98)
ax.set_xlabel("GRPO training step", color=INK2, fontsize=10)
ax.set_ylabel("reward  (0 unparseable / 0.5 valid-wrong / 1.0 correct)", color=INK2, fontsize=9.5)
ax.set_title("Fugu-Ultra Conductor — GRPO training reward trajectory", color=INK, fontsize=13,
             fontweight="bold", loc="left", pad=34)

ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#c3c2b7")
ax.tick_params(colors=MUTED, labelsize=9)
for lbl in ax.get_xticklabels() + ax.get_yticklabels():
    lbl.set_color(INK2)

leg = ax.legend(loc="lower right", frameon=True, fontsize=8.5, facecolor=SURF,
                edgecolor=GRID, framealpha=1.0)
for t in leg.get_texts():
    t.set_color(INK2)

# subtitle: the takeaway, in ink not color (short, clears the title + right edge)
ax.text(0, 1.02, f"flat — mean {mean:.3f}, +0.02 over 124 steps  (training reward is not the verdict)",
        transform=ax.transAxes, color=MUTED, fontsize=9, va="bottom")

fig.tight_layout()
out = "output/fugu_ultra_lcb/reward_trajectory.png"
fig.savefig(out, facecolor=SURF, bbox_inches="tight")
print(f"saved {out}  ({len(steps)} steps, mean={mean:.3f}, slope={slope:+.5f})")
