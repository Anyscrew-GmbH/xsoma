"""XSoma training: a world model of the robot and a policy trained by backpropagation through imagined rollouts in it, on
Holosoma's G1 velocity-tracking task.

One iteration:
  1. collect `env_steps` steps of all envs into a replay buffer (sim.Sim). The policy acts with Gaussian noise added, except in
     a share `det_share` of the envs, which act without noise;
  2. fit the world model `wm_batches` times on `wm_window`-step windows of real transitions. It predicts a window open loop:
     from the window's first state only, fed with the PD targets that were applied;
  3. update the policy `pi_updates` times: from replayed states, roll the policy out `horizon` steps in the world model and
     ascend the discounted sum of Holosoma's rewards along it. Observations, rewards and terminations of the imagined states
     are the task's own functions (task.py) on forward kinematics and the exact terrain surface (sim.Sim.view_state), so the
     gradient flows through the world model, the policy and the reward alike. There is no critic.
The three parts are separate programs, compiled ahead of time and in parallel. Each takes the whole training state and updates it
in place (donated buffers). The host only starts them and fetches statistics every `log_every` iterations. Checkpoints record
the training time and the wall-clock time since launch.

With `--init_tracker <checkpoint> --tracker_meta <meta.json>` the policy is a motion tracker's network (tracker.py), driven by
the task's command and fine-tuned by the same updates; `--anchor` penalizes its distance from the initial tracker's actions and
`--gait_clock 0` drops the reward term that follows the gait clock, which a tracker does not observe. `--tracker_noise` scales
the noise on the tracker's observation (1: as in the tracker's own training) and `--tracker_heading rate` relates that
observation to the base's own yaw instead of a held heading. `--feet_cross`, for any policy, weighs a penalty on the feet
crossing sideways in imagination's reward. For a robot with `self_collision` groups (task.py), `--self_clearance` weighs a
penalty on the two groups coming within 2 cm of each other, and `--self_clearance_term <m>` ends an imagined rollout once they
overlap by more than that depth; both act in imagination only. `--foot_clearance`, for any policy, weighs a penalty in
imagination's reward on a foot that moves horizontally lower than `--foot_clearance_height` over the terrain. `--cmd_scale x,y,z`
scales the command ranges of vx, vy and wz, and `--push_vel` sets the size of the pushes in collection (0: none).
`--term_penalty` charges an imagined fall a fixed amount, `--fall_starts` starts a share of the imagined rollouts shortly
before a real fall (`--fall_lead` steps at most), and `--tilt_term <rad>` ends an imagined rollout at that tilt of the base already.
`--ext_joints name:lo:hi,...` hands joints to something else than the policy while the command is zero: in a share `--ext_share`
of the commands their targets follow random goals (task.ext_step), in collection and in imagination, and imagination's pose and
action-rate penalties leave them out. `--stand_still` weighs a penalty on any motion of the base and the feet under a zero
command (task.stand_still), in imagination only. A fourth entry of `--cmd_scale` is the share of zero commands. With
`--ext_blind 1` a tracker policy is not shown those joints while they are driven from outside (`XSoma.hide`). `--track_l1`
weighs a penalty on the plain distance between the command and the base's velocity (task.track_l1), in imagination only;
`--track_rel` takes that distance relative to the command's size. `--torque_limit` weighs a penalty on the PD torque that an
imagined action asks for beyond `--torque_soft` x each joint's effort limit (task.torque_limit), in imagination only.
`--range_penalty` weighs a penalty on the joints of `--range_joints name:lo:hi,...` outside those ranges (task.joint_range), in
imagination only; `--range_target` weighs the same penalty on the PD targets that the imagined actions ask for. Sixth and
seventh entries of `--cmd_scale` scale a share of the commands down (slow commands).
`--touchdown_speed` weighs a penalty on a foot's downward speed lower than `--touchdown_height` over the terrain
(task.touchdown_speed) and `--impact_accel` one on the base's upward acceleration (task.impact_accel), both in imagination only.
`--turn_step` weighs a penalty, scaled by the yaw-rate command, on a foot that stands longer than `--turn_step_stance` seconds
(task.turn_step; stance from the feet's heights in the imagined states), in imagination only; with `--stance_carry 1` each imagined
rollout starts its stance clocks from collection's (kept in the replay buffer) instead of zero, so a `--turn_step_stance` near or
above the horizon's length can be reached. `--friction_surfaces lo:hi,...`
draws each episode's friction coefficient from a surface picked uniformly, then uniformly in its range (sim.Sim._mu).
`--joint_vel` weighs a penalty on the squared joint speeds of the imagined states (task.joint_vel), in imagination only. With a
tracker policy, `--kl_bound <eps>` keeps the trained policy near the initial one: the loss gains kl_beta x max(0, KL - eps), KL
being that between the two policies' action distributions on the imagined states, and with `--kl_beta_lr` the weight adapts
by dual ascent (XSoma.imagine, XSoma.policy_update). With a tracker policy, `--symmetry` weighs the squared distance between the
action for the left/right mirror image of an imagined state and the mirror image of the action for that state (tracker.mirror).
`--upper_tilt` weighs a penalty on the sideways tilt in the world of the body `--upper_body` (task.upper_tilt) and `--upper_rate` one
on the rate of that tilt (task.upper_rate), both in imagination only, the body's pose by forward kinematics of the imagined states.
`--start_share` starts a share of the episodes in another joint pose (`--start_joints name:lo:hi,...`, offsets from the default
pose) and, with `--start_cmd`, with a command from given ranges (sim.Sim._fresh).

The file holds `Config` (every option, also the command-line flags), `XSoma` (the simulator, the networks and the programs of
one iteration) and `main` (the loop with logging, checkpoints and the final evaluation).
"""

from __future__ import annotations

import argparse
import dataclasses
import gc
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from . import buffer as rb
from . import nets, optim, task, tracker, wandb_log
from .kin import Kinematics
from .sim import Sim, SimCfg, surfaces, unflatten


@dataclasses.dataclass
class Config:
    """Every training option; each field is also a command-line flag of the same name. The defaults are the published ones."""

    # task and simulator (sim.SimCfg)
    robot_dir: str = task.ROBOT_DIR or ""  # another robot's directory (task.py); "": the G1
    num_envs: int = 4096
    terrain: str = "mix"  # mix: Holosoma's terrain_locomotion_mix | plane
    terrain_scale: float = 0.5  # Holosoma's terrain scale factor; 0.5 as in its MuJoCo Warp runs (1.0 trains too, see scene.py)
    foot: str = "capsules"  # capsules: Holosoma's MJCF soles | box: one box per sole
    preset: str = "fast_sac"  # Holosoma's reward and penalty-curriculum preset: fast_sac (g1_29dof_fast_sac) | ppo (g1_29dof);
    #                           they differ in the penalty curriculum and in the alive weight; the larger one (10, against 1)
    #                           makes a policy that looks 1 s ahead value not falling
    sole_friction: int = 1  # friction randomization reaches the soles (Holosoma's MuJoCo backends keep them at 0.8)
    random_start: int = 1  # episode counters start uniform in [0, 1000), so episodes end at different times (as Holosoma's PPO)
    cmd_scale: str = ""  # "x,y,z": factors on the command ranges of vx, vy and wz, in collection and in imagination, e.g.
    #                      "1,0.5,1" for sideways commands up to 0.5 m/s ("": Holosoma's U[-1, 1]); the evaluation after training
    #                      keeps Holosoma's ranges (evaluate.py --cmd_scale). "x,y,z,p": p is the share of zero commands (Holosoma's 0.2);
    #                      "x,y,z,p,q": each component of a command is also zero with probability q
    #                      "x,y,z,p,q,s,f": a share s of the commands is also scaled by f (slow commands)
    push_vel: float = task.PUSH_VEL  # m/s: a push in collection sets the root's (vx, vy) to U[-push_vel, push_vel] each (0: none); the
    #                                  evaluation keeps Holosoma's pushes
    # schedule
    minutes: float = 15.0  # training time, start-up excluded
    env_steps: int = 1  # env steps of all envs per iteration
    buffer_steps: int = 200  # replay buffer length in env steps (x num_envs transitions)
    warmup_steps: int = 32  # env steps collected before the first iteration
    wm_only_iters: int = 200  # first iterations, which fit only the world model (no policy updates)
    # world model
    wm_hidden: str = "512,512,512"  # hidden layer widths
    wm_window: int = 8  # steps per training window; the model predicts them open loop
    wm_batches: int = 3  # gradient steps per iteration
    wm_batch: int = 2048  # windows per gradient step
    wm_lr: float = 4e-4
    wm_muon: int = 1  # Muon on the hidden weight matrices, Adam elsewhere (BlinkMimic's MuonAdam)
    wm_gate: float = 2.0  # skip an iteration's policy updates while the world-model loss exceeds wm_gate x its running average
    wm_eval: int = 1  # once per log interval, check the world model open loop on 32-step windows of real data
    # policy
    horizon: int = 48  # imagined steps per update (0.96 s)
    pi_batch: int = 2048  # imagined rollouts per update
    pi_updates: int = 2  # policy updates per iteration
    pi_lr: float = 3e-4
    pi_clip: float = 1.0  # gradient-norm clip
    pi_gn_gate: float = 10.0  # skip an update whose gradient norm exceeds pi_gn_gate x the running average
    pi_ema: float = 0.995  # the evaluated policy is an exponential moving average of the trained one
    gamma: float = 0.99  # discount of the imagined return
    action_noise: float = 0.3  # std of the collected actions' Gaussian exploration noise
    det_share: float = 0.5  # share of envs that collect without noise, spread over the terrain; when every collected action is
    #                         noisy, policies trained for about 45 minutes or more come to rely on the noise and fall without it
    cmd_fresh: float = 0.5  # share of imagined rollouts that start with a freshly drawn command
    term_penalty: float = 0.0  # subtracted from an imagined rollout's return at the step that ends it in a fall (discounted as a
    #                            reward of that step). Without it a fall costs only the rewards of the horizon's remaining steps,
    #                            which is next to nothing where the penalties cancel the alive reward
    fall_starts: float = 0.0  # share of the imagined rollouts that start in the `horizon` steps before a real fall of their env
    #                           (while the replay buffer holds one), instead of anywhere; uniform starts hit such a state rarely
    fall_lead: int = 0  # with fall_starts: those rollouts start up to this many steps before the fall (0: `horizon` steps)
    tilt_term: float = 0.0  # > 0: an imagined rollout also ends, as in a fall, once the base's up axis is tilted by more than this
    #                         many radians from the vertical (0: off). A falling robot passes such a tilt some tenths of a second
    #                         before it touches the ground, so more of the rollouts that lead to a fall end within the horizon; the
    #                         simulator ends no episode for it
    term_tau: float = 0.02  # width (m) of the sigmoid that gives imagined terminations their gradient; their value stays exact
    init_tracker: str = ""  # a tracker checkpoint (tracker.py): the policy is the tracker's network; its encoder and decoder are
    #                         trained, its normalizers stay as loaded. "": XSoma's own policy. Layout 1GPU only
    tracker_meta: str = ""  # with init_tracker: the meta.json of the tracker's robot, which must be the task's (tracker.load)
    anchor: float = 0.0  # with init_tracker: weight of the mean squared distance between the imagined actions and the actions of
    #                      the initial tracker on the same inputs (0: not computed)
    gait_clock: int = 1  # 0: imagination's reward leaves out feet_phase, the term that follows the gait clock (a tracker does not
    #                      observe the clock); the simulator's logged rewards and the evaluation keep the term
    tracker_noise: float = 1.0  # with init_tracker: scale of the uniform noise on the tracker's observation in collection and in
    #                             imagination; 1: the scales of the tracker's own training (tracker.NOISE), 0: no noise
    tracker_heading: str = "hold"  # with init_tracker, the heading that the tracker's observation refers to. hold: the base's yaw
    #                                at the episode's start plus the integrated commanded yaw rate | rate: the base's own yaw at
    #                                every step, so the observation carries the tilt but no yaw error
    feet_cross: float = 0.0  # weight of the penalty task.feet_cross on the feet crossing sideways, in imagination's reward only,
    #                          times dt and the penalty curriculum's scale as Holosoma's penalties (0: not computed); the
    #                          evaluation reports the crossing whatever the weight
    self_clearance: float = 0.0  # weight of the penalty task.self_clearance on the two `self_collision` groups (task.py) coming
    #                              closer than task.SELF_CLEARANCE_MARGIN, pair by pair of their colliders (sim.SelfDistance), in
    #                              imagination's reward only and weighted as feet_cross is (0: not computed)
    self_clearance_term: float = 0.0  # > 0: an imagined rollout also ends, as in a fall, at a step in which a pair of those
    #                                   colliders overlaps by more than this many meters (0: off); the simulator ends no episode
    foot_clearance: float = 0.0  # weight of the penalty task.foot_clearance on a foot that moves horizontally while its lowest
    #                              sole point is less than foot_clearance_height over the terrain, in imagination's reward only
    #                              and weighted as feet_cross is (0: not computed); it needs no gait clock
    foot_clearance_height: float = task.FOOT_CLEARANCE_HEIGHT  # m
    ext_joints: str = ""  # "name:lo:hi,...": every joint whose name contains `name` is driven by something else than the policy
    #                       while the command is zero, in a share `ext_share` of the commands: its target moves between random
    #                       goals in [lo, hi] rad (task.ext_step), starting from the policy's own target. The policy sees those
    #                       targets as its previous action. "": the policy drives every joint
    ext_share: float = 0.5  # with ext_joints: drawn per env with every command
    ext_blind: int = 0  # with ext_joints and init_tracker, 1: while those joints are driven from outside the policy is fed their
    #                     default pose, zero velocity and its own last action for them, as a runtime that hides them does
    track_l1: float = 0.0  # weight of the penalty task.track_l1, the distance between the command and the base's planar velocity
    #                        and yaw rate without the tracking rewards' tolerance, in imagination's reward only and weighted as
    #                        feet_cross is (0: not computed)
    track_rel: float = 0.0  # with track_l1: the error is divided by the size of the command (the norm of its three components,
    #                         at least this; not for a zero command), so that slow commands weigh as much as fast ones (0: off)
    stand_still: float = 0.0  # weight of the penalty task.stand_still on the base and the feet moving under a zero command, in
    #                           imagination's reward only and weighted as feet_cross is (0: not computed)
    torque_limit: float = 0.0  # weight of the penalty task.torque_limit on the PD torque that the imagined action asks for beyond
    #                            torque_soft x the effort limit, in imagination's reward only and weighted as feet_cross is (0: not computed)
    torque_soft: float = 0.9  # with torque_limit: the share of the effort limit from which on the torque is penalized
    range_penalty: float = 0.0  # weight of the penalty task.joint_range on the joints of range_joints outside their ranges, in
    #                             imagination's reward only and weighted as feet_cross is (0: not computed)
    range_target: float = 0.0  # weight of the same penalty on the PD targets of the joints of range_joints (target spikes that the
    #                            pose barely shows), in imagination's reward only and weighted as feet_cross is (0: not computed)
    range_joints: str = ""  # with range_penalty or range_target: "name:lo:hi,...": every joint whose name contains `name` is
    #                         penalized outside [lo, hi] rad (e.g. limits that the robot's runtime enforces inside the model's joint ranges)
    touchdown_speed: float = 0.0  # weight of the penalty task.touchdown_speed on a foot moving down while its lowest sole point is
    #                               less than touchdown_height over the terrain, in imagination's reward only and weighted as
    #                               feet_cross is (0: not computed)
    touchdown_height: float = task.TOUCHDOWN_HEIGHT  # m
    turn_step: float = 0.0  # weight of the penalty task.turn_step (|yaw-rate command| x a foot's stance beyond turn_step_stance), in
    #                         imagination's reward only and weighted as feet_cross is (0: not computed). Each imagined rollout starts
    #                         its stance clocks at zero (the replay buffer holds no contact history)
    turn_step_stance: float = 0.4  # s
    walk_step: float = 0.0  # weight of the penalty task.walk_step (a foot's stance beyond turn_step_stance under any nonzero command),
    #                         in imagination's reward only and weighted as feet_cross is (0: not computed)
    stance_carry: int = 0  # with turn_step or walk_step, 1: each imagined rollout starts its stance clocks from those of collection
    #                        (each foot's stance under the current command, task.stance_clock; kept in the replay buffer; zero for a
    #                        start whose command imagination redraws) instead of zero, which caps the stance it can see at the horizon
    balance: float = 0.0  # weight of the penalty task.balance (pelvis beyond balance_margin from the segment between the feet), in
    #                       imagination's reward only and weighted as feet_cross is (0: not computed)
    balance_margin: float = 0.03  # m
    step_length: float = 0.0  # weight of the penalty task.step_length (feet spread beyond step_base + step_per_speed x |command|),
    #                           in imagination's reward only and weighted as feet_cross is (0: not computed)
    step_base: float = 0.10  # m
    step_per_speed: float = 0.5  # s: the allowed spread grows by this times the commanded speed
    step_width: float = 0.20  # m: the nominal lateral distance of the feet, not charged
    friction_surfaces: str = ""  # "lo:hi,lo:hi,...": per episode a surface uniformly, then the friction coefficient U[lo, hi] of it,
    #                              for every geom (the soles with sole_friction); "": task.FRICTION once per env (sim.SimCfg)
    joint_vel: float = 0.0  # weight of the penalty task.joint_vel, the sum of the squared joint speeds (rad/s) of the state an imagined
    #                         step ends in, in imagination's reward only and weighted as feet_cross is (0: not computed)
    kl_bound: float = 0.0  # with init_tracker, > 0: the bound eps on KL, the mean over the imagined states of the KL divergence between
    #                        the policy's action distribution and the initial tracker's, both Gaussians around their actions with the
    #                        exploration std action_noise, so KL = |a - a0|^2 / (2 action_noise^2), on the same inputs held constant
    #                        (as the anchor's); the loss gains kl_beta x max(0, KL - eps) (0: off)
    kl_beta: float = 1.0  # with kl_bound: the weight of that term (its starting value with kl_beta_lr)
    kl_beta_lr: float = 0.0  # with kl_bound, > 0: dual ascent on the weight after every accepted policy update, kl_beta <- max(0,
    #                          kl_beta + kl_beta_lr x clip(KL / eps - 1, -1, 1)): it grows while KL is above the bound and shrinks below
    #                          it, by at most kl_beta_lr per update (0: fixed weight)
    symmetry: float = 0.0  # with init_tracker: weight of the mean squared distance between the action for the mirror image of an
    #                        imagined state (its observation history and command) and the mirror image of the action for the state
    #                        (tracker.mirror: left/right joints from tracker_meta's names and axes, M = diag(1, -1, 1)), on the inputs
    #                        held constant as the anchor's, over the steps that count for the return (0: not computed)
    start_share: float = 0.0  # share of the episodes (fresh, not on the rope) that start in the pose of start_joints (sim.Sim._fresh)
    start_joints: str = ""  # with start_share: "name:lo:hi,...": every joint whose name contains `name` starts at its default
    #                         position + U[lo, hi] rad (e.g. a stance a robot ends in after a particular start), the others as usual
    start_cmd: str = ""  # with start_share: "lo:hi,lo:hi,lo:hi": those episodes' first command (vx, vy, wz) uniform in these ranges
    #                      ("": drawn as any other)
    upper_body: str = ""  # with upper_tilt or upper_rate: the name of a body above the pelvis (e.g. the chest) in the robot's MJCF
    upper_tilt: float = 0.0  # weight of the penalty task.upper_tilt (the squared sine of upper_body's sideways tilt in the world, in the
    #                          pelvis heading's frame), in imagination's reward only and weighted as feet_cross is (0: not computed)
    upper_rate: float = 0.0  # weight of the penalty task.upper_rate (the squared rate of that tilt over a policy step, 1/s), in
    #                          imagination's reward only and weighted as feet_cross is (0: not computed)
    impact_accel: float = 0.0  # weight of the penalty task.impact_accel on the base's upward acceleration over a step (a proxy of
    #                            the ground force's peaks), in imagination's reward only and weighted as feet_cross is (0: not computed)
    support: float = 0.0  # share of the episodes that start with the robot on a rope that pulls its torso up (sim.Sim._support,
    #                       task.support_force): at rest in the default pose, the policy acting from the start, no command while on
    #                       the rope, which is paid out after a random hold and then taken off. The rope's top and pay-out speed are
    #                       the last two entries of the dynamics context; imagination pays the rope out as the simulator does (0: off)
    support_gap: str = "-0.02,0.05"  # with support: the range (m) of the lowest sole point's height over the terrain while hanging
    #                                  (negative: on the terrain, the rope carrying part of the weight or slack)
    support_point: str = ""  # with support: x,y,z (m) of the rope's attachment in the torso body's frame, the rope vertical above it
    #                          (an overhead crane): the robot hangs turned until the point is above its center of mass ("": the
    #                          body's center of mass, upright)
    # runtime
    imag_unroll: int = 4  # unrolling of the imagination loop (speed saturates at 4; compile time grows with it)
    imag_remat: str = "none"  # rematerialization of the imagination steps: none | dots | full (less memory, slower)
    mm_bf16: str = "imag"  # where matmuls output bf16, so their backward passes run on bf16 tensor cores: "" nowhere, "imag" in
    #                        imagination, "all" also in the world-model fits (collection and evaluation always output fp32)
    log_every: int = 50  # iterations per log line; the host waits for the GPU only here
    ckpt_minutes: float = 1.0  # minutes of training between checkpoints
    eval_n: int = 2048  # > 0: after training, evaluate every checkpoint with this many envs (evaluate.py) into <log_dir>/eval.jsonl
    eval_scale: float = 0.5  # evaluation terrain scale
    eval_sole_friction: int = 0  # 0: soles at friction 0.8 in evaluation, as in Holosoma's MuJoCo training
    seed: int = 0  # the terrain is drawn with seed 42 + seed, so seed 0 uses Holosoma's default terrain seed (42)
    log_dir: str = "runs/xsoma"
    wandb_project: str = ""  # Weights & Biases project of the log records' numbers and the options ("": off, unless WANDB_PROJECT is
    #                          set); wandb_log.py lists what is sent and what not
    wandb_entity: str = ""  # "": WANDB_ENTITY, else wandb's default
    wandb_name: str = ""  # run name; "": WANDB_NAME, else a generated one
    wandb_mode: str = ""  # online | offline | disabled; "": WANDB_MODE, else online
    jax_cache: str = "~/.cache/xsoma/jax"  # persistent compilation cache ("" for none): later runs load their programs
    xla_flags: str = "--xla_gpu_enable_triton_gemm=false"  # extra XLA flags, comma-separated, set before JAX's backend starts
    #                                                        (cuBLAS for every GEMM: faster on Hopper, neutral elsewhere)
    # several GPUs (multi_gpu.py)
    layout: str = "1GPU"  # 1GPU | AL: GPU 0 collects and fits the world model, the other GPUs update the policy
    learners: int = 0  # AL: the number of GPUs that update the policy; 0: every visible GPU but the first


# The world-model loss measures each state dimension's error in units of its typical one-step change, but never below these
# floors: position, rotation, linear and angular velocity (3 values each). The floors of q and qd are added in XSoma.__init__.
ERR_FLOOR = ((3, 3e-3), (3, 5e-3), (3, 2e-2), (3, 5e-2))


def widths(s: str):
    """Layer widths from a comma-separated string: "512,512,512" gives (512, 512, 512)."""
    return tuple(int(x) for x in s.split(","))


def set_jax(cache_dir: str = ""):
    """Global JAX settings: bf16 matrix multiplications, and (if cache_dir is set) a compilation cache on disk that keeps every
    program and XLA's autotuning results, so that later runs start faster."""
    jax.config.update("jax_default_matmul_precision", "bfloat16")
    if cache_dir:
        jax.config.update("jax_compilation_cache_dir", os.path.expanduser(cache_dir))
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")


def target_q(a):
    """Joint position targets of the PD law for raw actions a (Holosoma: default pose + 0.25 a)."""
    return task.DEFAULT_Q + task.ACTION_SCALE * a


class XSoma:
    """The learner. All training state lives in one pytree, the `carry` (see `init`): the simulator state, the replay buffer,
    the policy and the world model with their optimizer states, counters and running statistics. Every compiled program takes
    the carry and returns the updated one."""

    def __init__(self, cfg: Config):
        self.cfg = c = cfg
        set_jax(c.jax_cache)
        self.sim = Sim(SimCfg(num_envs=c.num_envs, terrain=c.terrain, terrain_scale=c.terrain_scale, foot=c.foot, preset=c.preset,
                              sole_friction=bool(c.sole_friction), random_start=bool(c.random_start), seed=42 + c.seed,
                              cmd_scale=self.cmd_scale, push_vel=c.push_vel, support=c.support,
                              support_gap=tuple(float(x) for x in c.support_gap.split(",")),
                              support_point=tuple(float(x) for x in c.support_point.split(",")) if c.support_point else None,
                              friction_surfaces=surfaces(c.friction_surfaces), start_share=c.start_share, start_joints=tuple(zip(*(x.tolist() for x in self.joint_ranges(c.start_joints)))) if c.start_share else None,
                              start_cmd=tuple(tuple(float(x) for x in r.split(":")) for r in c.start_cmd.split(",")) if c.start_cmd else None))  # fmt: skip
        if c.foot_clearance or c.touchdown_speed or c.turn_step or c.walk_step:  # (2, k): the sole collider points of each foot; frames 2 and 3 are the ankle-roll links (sim.py)
            self.sole_foot = np.asarray(self.sim.sole_frame) == np.arange(2, 4)[:, None]
        self.ext = None  # --ext_joints: (the joints' indices, their lowest and their highest goal in rad)
        if c.ext_joints:
            spec = [(i, float(lo), float(hi)) for name, lo, hi in (e.split(":") for e in c.ext_joints.split(",")) for i, d in enumerate(task.DOF) if name in d]
            assert spec and len({i for i, _, _ in spec}) == len(spec), f"--ext_joints {c.ext_joints}: no joint, or a joint twice"
            self.ext = tuple(np.asarray(x, t) for x, t in zip(zip(*spec), (int, np.float32, np.float32)))
            self.ext_q0, self.ext_scale = (np.broadcast_to(x, (task.J,))[self.ext[0]] for x in (task.DEFAULT_Q, task.ACTION_SCALE))
        self.rng = self.joint_ranges(c.range_joints) if c.range_penalty or c.range_target else None  # --range_joints
        self.upper = Kinematics(self.sim.m, [f"robot_{c.upper_body}"]) if c.upper_tilt or c.upper_rate else None  # --upper_body
        self.mirror = tracker.mirror(c.tracker_meta) if c.symmetry else None  # --symmetry: (perm, sign) of the tracker's joints
        self.reward_cfg = self.sim.reward_cfg if c.gait_clock else dataclasses.replace(self.sim.reward_cfg, feet_phase=0.0)  # imagination's reward
        self.ckpt_extra = None  # a tracker policy: what save_ckpt writes besides the networks, the normalizers and the metadata as loaded
        if c.init_tracker:
            z = np.load(c.init_tracker)
            # ponytail: the tracker's test vectors (test_oh, test_g, test_a) are left out, they would be stale; recompute test_a if a reader checks them
            self.ckpt_extra = {k: z[k] for k in z.files if k.startswith(("obs_norm.", "goal_norm.", "meta"))}
        self.J = task.J
        self.n_soles = len(self.sim.sole_radius)  # the number of sole collider points
        J = self.J
        # error floors of the world-model loss for every state dimension: ERR_FLOOR's four groups, then q and qd
        self.err_floor = np.concatenate([np.full((n,), v, np.float32) for n, v in ERR_FLOOR] + [np.full((J,), 3e-3, np.float32), np.full((J,), 5e-2, np.float32)])

    def upper_R(self, robot):
        """The world rotation (N, 3, 3) of --upper_body in a state of the task's view, by forward kinematics."""
        return self.upper.forward(robot.p, robot.R, robot.q)[1][..., 0, :, :]

    @staticmethod
    def joint_ranges(spec_s):
        """"name:lo:hi,..." (--range_joints, --start_joints) as (the joints' indices, their lower and their upper bound in rad)."""
        spec = [(i, float(lo), float(hi)) for name, lo, hi in (e.split(":") for e in spec_s.split(",")) for i, d in enumerate(task.DOF) if name in d]
        assert spec and len({i for i, _, _ in spec}) == len(spec), f"{spec_s}: no joint, or a joint twice"
        return tuple(np.asarray(x, t) for x, t in zip(zip(*spec), (int, np.float32, np.float32)))

    @property
    def cmd_scale(self):
        """--cmd_scale as three factors for task.sample_commands; None without the option."""
        c = self.cfg.cmd_scale
        return tuple(float(x) for x in c.split(",")) if c else None

    # ------------------------------------------------------------------ state
    def init(self, key):
        c, J = self.cfg, self.J
        ks = jax.random.split(key, 4)  # keys for the env, the policy, the world model and the training loop
        env = self.sim.init(ks[0])
        # the shapes of one step's record, found without running the step
        rec_t = jax.eval_shape(lambda s: self.record(s, *self.sim.step(s, jnp.zeros((c.num_envs, J))))[0], env)
        rec_t = rb.flatten({**rec_t, "ok": jax.ShapeDtypeStruct((c.num_envs,), jnp.float32)})
        self.xi_dim = rec_t["xi"].shape[-1]
        self.wm_in = (13 + 2 * J) + self.n_soles + J + self.xi_dim  # invariant state, sole clearances, PD error, dynamics context
        pi, extra = nets.pi_init(ks[1], task.ACTOR_DIM, J), {}
        if c.init_tracker:
            tr = tracker.load(c.init_tracker, c.tracker_meta)
            pi = {"enc": tr["enc"], "dec": tr["dec"]}
            self.frozen = {k: v for k, v in tr.items() if k not in pi}  # constants of the programs, never trained
            # mem: the tracker's memory in collection; pi0: the initial tracker, the anchor's reference
            extra = {"mem": tracker.memory(tr, c.num_envs), **({"pi0": jax.tree.map(jnp.copy, pi)} if c.anchor > 0 or c.kl_bound > 0 else {})}
            if c.kl_bound > 0:  # the KL term's weight, adapted with kl_beta_lr
                extra["kl_beta"] = jnp.float32(c.kl_beta)
            self.H = extra["mem"]["hist"].shape[1]
            assert min(c.warmup_steps, c.buffer_steps) >= self.H, f"a start needs {self.H} collected steps for the tracker's history"
            rec_t["heading"] = jax.ShapeDtypeStruct((c.num_envs,), jnp.float32)
        if self.ext is not None:  # on: the env's draw for its command; x, goal, speed: task.ext_step's state, x following the policy's target while it drives
            x = jnp.broadcast_to(jnp.asarray(self.ext_q0), (c.num_envs, len(self.ext_q0)))
            extra["ext"] = {"on": jnp.zeros((c.num_envs,), bool), "x": x + 0.0, "goal": x + 0.0, "speed": jnp.zeros_like(x)}  # separate buffers: the carry is donated
            rec_t["ext"] = jax.ShapeDtypeStruct((c.num_envs,), jnp.float32)
            if c.ext_blind:  # own: the policy's own last action for those joints, which it is fed instead of the applied one
                extra["ext"]["own"] = jnp.zeros_like(x)
                rec_t["ext_own"] = jax.ShapeDtypeStruct(x.shape, jnp.float32)
        if c.stance_carry:  # stance: each foot's stance clock in collection, at the state an action is taken in
            extra["stance"] = jnp.zeros((c.num_envs, 2))
            rec_t["stance"] = jax.ShapeDtypeStruct((c.num_envs, 2), jnp.float32)
        # wm: network weights, wm_b: its normalizers; 12 + 2 J outputs: the changes of position, rotation, v, w (3 each), q, qd
        wm, wm_b = nets.wm_init(ks[2], self.wm_in, 12 + 2 * J, widths(c.wm_hidden))
        # Muon masks (Muon where True, Adam elsewhere): none for the policy, the hidden weight matrices for the world model
        self.pi_mask = jax.tree.map(lambda _: False, pi)
        self.wm_mask = {"net": nets.muon_mask(wm["net"]) if c.wm_muon else jax.tree.map(lambda _: False, wm["net"])}
        # avg: the averaged policy that gets evaluated; wm_ema, gn_ema: the running averages behind the two gates; wm_ok: the
        # world-model gate's verdict for the iteration; nonfinite: the number of non-finite policy gradients
        carry = {"env": env, "buf": rb.init(c.buffer_steps, {k: jnp.zeros(v.shape, v.dtype) for k, v in rec_t.items()}),
                 "pi": pi, "avg": jax.tree.map(jnp.copy, pi), "wm": wm, "wm_b": wm_b,
                 "opt_pi": optim.init(pi, self.pi_mask), "opt_wm": optim.init(wm, self.wm_mask),
                 "key": ks[3], "it": jnp.zeros((), jnp.int32), "env_steps": jnp.zeros((), jnp.float32), "nonfinite": jnp.zeros((), jnp.int32),
                 "wm_ema": jnp.zeros((), jnp.float32), "gn_ema": jnp.zeros((), jnp.float32), "wm_ok": jnp.ones((), bool), **extra}  # fmt: skip
        # make weakly typed scalars strongly typed, so that the carry has the same types going into and coming out of each program
        strong = lambda x: jnp.asarray(x, dtype=x.dtype) if isinstance(x, jax.Array) and x.weak_type else x  # noqa: E731
        return jax.tree.map(strong, carry)

    # ------------------------------------------------------------------ collection
    def record(self, s, s1, tr):
        """One step's record for the replay buffer: the state the action was taken in with its task context (a start for
        imagination), the PD target and the state after the physics (one world-model transition), the dynamics context xi, and
        the done and termination flags and the reward."""
        rec = {"s_qpos": tr.qpos, "s_qvel": tr.qvel, "s1_qpos": tr.qpos1, "s1_qvel": tr.qvel1, "target": tr.target, "xi": self.sim.context(s),
               "a_last": s.a_last, "cmd": s.cmd, "phase": s.phase, "t": s.t, "freq": s.freq, "offset": s.offset, "delay": s.delay,
               "done": (tr.terminated | tr.truncated).astype(jnp.float32), "term": tr.terminated.astype(jnp.float32), "r": tr.reward}  # fmt: skip
        return rec, tr.stats

    def ingest(self, carry, flat):
        """Add one step of all envs' transitions to the replay buffer and to the world model's normalizers."""
        rec, heights = rb.nest(flat), carry["env"].world.heights
        S0, S1 = unflatten(rec["s_qpos"], rec["s_qvel"]), unflatten(rec["s1_qpos"], rec["s1_qvel"])
        v0 = self.sim.view_state(S0, heights)
        wb, ok = carry["wm_b"], rec["ok"] > 0
        upd = lambda st, x: nets.rn_update(st, x, weights=ok)  # noqa: E731
        wm_b = {"in_norm": upd(wb["in_norm"], nets.wm_inputs(S0, v0.height, v0.soles, target_q(rec["target"]), rec["xi"])),
                "out_norm": upd(wb["out_norm"], nets.wm_targets(S0, S1, task.DT)), "inc_norm": upd(wb["inc_norm"], nets.increments(S0, S1))}  # fmt: skip
        return {**carry, "buf": rb.add(carry["buf"], flat), "wm_b": wm_b, "env_steps": carry["env_steps"] + self.cfg.num_envs}

    def collect_step(self, carry, key):
        c = self.cfg
        s = carry["env"]
        noisy = (jnp.arange(c.num_envs) * c.det_share) % 1.0 >= c.det_share  # every 1 / det_share-th env acts without noise
        seen = s
        if self.ext is not None:
            (i, lo, hi), e = self.ext, carry["ext"]
            k_on, k_x = jax.random.split(jax.random.fold_in(key, 2))
            # a state whose episode counter is a multiple of RESAMPLE has a new command (a reset or a resample)
            on = jnp.where(s.t % task.RESAMPLE == 0, jax.random.uniform(k_on, (c.num_envs,)) < c.ext_share, e["on"])
            ext = (on & (jnp.abs(s.cmd) < 0.01).all(-1))[:, None]
            if c.ext_blind:
                own = jnp.where((s.t == 0)[:, None], 0.0, e["own"])  # a reset zeroes the last action
                q, qd, a_last = self.hide(ext, s.data.qpos[:, 7:], s.data.qvel[:, 6:], s.a_last, own)
                seen = SimpleNamespace(data=SimpleNamespace(qpos=jnp.concatenate((s.data.qpos[:, :7], q), -1), qvel=jnp.concatenate((s.data.qvel[:, :6], qd), -1)),
                                       a_last=a_last, t=s.t, cmd=s.cmd)  # fmt: skip
        if c.init_tracker:
            a, mem = tracker.step({**self.frozen, **carry["pi"]}, carry["mem"], seen, c.tracker_heading == "hold",
                                  jax.random.fold_in(key, 1) if c.tracker_noise else None, c.tracker_noise)  # fmt: skip
            carry = {**carry, "mem": mem}
        else:
            a = nets.pi_apply(carry["pi"], s.obs)
        a = a + jnp.where(noisy, c.action_noise, 0.0)[:, None] * jax.random.normal(key, (c.num_envs, self.J))
        if self.ext is not None:
            x, goal, speed = task.ext_step(k_x, e["x"], e["goal"], e["speed"], lo, hi)
            x = jnp.where(ext, x, target_q(a)[:, i])
            carry = {**carry, "ext": {"on": on, "x": x, "goal": goal, "speed": speed, **({"own": a[:, i]} if c.ext_blind else {})}}
            a = a.at[:, i].set((x - self.ext_q0) / self.ext_scale)  # the policy's own action where it drives
        s1, tr = self.sim.step(s, a)
        rec, stats = self.record(s, s1, tr)
        if self.ext is not None:
            rec["ext"] = ext[:, 0].astype(jnp.float32)
            if c.ext_blind:
                rec["ext_own"] = own
        if c.init_tracker:  # the commanded heading of this step's observation (the memory holds the next step's)
            rec["heading"] = mem["heading"] - task.DT * s.cmd[:, 2]
        if c.stance_carry:  # the clock at the state the action was taken in; then advanced on the state after the physics
            rec["stance"] = carry["stance"]
            v1 = self.sim.view_state(unflatten(tr.qpos1, tr.qvel1), s.world.heights)
            keep = ~(tr.terminated | tr.truncated | (s1.cmd != s.cmd).any(-1))
            carry = {**carry, "stance": task.stance_clock(carry["stance"], self.sole_low(v1), keep)}
        flat = rb.sanitize(rb.flatten(rec))
        flat["ok"] = flat["ok"] * (1 - tr.diverged.astype(jnp.float32))  # a physics failure is not a valid transition
        stats = {**stats, "nonfinite_env": 1 - flat["ok"].mean()}
        return self.ingest({**carry, "env": s1}, flat), stats

    def hide(self, on, q, qd, a_last, own):
        """(q, qd, a_last) as a policy gets them that is not shown the joints driven from outside: where `on` (..., 1), those
        joints are at their default pose and at rest, and their last action is the policy's `own`."""
        i, at = self.ext[0], lambda x, v: x.at[..., self.ext[0]].set(jnp.where(on, v, x[..., self.ext[0]]))  # noqa: E731
        return at(q, self.ext_q0), at(qd, 0.0), at(a_last, own)

    # ------------------------------------------------------------------ world model
    def wm_rollout(self, wm, wm_b, S, targets, xis, heights, unroll=True):
        """Open-loop rollout of the world model from the state S, fed with a window's recorded PD targets and its own predictions.
        Returns the k predicted states, (B, k, ...) per state part."""

        def step(S, x):
            tgt, xi = x
            v = self.sim.view_state(S, heights)
            S1 = nets.wm_step(wm, wm_b, S, v.height, v.soles, target_q(tgt), xi, task.DT, out_bf16=self.cfg.mm_bf16 == "all")
            return S1, S1

        _, Ss = jax.lax.scan(step, S, (jnp.swapaxes(targets, 0, 1), jnp.swapaxes(xis, 0, 1)), unroll=unroll)
        return tuple(jnp.swapaxes(P, 0, 1) for P in Ss)

    def wm_loss(self, wm, wm_b, w, heights):
        """World-model loss on windows w: the squared open-loop error of every state dimension (in units of its typical one-step
        change, at least its ERR_FLOOR), averaged over the dimensions and the windows, and summed over each window's valid steps.
        Also returns the error at the last step of each window."""
        S_hat = self.wm_rollout(wm, wm_b, unflatten(w["s_qpos"][:, 0], w["s_qvel"][:, 0]), w["target"], w["xi"], heights)
        S_true = unflatten(w["s1_qpos"], w["s1_qvel"])
        std = jnp.maximum(nets.rn_std(wm_b["inc_norm"], 1e-4), self.err_floor)
        e = ((nets.state_error(S_hat, S_true) / std) ** 2).mean(-1)
        return (e * w["mask"]).sum() / w["mask"].shape[0], e[:, -1]

    def wm_update(self, carry, key):
        """One gradient step of the world model on replayed windows. The statistics are the loss per step and the error at the
        window's last step."""
        c = self.cfg
        w = rb.sample_windows(carry["buf"], key, c.wm_batch, c.wm_window)
        (loss, e_last), grads = jax.value_and_grad(self.wm_loss, has_aux=True)(carry["wm"], carry["wm_b"], w, carry["env"].world.heights)
        grads, _ = optim.clip_by_global_norm(grads, 1.0)
        wm, opt = optim.step(carry["wm"], grads, carry["opt_wm"], self.wm_mask, c.wm_lr, betas=(0.9, 0.999))
        m = w["mask"][:, -1]
        return {**carry, "wm": wm, "opt_wm": opt}, {"wm": loss / c.wm_window, "wm8": (e_last * m).sum() / jnp.maximum(m.sum(), 1)}

    def wm_eval(self, carry, key, B: int = 512, K: int = 32):
        """Check of the world model on real data, for the log only: open-loop rollouts of K steps from real states with the
        recorded PD targets (valid steps of each window only). Returns the pelvis height error after 8 and after K steps, and how
        often the simulator terminated against how often the model's states imply a termination."""
        heights = carry["env"].world.heights
        w = rb.sample_windows(carry["buf"], key, B, K)
        S_hat = self.wm_rollout(carry["wm"], carry["wm_b"], unflatten(w["s_qpos"][:, 0], w["s_qvel"][:, 0]), w["target"], w["xi"], heights,
                                unroll=2)  # fmt: skip
        v_hat = self.sim.view_state(S_hat, heights)
        v_true = self.sim.view(w["s1_qpos"], w["s1_qvel"], heights)
        herr = jnp.abs(v_hat.height - v_true.height)
        m = w["mask"]
        mean = lambda x, j: (x[:, j] * m[:, j]).sum() / jnp.maximum(m[:, j].sum(), 1)  # noqa: E731
        tp = (v_hat.clearance.min(-1) < 0).astype(jnp.float32)
        return {"h_err8": mean(herr, 7), "h_errK": mean(herr, K - 1), "term_env": (w["term"] * m).sum() / jnp.maximum(m.sum(), 1),
                "term_wm": (tp * m).sum() / jnp.maximum(m.sum(), 1)}  # fmt: skip

    # ------------------------------------------------------------------ imagination
    def sample_starts(self, buf, key):
        """Starting states for imagination: replayed states with their task context. A share (`cmd_fresh`) gets a freshly drawn
        command and gait frequency, with the gait phase recomputed, because in the real task a new command can arrive at any
        state (every 500 steps and after a reset). For a tracker policy a start also has `hist`, the observation history that
        the tracker held there, rebuilt from the transitions before it (with new draws of the observation noise)."""
        c, B = self.cfg, self.cfg.pi_batch
        k_s, k_u, k_c = jax.random.split(key, 3)
        if c.init_tracker:
            w = rb.sample_histories(buf, k_s, B, self.H, c.fall_starts, c.fall_lead or c.horizon)
            start = rb.nest({k: v[:, -1] for k, v in w.items()})
            if self.ext is not None and c.ext_blind:  # the history as the policy saw it; `start` keeps the true state
                q, qd, a_last = self.hide((w["ext"] > 0)[..., None], w["s_qpos"][..., 7:], w["s_qvel"][..., 6:], w["a_last"], w["ext_own"])
                w = {**w, "s_qpos": jnp.concatenate((w["s_qpos"][..., :7], q), -1), "s_qvel": jnp.concatenate((w["s_qvel"][..., :6], qd), -1), "a_last": a_last}
            start["hist"] = tracker.history(self.frozen, w["s_qpos"], w["s_qvel"], w["a_last"], w["heading"] if c.tracker_heading == "hold" else None,
                                            w["done"], jax.random.fold_in(k_s, 1) if c.tracker_noise else None, c.tracker_noise)  # fmt: skip
        else:
            start = rb.nest(rb.sample_starts(buf, k_s, B, c.fall_starts, c.fall_lead or c.horizon))
        cmd, freq = task.sample_commands(k_c, B, self.cmd_scale)
        fresh = jax.random.uniform(k_u, (B,)) < c.cmd_fresh
        if c.support:  # no command on the rope, as in the simulator
            on = start["xi"][:, -2] > task.SUPPORT_OFF
            cmd, fresh = jnp.where(on[:, None], 0.0, cmd), fresh | on
        start["cmd"] = jnp.where(fresh[:, None], cmd, start["cmd"])
        start["freq"] = jnp.where(fresh, freq, start["freq"])
        start["phase"] = jnp.where(fresh[:, None], task.gait_phase(start["t"], start["freq"], start["offset"], start["cmd"]), start["phase"])
        if c.stance_carry:  # a new command starts the clocks again, as in collection
            start["stance"] = jnp.where(fresh[:, None], 0.0, start["stance"])
        return start

    def termination(self, clearance):
        """Termination flags of imagined states. The value is the exact 0/1 event: a termination collider below the terrain
        surface, which agrees with the simulator's contact-force termination on about 99 % of steps. The gradient is that of a
        sigmoid of the smallest clearance, so that the policy gets a signal before the robot has fallen."""
        cmin = clearance.min(-1)
        soft = jax.nn.sigmoid(-cmin / self.cfg.term_tau)
        hard = (cmin < 0).astype(jnp.float32)
        return hard + soft - jax.lax.stop_gradient(soft)

    def foot_clearance(self, v, v1):
        """task.foot_clearance of the step from the view `v` to `v1`: each foot's lowest sole point over the terrain after the
        step, and how far its contact point moved."""
        return task.foot_clearance(self.sole_low(v1), (v1.robot.foot_p - v.robot.foot_p)[..., :2], self.cfg.foot_clearance_height)

    def sole_low(self, v):
        """(N, 2) each foot's lowest sole point over the terrain in the view `v`."""
        return jnp.where(self.sole_foot, v.soles[..., None, :], jnp.inf).min(-1)

    def imagine(self, pi, wm, wm_b, start, keys, scale, heights, pi0=None, kl_beta=None):
        """The loss of the policy update: the negative mean discounted return of `horizon` imagined steps from `start`. A step
        follows the task's order: the policy acts on the observation, the world model moves the robot, the reward and the
        termination are read on the new state, then the gait phase advances and the command resamples every 500 steps. A tracker
        policy (`pi`: its encoder and decoder) acts as in tracker.step, on its noisy observation of the imagined state, the
        history and the commanded heading; with `pi0`, the initial tracker, the loss gains `anchor` times the mean squared distance
        between the two policies' actions on the same inputs (held constant for it), over the steps that count for the return;
        with --symmetry, `symmetry` times the same mean of |pi(mirrored inputs) - mirrored pi(inputs)|^2 (inputs held constant);
        and with `kl_beta` (--kl_bound) kl_beta x max(0, KL - kl_bound), KL being the anchor's distance's mean over the same steps
        (weighted by their survival, held constant) over 2 action_noise^2: the KL divergence of two Gaussians of that std around
        the actions."""
        c, B = self.cfg, self.cfg.pi_batch
        bf16 = c.mm_bf16 in ("imag", "all")
        rcfg = self.reward_cfg
        S0 = unflatten(start["s_qpos"], start["s_qvel"])
        delay, offset = start["delay"], start["offset"]
        if c.init_tracker:
            pi = {**self.frozen, **pi}

        ext = None
        if self.ext is not None:  # as in collection, from the last target; a start whose command was redrawn hands the joints back
            (i, lo, hi), k_g, k_s = self.ext, *jax.random.split(jax.random.fold_in(keys[0], 2))
            x = target_q(start["a_last"])[:, i]
            ext_on = ((start["ext"] > 0) & (jnp.abs(start["cmd"]) < 0.01).all(-1))[:, None]
            ext_j = ext_on & (jnp.arange(self.J) == i[:, None]).any(0)  # (B, J): the joints that the policy does not drive
            ext = (x, jax.random.uniform(k_g, x.shape, minval=lo, maxval=hi), task.EXT_SPEED * jax.random.uniform(k_s, x.shape), start.get("ext_own", 0 * x))

        def step(carry, k):
            S, v, a_last, cmd, freq, phase, t, mem, ext, xi, st = carry
            k_o, k_c = jax.random.split(k)
            pen = sym = None
            if c.init_tracker:
                prev, heading = mem  # the H - 1 observations before this step's, the commanded heading
                seen = self.hide(ext_on, S[4], S[5], a_last, ext[3]) if ext is not None and c.ext_blind else (S[4], S[5], a_last)
                o = tracker.observation(pi, seen[0], seen[1], S[1], v.robot.w, seen[2][:, pi["to_tracker"]], heading if c.tracker_heading == "hold" else None,
                                        k_o if c.tracker_noise else None, c.tracker_noise)  # fmt: skip
                # ponytail: the goal is encoded at every step though the command changes every 500; encode per rollout and at resamples if imagination is too slow
                hist, g = jnp.concatenate((prev, o[:, None]), 1), tracker.goal(pi, cmd)
                a = tracker.act(pi, hist, g, out_bf16=bf16)[:, pi["to_sim"]]
                mem = (hist[:, 1:], heading + task.DT * cmd[:, 2])
                # the anchor's and the symmetry's actions on the inputs as constants: through the rollout the penalty's gradient
                # explodes (the distance then grows with its weight); as a regression at fixed inputs it pulls the weights back
                fixed = jax.lax.stop_gradient(hist)
                if pi0 is not None:
                    a0 = tracker.act({**pi, **pi0}, fixed, g, out_bf16=bf16)
                if pi0 is not None or self.mirror is not None:
                    a_fix = tracker.act(pi, fixed, g, out_bf16=bf16)
                if pi0 is not None:
                    pen = jnp.sum((a_fix - a0) ** 2, -1)
                if self.mirror is not None:  # tracker order: the mirror map's
                    a_m = tracker.act(pi, tracker.mirror_obs(self.mirror, fixed), tracker.goal(pi, cmd * tracker.MIRROR_CMD), out_bf16=bf16)
                    sym = jnp.sum((a_m - a_fix[..., self.mirror[0]] * self.mirror[1]) ** 2, -1)
            else:
                obs, _ = task.observations(v.robot, a_last, cmd, phase, k_o)
                a = nets.pi_apply(pi, obs, out_bf16=bf16)
            if ext is not None:
                ext = (*task.ext_step(jax.random.fold_in(k, 2), *ext[:3], lo, hi), a[:, i])
                a = a.at[:, i].set(jnp.where(ext_on, (ext[0] - self.ext_q0) / self.ext_scale, a[:, i]))
            # the PD target: an env with an action delay applies the previous action instead of this one
            tgt = jnp.clip(jnp.where(delay[:, None] == 0, a, a_last), -task.ACTION_CLIP, task.ACTION_CLIP)
            S1 = nets.wm_step(wm, wm_b, S, v.height, v.soles, target_q(tgt), xi, task.DT, out_bf16=bf16)
            v1 = self.sim.view_state(S1, heights)
            if ext is None:
                r = task.reward(task.reward_terms(v1.robot, v1.foot_h, a, a_last, cmd, phase), rcfg, scale)
            else:  # the pose and the action rate of joints that the policy does not drive are not its cost
                rb = v1.robot._replace(q=jnp.where(ext_j, task.DEFAULT_Q, v1.robot.q))
                r = task.reward(task.reward_terms(rb, v1.foot_h, jnp.where(ext_j, 0.0, a), jnp.where(ext_j, 0.0, a_last), cmd, phase), rcfg, scale)
            if c.track_l1:  # weighted as task.weighted weighs a penalty
                r = r - task.DT * c.track_l1 * scale * task.track_l1(v1.robot, cmd, c.track_rel)
            if c.stand_still:  # weighted as task.weighted weighs a penalty
                r = r - task.DT * c.stand_still * scale * task.stand_still(v1.robot, v1.robot.foot_p - v.robot.foot_p, cmd)
            if c.feet_cross:  # weighted as task.weighted weighs a penalty
                r = r - task.DT * c.feet_cross * scale * task.feet_cross(v1.robot)
            if c.foot_clearance:  # weighted as task.weighted weighs a penalty
                r = r - task.DT * c.foot_clearance * scale * self.foot_clearance(v, v1)
            if c.torque_limit:  # weighted as task.weighted weighs a penalty; the torque of the step's first substep, from its start
                x = task.torque_limit(v.robot.q, v.robot.qd, target_q(tgt), c.torque_soft)
                r = r - task.DT * c.torque_limit * scale * (x if ext is None else jnp.where(ext_j, 0.0, x)).sum(-1)
            if c.range_penalty:  # weighted as task.weighted weighs a penalty; the pose the step ends in
                r = r - task.DT * c.range_penalty * scale * task.joint_range(v1.robot.q[:, self.rng[0]], *self.rng[1:]).sum(-1)
            if c.range_target:  # weighted as task.weighted weighs a penalty; the PD target of the step's action
                r = r - task.DT * c.range_target * scale * task.joint_range(target_q(tgt)[:, self.rng[0]], *self.rng[1:]).sum(-1)
            if c.touchdown_speed:  # weighted as task.weighted weighs a penalty; feet near the terrain at the step's end
                dz = (v1.robot.foot_p - v.robot.foot_p)[..., 2]
                r = r - task.DT * c.touchdown_speed * scale * task.touchdown_speed(self.sole_low(v1), dz, c.touchdown_height)
            if c.turn_step or c.walk_step:  # weighted as task.weighted weighs a penalty; the feet's heights in the imagined state
                st = task.turn_stance(st, self.sole_low(v1))
                r = r - task.DT * c.turn_step * scale * task.turn_step(st, cmd[:, 2], c.turn_step_stance)
                r = r - task.DT * c.walk_step * scale * task.walk_step(st, cmd, c.turn_step_stance)
            if c.balance:  # weighted as task.weighted weighs a penalty; the state the step ends in
                r = r - task.DT * c.balance * scale * task.balance(v1.robot, c.balance_margin)
            if c.step_length:  # weighted as task.weighted weighs a penalty; the state the step ends in
                r = r - task.DT * c.step_length * scale * task.step_length(v1.robot, cmd, c.step_base, c.step_per_speed, c.step_width)
            if c.joint_vel:  # weighted as task.weighted weighs a penalty; the joint speeds of the state the step ends in
                r = r - task.DT * c.joint_vel * scale * task.joint_vel(v1.robot.qd if ext is None else jnp.where(ext_j, 0.0, v1.robot.qd))
            if c.upper_tilt:  # weighted as task.weighted weighs a penalty; the state the step ends in
                r = r - task.DT * c.upper_tilt * scale * task.upper_tilt(self.upper_R(v1.robot), v1.robot.R)
            if c.upper_rate:  # weighted as task.weighted weighs a penalty; the step's start and end states
                r = r - task.DT * c.upper_rate * scale * task.upper_rate(self.upper_R(v.robot), v.robot.R, self.upper_R(v1.robot), v1.robot.R)
            if c.impact_accel:  # weighted as task.weighted weighs a penalty; the world model's base velocities
                r = r - task.DT * c.impact_accel * scale * task.impact_accel(v.robot.v, v1.robot.v)
            d = self.termination(v1.clearance)
            if c.tilt_term:  # S1[1]: the base's rotation matrix, whose last entry is the cosine of its tilt (width: term_tau, in cosine)
                d = 1 - (1 - d) * (1 - self.termination((S1[1][..., 2, 2] - np.cos(c.tilt_term))[..., None]))
            if c.self_clearance or c.self_clearance_term:
                gap = self.sim.self_distance(S1)
                if c.self_clearance:  # weighted as task.weighted weighs a penalty
                    r = r - task.DT * c.self_clearance * scale * task.self_clearance(gap)
                if c.self_clearance_term:  # either event ends the rollout, each with its own sigmoid's gradient
                    d = 1 - (1 - d) * (1 - self.termination(gap + c.self_clearance_term))
            t1 = t + 1
            phase1 = task.gait_phase(t1, freq, offset, cmd)
            cmd2, freq2 = task.sample_commands(k_c, B, self.cmd_scale)
            new = t1 % task.RESAMPLE == 0
            cmd1, freq1 = jnp.where(new[:, None], cmd2, cmd), jnp.where(new, freq2, freq)
            if c.support:  # the rope is paid out as in the simulator, and a robot on it gets no new command
                top, rate = task.support_pay_out(xi[:, -2], xi[:, -1])
                xi = jnp.concatenate((xi[:, :-2], top[:, None], rate[:, None]), -1)
                cmd1 = jnp.where((top > task.SUPPORT_OFF)[:, None], 0.0, cmd1)
            return (S1, v1, a, cmd1, freq1, phase1, t1, mem, ext, xi, st), (r, d, pen, sym)

        if c.imag_remat == "dots":
            step = jax.checkpoint(step, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
        elif c.imag_remat == "full":
            step = jax.checkpoint(step)
        mem = (start["hist"][:, :-1], start["heading"]) if c.init_tracker else None
        init = (S0, self.sim.view_state(S0, heights), start["a_last"], start["cmd"], start["freq"], start["phase"], start["t"], mem, ext, start["xi"], start["stance"] if c.stance_carry else jnp.zeros((B, 2)))
        _, (r, d, pen, sym) = jax.lax.scan(step, init, keys, unroll=c.imag_unroll)
        alive = jnp.cumprod(1 - d, 0)
        alive_before = jnp.concatenate((jnp.ones_like(alive[:1]), alive[:-1]), 0)  # a step's reward counts if alive before it
        disc = c.gamma ** jnp.arange(c.horizon, dtype=jnp.float32)[:, None]
        G = (disc * alive_before * (r - c.term_penalty * d)).sum(0)
        # statistics: mean return, mean reward per step, and the share of rollouts still alive at the end of the horizon
        loss, stats = -G.mean(), {"G": G.mean(), "r": (alive_before * r).sum(0).mean() / c.horizon, "surv": alive[-1].mean()}
        if pen is not None:
            anchor = (alive_before * pen).mean()
            loss, stats = loss + c.anchor * anchor, {**stats, "anchor": anchor}
        if sym is not None:
            sym = (alive_before * sym).mean()
            loss, stats = loss + c.symmetry * sym, {**stats, "sym": sym}
        if kl_beta is not None:
            w = jax.lax.stop_gradient(alive_before)
            kl = (w * pen).sum() / jnp.maximum(w.sum(), 1.0) / (2 * c.action_noise**2)
            kl_term = kl_beta * jnp.maximum(0.0, kl - c.kl_bound)
            loss, stats = loss + kl_term, {**stats, "kl": kl, "kl_term": kl_term}
        return loss, stats

    def policy_update(self, carry, start, key):
        """One gradient step of the policy through imagination. The step is skipped when the gradient norm is not finite or
        exceeds `pi_gn_gate` times its running average."""
        c = self.cfg
        env = carry["env"]
        (_, aux), grads = jax.value_and_grad(self.imagine, has_aux=True)(carry["pi"], carry["wm"], carry["wm_b"], start, jax.random.split(key, c.horizon),
                                                                         env.scale, env.world.heights, **({"pi0": carry["pi0"]} if "pi0" in carry else {}),
                                                                         **({"kl_beta": carry["kl_beta"]} if "kl_beta" in carry else {}))  # fmt: skip
        grads, gn = optim.clip_by_global_norm(grads, c.pi_clip)  # gn is the norm before clipping
        finite = jnp.isfinite(gn)
        first = carry["gn_ema"] <= 0
        ok = finite & (first | (c.pi_gn_gate <= 0) | (gn < c.pi_gn_gate * carry["gn_ema"]))
        # the average follows accepted steps and grows by 1 % after a skipped one, so that a lasting rise is accepted in the end
        gn_ema = jnp.where(ok, jnp.where(first, gn, 0.98 * carry["gn_ema"] + 0.02 * gn), 1.01 * carry["gn_ema"])
        grads = jax.tree.map(lambda g: jnp.where(ok, g, 0.0), grads)
        pi, opt_pi = optim.step(carry["pi"], grads, carry["opt_pi"], self.pi_mask, c.pi_lr, betas=(0.9, 0.95), ok=ok)
        # the averaged policy is a copy of the policy at the first policy update, then follows it as an exponential moving average
        avg = optim.ema(carry["avg"], pi, c.pi_ema, carry["it"] == c.wm_only_iters)
        stats = {**aux, "gn": gn, "gn_gated": (finite & ~ok).astype(jnp.float32)}
        out = {"pi": pi, "opt_pi": opt_pi, "avg": avg, "gn_ema": gn_ema, "nonfinite": carry["nonfinite"] + (~finite).astype(jnp.int32)}
        if "kl_beta" in carry:  # dual ascent on the KL term's weight, from the KL before this step (stats: the weight it was taken with).
            # Only after an accepted step: a weight that kept growing while the gate skips the steps it causes would lock the gate
            step = c.kl_beta_lr * jnp.clip(aux["kl"] / c.kl_bound - 1.0, -1.0, 1.0)
            out["kl_beta"] = jnp.where(ok, jnp.maximum(0.0, carry["kl_beta"] + step), carry["kl_beta"])
            stats["kl_beta"] = carry["kl_beta"]
        return out, stats

    # ------------------------------------------------------------------ iteration programs
    def collect_iter(self, carry):
        """env_steps steps of all envs; per-step statistics, summed (counts) or averaged (rates) over the steps."""
        key, k_c = jax.random.split(carry["key"])
        carry, st = jax.lax.scan(self.collect_step, {**carry, "key": key}, jax.random.split(k_c, self.cfg.env_steps))
        summed = ("done", "terminated", "len_sum", "rew_sums", "pushes", "diverged")
        return carry, {k: (v.sum(0) if k in summed else v.mean(0)) for k, v in st.items()}

    def wm_iter(self, carry):
        """wm_batches world-model fits, and the gate that keeps the policy from training on a model whose loss jumped."""
        c = self.cfg
        key, k_w = jax.random.split(carry["key"])
        carry, wstats = jax.lax.scan(self.wm_update, {**carry, "key": key}, jax.random.split(k_w, c.wm_batches))
        wm_now = wstats["wm"].mean()
        first = carry["wm_ema"] <= 0
        healthy = first | (c.wm_gate <= 0) | (wm_now < c.wm_gate * carry["wm_ema"])
        # as for the gradient gate: the average grows by 0.5 % after an unhealthy iteration, so a lasting rise gets through
        wm_ema = jnp.where(healthy, jnp.where(first, wm_now, 0.98 * carry["wm_ema"] + 0.02 * wm_now), 1.005 * carry["wm_ema"])
        return {**carry, "wm_ema": wm_ema, "wm_ok": healthy}, {"wm": wm_now, "wm8": wstats["wm8"][-1]}

    def pi_iter(self, carry):
        """pi_updates policy updates (after wm_only_iters iterations, while the world model is healthy)."""
        c = self.cfg
        key, k_p = jax.random.split(carry["key"])
        names = ("pi", "opt_pi", "avg", "gn_ema", "nonfinite") + (("kl_beta",) if "kl_beta" in carry else ())

        def one(sub, k):
            k_s, k_u = jax.random.split(k)
            return self.policy_update({**carry, **sub}, self.sample_starts(carry["buf"], k_s), k_u)

        def do(sub):
            sub, st = jax.lax.scan(one, sub, jax.random.split(k_p, c.pi_updates))
            return sub, jax.tree.map(jnp.mean, st)

        def skip(sub):
            kl = ("kl", "kl_term") if "kl_beta" in carry else ()
            st = {k: jnp.zeros((), jnp.float32) for k in ("G", "r", "surv", "gn", "gn_gated") + (("anchor",) if "pi0" in carry else ()) + (("sym",) if c.symmetry else ()) + kl}
            return sub, {**st, **({"kl_beta": sub["kl_beta"]} if kl else {})}

        active = carry["it"] >= c.wm_only_iters
        sub, pstats = jax.lax.cond(active & carry["wm_ok"], do, skip, {k: carry[k] for k in names})
        return {**carry, **sub, "key": key, "it": carry["it"] + 1}, {**pstats, "wm_gated": (active & ~carry["wm_ok"]).astype(jnp.float32)}

    def compile_programs(self, carry):
        """Compile the programs of an iteration ahead of time: collection, world-model fits and policy updates (each updates the
        carry in place, by donation), plus the world-model check, which leaves the carry intact. Lowering (tracing) holds Python's
        GIL, so the programs are lowered one after the other; XLA's compile step releases it, so they compile in parallel threads
        and the start-up waits for the largest program, not for their sum."""
        lowered = {"collect": jax.jit(self.collect_iter, donate_argnums=(0,)).lower(carry),
                   "wm": jax.jit(self.wm_iter, donate_argnums=(0,)).lower(carry),
                   "pi": jax.jit(self.pi_iter, donate_argnums=(0,)).lower(carry)}  # fmt: skip
        if self.cfg.wm_eval:
            lowered["wm_eval"] = jax.jit(lambda cr: self.wm_eval(cr, jax.random.fold_in(cr["key"], 3))).lower(carry)
        with ThreadPoolExecutor(len(lowered)) as ex:
            return dict(zip(lowered, ex.map(lambda low: low.compile(), lowered.values())))


def save_ckpt(path, pi_avg, pi_live, meta, extra=None):
    """Write the averaged and the live policy to <path>.npz, through a temporary file so that a reader never sees a partial one,
    and the metadata (iteration, times, env steps) to <path>.json. A tracker policy, given with `extra` (the normalizers and
    the metadata of its initial checkpoint), is written as a tracker checkpoint (tracker.load): the averaged policy under the
    tracker's own names, the live one under the prefix `live.`."""
    flat = dict(extra or {})
    for name, layers in (("avg", pi_avg), ("live", pi_live)):
        if extra:
            flat |= {f"{'' if name == 'avg' else 'live.'}{net}.{i}.{k}": np.asarray(v) for net in layers for i, layer in enumerate(layers[net]) for k, v in layer.items()}
            continue
        for i, layer in enumerate(layers):
            for k, v in layer.items():
                flat[f"{name}/{i}/{k}"] = np.asarray(v)
    np.savez(path + ".tmp.npz", **flat)
    os.replace(path + ".tmp.npz", path + ".npz")
    with open(path + ".json", "w") as f:
        json.dump(meta, f)


def parse(argv=None) -> Config:
    """The Config given by the command-line flags, `--field value` for every field."""
    p = argparse.ArgumentParser(description="Train XSoma on Holosoma's G1 velocity-tracking task.")
    for f in dataclasses.fields(Config):
        p.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    return Config(**vars(p.parse_args(argv)))


def main(argv=None):
    """Train for `minutes`, log every `log_every` iterations and save checkpoints, then evaluate every checkpoint."""
    t_launch = time.time()  # the wall clock of a comparison counts from here (start-up and compilation included)
    cfg = parse(argv)
    # the robot is read when xsoma.task is imported: train.py passes the flag on in time, other entry points set the variable
    assert cfg.robot_dir == (task.ROBOT_DIR or ""), "--robot_dir needs train.py, or XSOMA_ROBOT_DIR set to the same directory"
    assert bool(cfg.init_tracker) == bool(cfg.tracker_meta) and (cfg.init_tracker or not cfg.anchor), "--init_tracker and --tracker_meta go together, --anchor with them"
    assert cfg.tracker_heading in ("hold", "rate") and cfg.feet_cross >= 0, "--tracker_heading is hold or rate, --feet_cross is not negative"
    assert min(cfg.self_clearance, cfg.self_clearance_term) >= 0, "--self_clearance and --self_clearance_term are not negative"
    assert min(cfg.foot_clearance, cfg.foot_clearance_height, cfg.push_vel) >= 0, "--foot_clearance, --foot_clearance_height and --push_vel are not negative"
    assert not cfg.cmd_scale or len(cfg.cmd_scale.split(",")) in (3, 4, 5, 7), "--cmd_scale is three factors, for vx, vy and wz: 1,0.5,1 (a fourth: the share of zero commands, a fifth: of zero components, a sixth and seventh: the share of commands scaled and the factor)"
    assert min(cfg.range_penalty, cfg.range_target) >= 0 and bool(cfg.range_penalty or cfg.range_target) == bool(cfg.range_joints), \
        "--range_penalty and --range_target are not negative, either goes with --range_joints"
    assert min(cfg.track_l1, cfg.stand_still) >= 0 and (not cfg.ext_blind or (cfg.ext_joints and cfg.init_tracker)), "--track_l1 is not negative, --ext_blind needs --ext_joints and --init_tracker"
    assert cfg.track_rel >= 0 and (cfg.track_l1 or not cfg.track_rel), "--track_rel is not negative and needs --track_l1"
    assert cfg.torque_limit >= 0 and cfg.torque_soft > 0, "--torque_limit is not negative, --torque_soft is positive"
    assert min(cfg.touchdown_speed, cfg.touchdown_height, cfg.impact_accel) >= 0, "--touchdown_speed, --touchdown_height and --impact_accel are not negative"
    assert cfg.turn_step >= 0 and cfg.turn_step_stance > 0, "--turn_step is not negative, --turn_step_stance is positive"
    assert not cfg.stance_carry or cfg.turn_step or cfg.walk_step, "--stance_carry needs --turn_step or --walk_step"
    assert cfg.joint_vel >= 0, "--joint_vel is not negative"
    assert min(cfg.walk_step, cfg.balance, cfg.balance_margin, cfg.step_length, cfg.step_base, cfg.step_per_speed, cfg.step_width) >= 0, \
        "--walk_step, --balance, --balance_margin, --step_length, --step_base, --step_per_speed and --step_width are not negative"
    assert min(cfg.kl_bound, cfg.kl_beta, cfg.kl_beta_lr) >= 0 and (not cfg.kl_bound or (cfg.init_tracker and cfg.action_noise > 0)), \
        "--kl_bound, --kl_beta and --kl_beta_lr are not negative; --kl_bound needs --init_tracker and a positive --action_noise"
    assert min(cfg.upper_tilt, cfg.upper_rate) >= 0 and (not (cfg.upper_tilt or cfg.upper_rate) or cfg.upper_body), \
        "--upper_tilt and --upper_rate are not negative and need --upper_body"
    assert cfg.symmetry >= 0 and (cfg.init_tracker or not cfg.symmetry), "--symmetry is not negative and needs --init_tracker"
    assert 0 <= cfg.start_share <= 1 and bool(cfg.start_share) == bool(cfg.start_joints) and (cfg.start_share or not cfg.start_cmd), \
        "--start_share is a share and goes with --start_joints, --start_cmd needs them"
    assert not cfg.start_cmd or len(cfg.start_cmd.split(",")) == 3, "--start_cmd is lo:hi for vx, vy and wz"
    surfaces(cfg.friction_surfaces)
    gap = [float(x) for x in cfg.support_gap.split(",")]
    assert 0 <= cfg.support <= 1 and len(gap) == 2 and gap[0] <= gap[1], "--support is a share, --support_gap is lo,hi in m"
    assert not cfg.support_point or (cfg.support and len(cfg.support_point.split(",")) == 3), "--support_point is x,y,z in m and needs --support"
    assert 0 <= cfg.ext_share <= 1 and (cfg.layout == "1GPU" or not cfg.ext_joints), "--ext_share is a share, --ext_joints needs --layout 1GPU"
    assert task.SELF_COLLISION or cfg.self_clearance == cfg.self_clearance_term == 0, "--self_clearance and --self_clearance_term need robot.json's self_collision"
    flags = [f for f in cfg.xla_flags.split(",") if f.startswith("--")]
    if flags:  # XLA reads XLA_FLAGS when its backend starts, so they are set before XSoma(cfg) first uses the GPU
        os.environ["XLA_FLAGS"] = " ".join([os.environ.get("XLA_FLAGS", "")] + flags).strip()
    os.makedirs(os.path.join(cfg.log_dir, "ckpt"), exist_ok=True)
    with open(os.path.join(cfg.log_dir, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=1)
    wandb_log.start(dataclasses.asdict(cfg), cfg.log_dir)
    if cfg.layout != "1GPU":  # several GPUs: multi_gpu.run trains, logs, saves checkpoints and evaluates in the same formats
        from .multi_gpu import run

        return run(cfg, t_launch)
    t_build = time.time()
    xs = XSoma(cfg)
    carry = xs.init(jax.random.PRNGKey(cfg.seed))
    print(f"[xsoma] {cfg.terrain} x{cfg.terrain_scale} {cfg.foot} {cfg.preset}, {cfg.num_envs} envs, world-model input {xs.wm_in}, "
          f"buffer {sum(v.nbytes for v in jax.tree.leaves(carry['buf'])) / 2**30:.2f} GiB, device {jax.devices()[0]}", flush=True)  # fmt: skip
    t_c = time.time()
    progs = xs.compile_programs(carry)
    print(f"[xsoma] build {t_c - t_build:.0f} s, compile {time.time() - t_c:.0f} s, start-up {time.time() - t_launch:.0f} s since launch", flush=True)
    gc.collect()
    gc.freeze()  # Python's generation-2 collections would otherwise stall the host loop
    logf = open(os.path.join(cfg.log_dir, "log.jsonl"), "a", buffering=1)
    t0 = time.time()  # the training clock: compilation is over, the warm-up counts
    for _ in range(-(-cfg.warmup_steps // cfg.env_steps)):  # warm-up collection, rounded up to whole iterations
        carry, _ = progs["collect"](carry)
    # time of the last checkpoint, checkpoint counter, and training time and env steps at the previous log line (for the speed)
    last_ckpt, n_ckpt, prev_s, prev_env = 0.0, 0, 0.0, 0.0
    while True:
        cs, ls = [], []  # per-iteration statistics of the collection program and of the world-model and policy programs
        for _ in range(cfg.log_every):  # dispatched asynchronously; the host waits only for the statistics below
            carry, cstats = progs["collect"](carry)
            carry, wstats = progs["wm"](carry)
            carry, pstats = progs["pi"](carry)
            cs.append(cstats)
            ls.append({**wstats, **pstats})
        cs, ls, ev = jax.device_get((cs, ls, progs["wm_eval"](carry) if cfg.wm_eval else {}))
        n_done = max(sum(float(x["done"]) for x in cs), 1.0)
        rew = np.sum([x["rew_sums"] for x in cs], 0) / n_done
        stats = {"ep_len": sum(float(x["len_sum"]) for x in cs) / n_done, "done": n_done,
                 "contact": sum(float(x["terminated"]) for x in cs) / n_done, **{f"rew_{k}": float(v) for k, v in zip(task.TERMS, rew)},
                 **{k: float(np.mean([x[k] for x in cs])) for k in ("reward", "scale", "avg_len", "term_geom", "nacon", "nonfinite_env") + (("support",) if cfg.support else ())},
                 "diverged": sum(float(x["diverged"]) for x in cs),
                 **{k: float(np.mean([x[k] for x in ls])) for k in ls[0]}, **{k: float(v) for k, v in ev.items()}}  # fmt: skip
        train_s = time.time() - t0
        it, env_steps = int(carry["it"]), float(carry["env_steps"])
        rec = {"it": it, "train_s": train_s, "wall_s": time.time() - t_launch, "env_steps": env_steps, "nonfinite": int(carry["nonfinite"]), **stats}
        logf.write(json.dumps(rec) + "\n")
        wandb_log.log(rec, step=it)
        # The printed line: iteration, training time, env steps | len (mean length of finished episodes), lin and ang (Holosoma's
        # tracking rewards of finished episodes), r (reward per step), s (penalty scale) | wm (world-model loss), e8 (its error at
        # a window's 8th step) | G (imagined return), r_im (imagined reward per step), surv (imagined rollouts alive at the end),
        # gn (policy gradient norm) | speed | h_err (pelvis height error of the open-loop model after 8 and 32 steps), term
        # (termination rate of the simulator, of the model) | skipped policy updates (world-model gate, gradient gate) | physics
        # failures
        print(f"it {it} | {train_s / 60:.2f} min | {env_steps / 1e6:.1f}M steps | len {stats['ep_len']:.0f} lin {stats['rew_tracking_lin_vel']:.3f} "
              f"ang {stats['rew_tracking_ang_vel']:.3f} r {stats['reward']:.4f} s {stats['scale']:.3f} | wm {stats['wm']:.4f} e8 {stats['wm8']:.4f} | "
              f"G {stats['G']:.3f} r_im {stats['r']:.4f} surv {stats['surv']:.3f} gn {stats['gn']:.2f} | {cfg.log_every / max(train_s - prev_s, 1e-9):.1f} it/s "
              f"{(env_steps - prev_env) / max(train_s - prev_s, 1e-9) / 1e3:.0f}k steps/s"
              + (f" | h_err {stats['h_err8'] * 100:.1f}/{stats['h_errK'] * 100:.1f} cm term env {stats['term_env']:.3f} wm {stats['term_wm']:.3f}" if cfg.wm_eval else "")
              + (f" | gated wm {stats['wm_gated']:.2f} gn {stats['gn_gated']:.2f}" if stats["wm_gated"] or stats["gn_gated"] else "")
              + (f" | diverged {stats['diverged']:.0f}" if stats["diverged"] else "")
              + (f" | on rope {stats['support']:.3f}" if cfg.support else ""),
              flush=True)  # fmt: skip
        prev_s, prev_env = train_s, env_steps
        # a checkpoint every ckpt_minutes, and a last one when training ends
        if train_s - last_ckpt >= 60 * cfg.ckpt_minutes or train_s >= 60 * cfg.minutes:
            save_ckpt(os.path.join(cfg.log_dir, "ckpt", f"{n_ckpt:04d}"), carry["avg"], carry["pi"],
                      {"it": it, "train_s": train_s, "wall_s": time.time() - t_launch, "env_steps": env_steps}, xs.ckpt_extra)  # fmt: skip
            last_ckpt, n_ckpt = train_s, n_ckpt + 1
        if train_s >= 60 * cfg.minutes:
            break
    print(f"done: {it} iterations, {env_steps / 1e6:.1f}M env steps, {train_s / 60:.2f} min", flush=True)
    if cfg.eval_n > 0:
        from . import evaluate

        del carry, progs, xs  # free the training state before the evaluation builds its own simulator
        gc.collect()
        evaluate.main([cfg.log_dir, "--n", str(cfg.eval_n), "--scale", str(cfg.eval_scale), "--sole_friction", str(cfg.eval_sole_friction),
                       "--preset", cfg.preset] + (["--tracker_meta", cfg.tracker_meta] if cfg.init_tracker else []))  # fmt: skip


if __name__ == "__main__":
    main()
