from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

@dataclass(frozen=True)
class CommandScenario:
    label: str
    command: tuple[float, float, float]


@dataclass(frozen=True)
class RunSpec:
    run_name: str
    experiment: str
    reward_term: str
    reward_weight: float
    uses_penalty_scale_curriculum: bool
    resume_checkpoint: str | None
    category: str
    start_stage: str


@dataclass(frozen=True)
class PenaltyScalePoint:
    global_step: int | None
    train_num_samples: float | None
    penalty_scale: float


@dataclass(frozen=True)
class ResolvedRun:
    spec: RunSpec
    wandb_run_path: str
    checkpoint_name: str
    checkpoint_step: int
    penalty_scale_history: tuple[PenaltyScalePoint, ...]

    @property
    def checkpoint_uri(self) -> str:
        return f"wandb://{self.wandb_run_path}/{self.checkpoint_name}"


@dataclass(frozen=True)
class BenchmarkResult:
    metrics: dict[str, Any]
    arrays: dict[str, np.ndarray]


@dataclass
class AnalysisRuntime:
    analysis_cfg: Any
    env: Any
    device: str
    simulation_app: Any
    algo: Any | None = None


def parse_quiet_reward_commands(text: str) -> list[RunSpec]:
    blocks = _extract_python_command_blocks(text)
    specs: list[RunSpec] = []

    for block in blocks:
        command_text = block
        run_name = _extract_flag_value(command_text, "logger.name")
        experiment = _extract_experiment_name(command_text)
        reward_term, reward_weight = _extract_reward_override(command_text)
        uses_penalty_scale_curriculum = _uses_penalty_scale_curriculum(command_text)
        resume_checkpoint = _extract_flag_value(command_text, "training.checkpoint", required=False)

        category = _categorize_run(run_name, reward_term)
        start_stage = "resume_12k" if resume_checkpoint else "from_scratch"

        specs.append(
            RunSpec(
                run_name=run_name,
                experiment=experiment,
                reward_term=reward_term,
                reward_weight=reward_weight,
                uses_penalty_scale_curriculum=uses_penalty_scale_curriculum,
                resume_checkpoint=resume_checkpoint,
                category=category,
                start_stage=start_stage,
            )
        )

    return specs


def parse_quiet_reward_commands_file(path: str | Path) -> list[RunSpec]:
    return parse_quiet_reward_commands(Path(path).read_text())


def default_command_scenarios() -> list[CommandScenario]:
    return [
        CommandScenario(label="stand", command=(0.0, 0.0, 0.0)),
        CommandScenario(label="forward", command=(0.5, 0.0, 0.0)),
        CommandScenario(label="turn", command=(0.0, 0.0, 0.5)),
    ]


def parse_command_scenarios(values: list[str] | None) -> list[CommandScenario]:
    if not values:
        return default_command_scenarios()

    scenarios: list[CommandScenario] = []
    for raw_value in values:
        if ":" not in raw_value:
            raise ValueError(
                f"Invalid command scenario '{raw_value}'. Expected format '<label>:vx,vy,yaw'."
            )
        label, raw_command = raw_value.split(":", 1)
        parts = [piece.strip() for piece in raw_command.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"Invalid command scenario '{raw_value}'. Expected exactly three comma-separated values."
            )
        scenarios.append(CommandScenario(label=label.strip(), command=tuple(float(piece) for piece in parts)))
    return scenarios


def resolve_wandb_runs(
    entity: str,
    project: str,
    specs: list[RunSpec],
    include_run_names: set[str] | None = None,
) -> list[ResolvedRun]:
    import wandb

    requested_specs = [
        spec for spec in specs if include_run_names is None or spec.run_name in include_run_names
    ]
    requested_names = {spec.run_name for spec in requested_specs}

    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")

    name_to_run: dict[str, Any] = {}
    for run in runs:
        candidate_names = {
            getattr(run, "name", None),
            getattr(run, "display_name", None),
            run.config.get("name") if getattr(run, "config", None) else None,
        }
        for candidate_name in candidate_names:
            if candidate_name in requested_names and candidate_name not in name_to_run:
                name_to_run[candidate_name] = run

    missing = requested_names - set(name_to_run)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Failed to resolve the following run names from W&B: {missing_list}")

    resolved_runs: list[ResolvedRun] = []
    for spec in requested_specs:
        run = name_to_run[spec.run_name]
        checkpoint_name, checkpoint_step = _select_latest_checkpoint(run)
        penalty_scale_history = (
            _fetch_penalty_scale_history(run) if spec.uses_penalty_scale_curriculum else ()
        )
        resolved_runs.append(
            ResolvedRun(
                spec=spec,
                wandb_run_path=f"{entity}/{project}/{run.id}",
                checkpoint_name=checkpoint_name,
                checkpoint_step=checkpoint_step,
                penalty_scale_history=penalty_scale_history,
            )
        )

    return resolved_runs


def run_analysis(
    resolved_runs: list[ResolvedRun],
    scenarios: list[CommandScenario],
    output_dir: Path,
    *,
    num_envs: int,
    num_steps: int,
    episode_length_s: float,
    contact_threshold: float,
    headless: bool,
) -> list[dict[str, Any]]:
    logger = _get_logger()

    results: list[dict[str, Any]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime: AnalysisRuntime | None = None

    try:
        for resolved_run in resolved_runs:
            logger.info(f"Analyzing run '{resolved_run.spec.run_name}' from {resolved_run.checkpoint_uri}")
            per_run_dir = output_dir / resolved_run.spec.run_name
            per_run_dir.mkdir(parents=True, exist_ok=True)

            if runtime is None:
                runtime = _create_analysis_runtime(
                    checkpoint_uri=resolved_run.checkpoint_uri,
                    log_dir=per_run_dir,
                    num_envs=num_envs,
                    episode_length_s=episode_length_s,
                    headless=headless,
                )

            algo = _load_algo_for_runtime(
                runtime=runtime,
                checkpoint_uri=resolved_run.checkpoint_uri,
                log_dir=per_run_dir,
            )

            for scenario in scenarios:
                benchmark_result = _evaluate_fast_sac_checkpoint(
                    algo=algo,
                    scenario=scenario,
                    num_steps=num_steps,
                    contact_threshold=contact_threshold,
                )
                scenario_metrics = benchmark_result.metrics
                result_row = {
                    "run_name": resolved_run.spec.run_name,
                    "wandb_run_path": resolved_run.wandb_run_path,
                    "checkpoint_name": resolved_run.checkpoint_name,
                    "checkpoint_step": resolved_run.checkpoint_step,
                    "experiment": resolved_run.spec.experiment,
                    "reward_term": resolved_run.spec.reward_term,
                    "reward_weight": resolved_run.spec.reward_weight,
                    "effective_reward_weight_at_checkpoint": _effective_reward_weight_at_checkpoint(
                        resolved_run
                    ),
                    "penalty_scale_at_checkpoint": _penalty_scale_at_checkpoint(resolved_run),
                    "category": resolved_run.spec.category,
                    "start_stage": resolved_run.spec.start_stage,
                    "resume_checkpoint": resolved_run.spec.resume_checkpoint or "",
                    "scenario": scenario.label,
                    "command_x": scenario.command[0],
                    "command_y": scenario.command[1],
                    "command_yaw": scenario.command[2],
                }
                result_row.update(scenario_metrics)
                results.append(result_row)
                _save_benchmark_arrays(
                    output_dir=per_run_dir,
                    resolved_run=resolved_run,
                    scenario=scenario,
                    benchmark_result=benchmark_result,
                )
            gc.collect()
            _maybe_empty_cuda_cache()
    finally:
        from holosoma.utils.sim_utils import close_simulation_app

        if runtime is not None and runtime.simulation_app is not None:
            close_simulation_app(runtime.simulation_app)

    return results


def write_results(results: list[dict[str, Any]], output_dir: Path, *, generate_plots: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = output_dir / "quiet_run_analysis_summary.csv"
    summary_json = output_dir / "quiet_run_analysis_summary.json"

    fieldnames = sorted({key for row in results for key in row})
    with summary_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    summary_json.write_text(json.dumps(results, indent=2))
    _save_aggregate_numpy(results, output_dir)
    if generate_plots:
        _generate_plots(results, output_dir)


def _save_penalty_scale_history(
    resolved_runs: list[ResolvedRun],
    output_dir: Path,
    *,
    generate_plots: bool,
) -> None:
    rows: list[dict[str, Any]] = []
    for resolved_run in resolved_runs:
        for index, point in enumerate(resolved_run.penalty_scale_history):
            rows.append(
                {
                    "run_name": resolved_run.spec.run_name,
                    "wandb_run_path": resolved_run.wandb_run_path,
                    "history_index": index,
                    "global_step": _empty_if_none(point.global_step),
                    "train_num_samples": _empty_if_none(point.train_num_samples),
                    "penalty_scale": point.penalty_scale,
                    "reward_weight": resolved_run.spec.reward_weight,
                    "effective_reward_weight": resolved_run.spec.reward_weight * point.penalty_scale,
                    "effective_reward_weight_magnitude": abs(
                        resolved_run.spec.reward_weight * point.penalty_scale
                    ),
                }
            )

    history_csv = output_dir / "penalty_scale_history.csv"
    fieldnames = [
        "run_name",
        "wandb_run_path",
        "history_index",
        "global_step",
        "train_num_samples",
        "penalty_scale",
        "reward_weight",
        "effective_reward_weight",
        "effective_reward_weight_magnitude",
    ]
    with history_csv.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if rows and generate_plots:
        _plot_penalty_scale_history(resolved_runs, output_dir)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Offline analysis for quiet-reward locomotion runs stored in Weights & Biases."
    )
    parser.add_argument(
        "--manifest",
        default="quiet_reward_run_commands.txt",
        help="Path to the quiet reward run manifest containing the original training commands.",
    )
    parser.add_argument("--wandb-entity", required=True, help="Weights & Biases entity that owns the runs.")
    parser.add_argument("--wandb-project", required=True, help="Weights & Biases project that owns the runs.")
    parser.add_argument(
        "--include-run-name",
        action="append",
        default=None,
        help="Optional exact run name filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Evaluation command scenario in the format '<label>:vx,vy,yaw'. Can be passed multiple times.",
    )
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel environments for evaluation.")
    parser.add_argument("--num-steps", type=int, default=2000, help="Number of control steps per scenario.")
    parser.add_argument(
        "--episode-length-s",
        type=float,
        default=20.0,
        help="Episode length used during offline evaluation.",
    )
    parser.add_argument(
        "--contact-threshold",
        type=float,
        default=1.0,
        help="Vertical contact force threshold used to detect touchdown.",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/quiet_reward",
        help="Directory where CSV and JSON summaries will be written.",
    )
    _add_bool_flag(
        parser,
        name="headless",
        default=True,
        help_text="Run the evaluation headless.",
    )
    _add_bool_flag(
        parser,
        name="plots",
        default=True,
        help_text="Generate benchmark plots from the saved NumPy arrays.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    scenarios = parse_command_scenarios(args.scenario)
    specs = parse_quiet_reward_commands_file(args.manifest)
    include_run_names = set(args.include_run_name) if args.include_run_name else None
    resolved_runs = resolve_wandb_runs(
        entity=args.wandb_entity,
        project=args.wandb_project,
        specs=specs,
        include_run_names=include_run_names,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "resolved_runs.json").write_text(
        json.dumps(
            [
                {
                    "run_name": resolved.spec.run_name,
                    "wandb_run_path": resolved.wandb_run_path,
                    "checkpoint_name": resolved.checkpoint_name,
                    "checkpoint_step": resolved.checkpoint_step,
                    "checkpoint_uri": resolved.checkpoint_uri,
                    "penalty_scale_at_checkpoint": _penalty_scale_at_checkpoint(resolved),
                    "effective_reward_weight_at_checkpoint": _effective_reward_weight_at_checkpoint(resolved),
                    "penalty_scale_history_count": len(resolved.penalty_scale_history),
                }
                for resolved in resolved_runs
            ],
            indent=2,
        )
    )
    _save_penalty_scale_history(resolved_runs, output_dir, generate_plots=args.plots)

    results = run_analysis(
        resolved_runs=resolved_runs,
        scenarios=scenarios,
        output_dir=output_dir,
        num_envs=args.num_envs,
        num_steps=args.num_steps,
        episode_length_s=args.episode_length_s,
        contact_threshold=args.contact_threshold,
        headless=args.headless,
    )
    write_results(results, output_dir, generate_plots=args.plots)


def _extract_flag_value(command_text: str, flag_name: str, *, required: bool = True) -> str | None:
    pattern = rf"--{re.escape(flag_name)}(?:=|\s+)([^\s\\]+)"
    match = re.search(pattern, command_text)
    if match:
        return match.group(1)
    if required:
        raise ValueError(f"Could not find '--{flag_name}' in command:\n{command_text}")
    return None


def _extract_python_command_blocks(text: str) -> list[str]:
    blocks: list[str] = []
    current_block: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()

        if not stripped:
            if current_block:
                current_block.append(line)
            continue

        if stripped.startswith("#"):
            if current_block:
                current_block.append(line)
            continue

        if stripped.startswith("python "):
            if current_block:
                blocks.append("\n".join(current_block).strip())
            current_block = [stripped]
            continue

        if current_block:
            current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block).strip())

    return blocks


def _add_bool_flag(parser: argparse.ArgumentParser, *, name: str, default: bool, help_text: str) -> None:
    parser.set_defaults(**{name: default})
    parser.add_argument(
        f"--{name}",
        dest=name,
        action="store_true",
        help=help_text,
    )
    parser.add_argument(
        f"--no-{name}",
        dest=name,
        action="store_false",
        help=f"Disable: {help_text.lower()}",
    )


def _extract_experiment_name(command_text: str) -> str:
    match = re.search(r"exp:([^\s\\]+)", command_text)
    if not match:
        raise ValueError(f"Could not find experiment name in command:\n{command_text}")
    return match.group(1)


def _extract_reward_override(command_text: str) -> tuple[str, float]:
    match = re.search(r"--reward\.terms\.([^.]+)\.weight=([-+]?[\d.]+)", command_text)
    if not match:
        raise ValueError(f"Could not find reward override in command:\n{command_text}")
    return match.group(1), float(match.group(2))


def _uses_penalty_scale_curriculum(command_text: str) -> bool:
    flag_prefix = "curriculum.setup-terms.penalty-curriculum.params"
    return _extract_flag_value(command_text, f"{flag_prefix}.max-scale", required=False) is not None


def _categorize_run(run_name: str, reward_term: str) -> str:
    if run_name == "basic_fast_sac":
        return "baseline"
    if "olaf" in reward_term or "olaf" in run_name:
        return "olaf"
    if "not-gated" in reward_term:
        return "not_gated"
    if "gated" in reward_term:
        return "gated"
    return "baseline"


def _fetch_penalty_scale_history(run: Any) -> tuple[PenaltyScalePoint, ...]:
    logger = _get_logger()
    points = _scan_penalty_scale_history(run, "Env/penalty_scale")
    if not points:
        points = _scan_penalty_scale_history(run, "penalty_scale")
    if points:
        logger.info(f"Loaded {len(points)} penalty_scale points for W&B run '{run.id}'.")
    return tuple(points)


def _scan_penalty_scale_history(run: Any, key: str) -> list[PenaltyScalePoint]:
    logger = _get_logger()
    key_sets = (
        ["global_step", "Train/num_samples", key],
        ["global_step", key],
        [key],
    )
    for keys in key_sets:
        points: list[PenaltyScalePoint] = []
        try:
            history_iter = run.scan_history(keys=keys)
            for row in history_iter:
                value = row.get(key)
                if value is None:
                    continue
                points.append(
                    PenaltyScalePoint(
                        global_step=_optional_int(row.get("global_step")),
                        train_num_samples=_optional_float(row.get("Train/num_samples")),
                        penalty_scale=float(value),
                    )
                )
        except Exception as exc:
            logger.warning(f"Could not load '{key}' history for W&B run '{run.id}': {exc}")
            return []
        if points:
            return points
    return []


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if math.isnan(float(value)):
            return None
    except (TypeError, ValueError):
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    numeric_value = _optional_float(value)
    return int(numeric_value) if numeric_value is not None else None


def _select_latest_checkpoint(run: Any) -> tuple[str, int]:
    checkpoint_candidates: list[tuple[int, str]] = []

    for file_obj in run.files():
        file_name = file_obj.name
        match = re.match(r"model_(\d+)\.pt$", Path(file_name).name)
        if match:
            checkpoint_candidates.append((int(match.group(1)), file_name))

    if not checkpoint_candidates:
        raise ValueError(f"No checkpoint files matching 'model_<step>.pt' found for run '{run.id}'.")

    checkpoint_step, checkpoint_name = max(checkpoint_candidates, key=lambda item: item[0])
    return checkpoint_name, checkpoint_step


def _create_analysis_runtime(
    *,
    checkpoint_uri: str,
    log_dir: Path,
    num_envs: int,
    episode_length_s: float,
    headless: bool,
):
    import holosoma.config_values.logger
    from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
    from holosoma.utils.sim_utils import setup_simulation_environment

    checkpoint_cfg = CheckpointConfig(checkpoint=checkpoint_uri)
    saved_cfg, _saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    analysis_cfg = _build_analysis_config(
        saved_cfg=saved_cfg,
        num_envs=num_envs,
        episode_length_s=episode_length_s,
        headless=headless,
    )
    env, device, simulation_app = setup_simulation_environment(analysis_cfg)
    return AnalysisRuntime(
        analysis_cfg=analysis_cfg,
        env=env,
        device=device,
        simulation_app=simulation_app,
    )


def _load_algo_for_runtime(
    *,
    runtime: AnalysisRuntime,
    checkpoint_uri: str,
    log_dir: Path,
):
    from holosoma.utils.eval_utils import CheckpointConfig, load_checkpoint, load_saved_experiment_config
    from holosoma.utils.helpers import get_class

    checkpoint_cfg = CheckpointConfig(checkpoint=checkpoint_uri)
    saved_cfg, saved_wandb_path = load_saved_experiment_config(checkpoint_cfg)
    if saved_cfg.algo._target_ != runtime.analysis_cfg.algo._target_:
        raise ValueError(
            "All analyzed runs must use the same algorithm target when reusing the same evaluation runtime. "
            f"Expected {runtime.analysis_cfg.algo._target_}, got {saved_cfg.algo._target_}."
        )

    if runtime.algo is None:
        algo_class = get_class(runtime.analysis_cfg.algo._target_)
        runtime.algo = algo_class(
            device=runtime.device,
            env=runtime.env,
            config=runtime.analysis_cfg.algo.config,
            log_dir=str(log_dir),
            multi_gpu_cfg=None,
        )
        runtime.algo.setup()

    algo = runtime.algo
    algo.log_dir = str(log_dir)
    algo.attach_checkpoint_metadata(saved_cfg, saved_wandb_path)

    checkpoint_path = load_checkpoint(checkpoint_uri, str(log_dir))
    algo.load(str(checkpoint_path))
    return algo


def _build_analysis_config(saved_cfg, *, num_envs: int, episode_length_s: float, headless: bool):
    import holosoma.config_values.logger

    spawn_cfg = dataclasses.replace(
        saved_cfg.terrain.terrain_term.spawn,
        randomize_tiles=False,
        xy_offset_range=0.0,
    )
    terrain_cfg = dataclasses.replace(
        saved_cfg.terrain,
        terrain_term=dataclasses.replace(saved_cfg.terrain.terrain_term, spawn=spawn_cfg),
    )
    simulator_cfg = dataclasses.replace(
        saved_cfg.simulator,
        config=dataclasses.replace(
            saved_cfg.simulator.config,
            sim=dataclasses.replace(
                saved_cfg.simulator.config.sim,
                max_episode_length_s=episode_length_s,
            ),
        ),
    )
    training_cfg = dataclasses.replace(
        saved_cfg.training,
        headless=headless,
        num_envs=num_envs,
        max_eval_steps=None,
        export_onnx=False,
    )

    return dataclasses.replace(
        saved_cfg,
        terrain=terrain_cfg,
        simulator=simulator_cfg,
        training=training_cfg,
        logger=holosoma.config_values.logger.disabled,
    )


def _evaluate_fast_sac_checkpoint(
    *,
    algo,
    scenario: CommandScenario,
    num_steps: int,
    contact_threshold: float,
) -> BenchmarkResult:
    from holosoma.agents.fast_sac.fast_sac_agent import FastSACAgent
    from holosoma.managers.observation.terms.locomotion import (
        get_base_ang_vel,
        get_base_lin_vel,
        get_projected_gravity,
    )
    from holosoma.utils.safe_torch_import import torch

    if not isinstance(algo, FastSACAgent):
        raise TypeError(
            f"This analysis script currently supports FastSACAgent checkpoints only, got {type(algo).__name__}."
        )

    env = algo.unwrapped_env
    wrapped_env = algo.env
    policy = algo.get_inference_policy(device=algo.device)

    command_tensor = torch.tensor(scenario.command, device=env.device, dtype=env.command_manager.commands.dtype)
    env.set_is_evaluating()
    obs = wrapped_env.reset()
    obs = _apply_fixed_command_and_rebuild_actor_obs(
        env=env,
        wrapped_env=wrapped_env,
        command_tensor=command_tensor,
        torch=torch,
    )

    touchdown_vz_samples: list[torch.Tensor] = []
    touchdown_fz_samples: list[torch.Tensor] = []
    contact_fz_samples: list[torch.Tensor] = []
    foot_fz_all_samples: list[torch.Tensor] = []
    raw_tracking_lin_samples: list[torch.Tensor] = []
    raw_tracking_ang_samples: list[torch.Tensor] = []
    lin_vel_samples: list[torch.Tensor] = []
    yaw_rate_samples: list[torch.Tensor] = []
    lin_vel_error_trace: list[torch.Tensor] = []
    yaw_rate_error_trace: list[torch.Tensor] = []
    foot_fz_all_trace: list[torch.Tensor] = []
    gravity_xy_trace: list[torch.Tensor] = []
    command_trace: list[torch.Tensor] = []
    done_count_trace: list[int] = []
    timeout_count_trace: list[int] = []
    fall_count_trace: list[int] = []

    prev_contact = env.simulator.contact_forces[:, env.feet_indices, 2] > contact_threshold
    prev_foot_vz = env.simulator._rigid_body_vel[:, env.feet_indices, 2].clone()

    sample_count = env.num_envs * num_steps
    lin_vel_sq_error_sum = 0.0
    lin_vel_error_sum = 0.0
    yaw_rate_sq_error_sum = 0.0
    yaw_rate_error_sum = 0.0
    gravity_xy_norm_sum = 0.0
    fall_count = 0
    timeout_count = 0
    episode_count = 0

    with torch.inference_mode():
        for _ in range(num_steps):
            _apply_fixed_command(env=env, command_tensor=command_tensor)
            actions = policy({"actor_obs": obs})
            obs, _, dones, extras = wrapped_env.step(actions)
            _apply_fixed_command(env=env, command_tensor=command_tensor)

            commands = env.command_manager.commands
            lin_vel = get_base_lin_vel(env)[:, :2]
            yaw_rate = get_base_ang_vel(env)[:, 2]
            gravity_xy_norm = torch.linalg.norm(get_projected_gravity(env)[:, :2], dim=1)
            lin_vel_error = torch.linalg.norm(commands[:, :2] - lin_vel, dim=1)
            yaw_rate_error = torch.abs(commands[:, 2] - yaw_rate)

            lin_vel_sq_error_sum += torch.sum(torch.square(commands[:, :2] - lin_vel)).item()
            lin_vel_error_sum += lin_vel_error.sum().item()
            yaw_rate_sq_error_sum += torch.sum(torch.square(commands[:, 2] - yaw_rate)).item()
            yaw_rate_error_sum += yaw_rate_error.sum().item()
            gravity_xy_norm_sum += gravity_xy_norm.sum().item()
            lin_vel_samples.append(lin_vel.detach().cpu())
            yaw_rate_samples.append(yaw_rate.detach().cpu())
            lin_vel_error_trace.append(lin_vel_error.detach().cpu())
            yaw_rate_error_trace.append(yaw_rate_error.detach().cpu())
            gravity_xy_trace.append(gravity_xy_norm.detach().cpu())
            command_trace.append(commands[0].detach().cpu())

            contact_fz = torch.clamp(env.simulator.contact_forces[:, env.feet_indices, 2], min=0.0)
            foot_fz_all_samples.append(contact_fz.reshape(-1).detach().cpu())
            foot_fz_all_trace.append(contact_fz.detach().cpu())
            contact_now = contact_fz > contact_threshold
            touchdown_now = contact_now & ~prev_contact
            downward_speed = torch.clamp(-prev_foot_vz, min=0.0)

            if touchdown_now.any():
                touchdown_vz_samples.append(downward_speed[touchdown_now].detach().cpu())
                touchdown_fz_samples.append(contact_fz[touchdown_now].detach().cpu())
            if contact_now.any():
                contact_fz_samples.append(contact_fz[contact_now].detach().cpu())

            done_mask = dones.bool()
            if done_mask.any():
                timeout_mask = extras["time_outs"].bool()
                fall_mask = done_mask & ~timeout_mask
                fall_events = int(fall_mask.sum().item())
                timeout_events = int((done_mask & timeout_mask).sum().item())
                done_events = int(done_mask.sum().item())
                fall_count += fall_events
                timeout_count += timeout_events
                episode_count += done_events

                raw_episode = extras.get("raw_episode", {})
                if "raw_rew_tracking_lin_vel" in raw_episode:
                    raw_tracking_lin_samples.append(raw_episode["raw_rew_tracking_lin_vel"].detach().cpu())
                if "raw_rew_tracking_ang_vel" in raw_episode:
                    raw_tracking_ang_samples.append(raw_episode["raw_rew_tracking_ang_vel"].detach().cpu())
                done_count_trace.append(done_events)
                timeout_count_trace.append(timeout_events)
                fall_count_trace.append(fall_events)
            else:
                done_count_trace.append(0)
                timeout_count_trace.append(0)
                fall_count_trace.append(0)

            prev_contact = contact_now.clone()
            prev_foot_vz = env.simulator._rigid_body_vel[:, env.feet_indices, 2].clone()
            if done_mask.any():
                fixed_obs = _rebuild_actor_obs_without_history_update(env=env, wrapped_env=wrapped_env, torch=torch)
                obs[done_mask] = fixed_obs[done_mask]

    lin_vel_rmse = math.sqrt(lin_vel_sq_error_sum / max(sample_count, 1))
    lin_vel_error_mean = lin_vel_error_sum / max(sample_count, 1)
    yaw_rate_rmse = math.sqrt(yaw_rate_sq_error_sum / max(sample_count, 1))
    yaw_rate_error_mean = yaw_rate_error_sum / max(sample_count, 1)
    gravity_xy_mean = gravity_xy_norm_sum / max(sample_count, 1)
    mean_episode_length_s = float(env.average_episode_length) * float(env.dt)

    touchdown_vz_tensor = _concat_samples(touchdown_vz_samples)
    touchdown_fz_tensor = _concat_samples(touchdown_fz_samples)
    contact_fz_tensor = _concat_samples(contact_fz_samples)
    foot_fz_all_tensor = _concat_samples(foot_fz_all_samples)
    raw_tracking_lin_tensor = _concat_samples(raw_tracking_lin_samples)
    raw_tracking_ang_tensor = _concat_samples(raw_tracking_ang_samples)
    metrics = {
        "num_envs": env.num_envs,
        "num_steps": num_steps,
        "lin_vel_rmse": lin_vel_rmse,
        "lin_vel_error_mean": lin_vel_error_mean,
        "yaw_rate_rmse": yaw_rate_rmse,
        "yaw_rate_error_mean": yaw_rate_error_mean,
        "gravity_xy_mean": gravity_xy_mean,
        "episode_count": episode_count,
        "fall_count": fall_count,
        "timeout_count": timeout_count,
        "fall_rate": (fall_count / episode_count) if episode_count else float("nan"),
        "mean_episode_length_s": mean_episode_length_s,
        "touchdown_count": int(touchdown_vz_tensor.numel()),
        "touchdown_vz_mean": _safe_mean(touchdown_vz_tensor),
        "touchdown_vz_p95": _safe_quantile(touchdown_vz_tensor, 0.95),
        "touchdown_fz_mean": _safe_mean(touchdown_fz_tensor),
        "touchdown_fz_p95": _safe_quantile(touchdown_fz_tensor, 0.95),
        "contact_fz_mean": _safe_mean(contact_fz_tensor),
        "contact_fz_p95": _safe_quantile(contact_fz_tensor, 0.95),
        "foot_fz_all_mean": _safe_mean(foot_fz_all_tensor),
        "foot_fz_all_p95": _safe_quantile(foot_fz_all_tensor, 0.95),
        "raw_tracking_lin_mean": _safe_mean(raw_tracking_lin_tensor),
        "raw_tracking_ang_mean": _safe_mean(raw_tracking_ang_tensor),
    }
    arrays = {
        "command": np.asarray(scenario.command, dtype=np.float32),
        "command_trace": _stack_step_vectors(command_trace),
        "lin_vel_xy": _stack_step_tensors(lin_vel_samples),
        "yaw_rate": _stack_step_tensors(yaw_rate_samples),
        "lin_vel_error_trace": _stack_step_means(lin_vel_error_trace),
        "yaw_rate_error_trace": _stack_step_means(yaw_rate_error_trace),
        "gravity_xy_trace": _stack_step_means(gravity_xy_trace),
        "foot_fz_all_trace": _stack_step_tensors(foot_fz_all_trace),
        "foot_fz_all": _tensor_to_numpy(foot_fz_all_tensor),
        "touchdown_vz": _tensor_to_numpy(touchdown_vz_tensor),
        "touchdown_fz": _tensor_to_numpy(touchdown_fz_tensor),
        "contact_fz": _tensor_to_numpy(contact_fz_tensor),
        "raw_tracking_lin": _tensor_to_numpy(raw_tracking_lin_tensor),
        "raw_tracking_ang": _tensor_to_numpy(raw_tracking_ang_tensor),
        "done_count_trace": np.asarray(done_count_trace, dtype=np.int32),
        "timeout_count_trace": np.asarray(timeout_count_trace, dtype=np.int32),
        "fall_count_trace": np.asarray(fall_count_trace, dtype=np.int32),
    }
    return BenchmarkResult(metrics=metrics, arrays=arrays)


def _apply_fixed_command_and_rebuild_actor_obs(*, env, wrapped_env, command_tensor, torch):
    _apply_fixed_command(env=env, command_tensor=command_tensor)
    return _rebuild_actor_obs_without_history_update(env=env, wrapped_env=wrapped_env, torch=torch)


def _apply_fixed_command(*, env, command_tensor) -> None:
    env.command_manager.commands[:] = command_tensor.view(1, -1).expand_as(env.command_manager.commands)


def _rebuild_actor_obs_without_history_update(*, env, wrapped_env, torch):
    obs_dict = env.observation_manager.compute(modify_history=False)
    clip_limit = env.observation_manager.cfg.clip_observations
    actor_obs_keys = getattr(wrapped_env, "_actor_obs_keys")
    return torch.cat([torch.clip(obs_dict[key], -clip_limit, clip_limit) for key in actor_obs_keys], dim=1)


def _save_benchmark_arrays(
    *,
    output_dir: Path,
    resolved_run: ResolvedRun,
    scenario: CommandScenario,
    benchmark_result: BenchmarkResult,
) -> None:
    scenario_stem = _slugify(scenario.label)
    npz_path = output_dir / f"{scenario_stem}_benchmark_data.npz"
    metadata = {
        "run_name": resolved_run.spec.run_name,
        "scenario": scenario.label,
        "command": list(scenario.command),
        "category": resolved_run.spec.category,
        "reward_weight": resolved_run.spec.reward_weight,
        "penalty_scale_at_checkpoint": _penalty_scale_at_checkpoint(resolved_run),
        "effective_reward_weight_at_checkpoint": _effective_reward_weight_at_checkpoint(resolved_run),
        "checkpoint_step": resolved_run.checkpoint_step,
    }
    np.savez_compressed(
        npz_path,
        metadata_json=np.asarray(json.dumps(metadata)),
        **benchmark_result.arrays,
    )


def _save_aggregate_numpy(results: list[dict[str, Any]], output_dir: Path) -> None:
    metric_names = _metric_names_for_plots()
    run_names = sorted({row["run_name"] for row in results})
    scenario_names = sorted({row["scenario"] for row in results})
    metric_cube = np.full((len(run_names), len(scenario_names), len(metric_names)), np.nan, dtype=np.float32)

    run_index = {run_name: idx for idx, run_name in enumerate(run_names)}
    scenario_index = {scenario_name: idx for idx, scenario_name in enumerate(scenario_names)}

    category_by_run: dict[str, str] = {}
    weight_by_run: dict[str, float] = {}
    effective_weight_by_run: dict[str, float] = {}
    penalty_scale_by_run: dict[str, float] = {}
    checkpoint_step_by_run: dict[str, int] = {}

    for row in results:
        run_idx = run_index[row["run_name"]]
        scenario_idx = scenario_index[row["scenario"]]
        category_by_run[row["run_name"]] = str(row["category"])
        weight_by_run[row["run_name"]] = float(row["reward_weight"])
        effective_weight_by_run[row["run_name"]] = float(row["effective_reward_weight_at_checkpoint"])
        penalty_scale_by_run[row["run_name"]] = float(row["penalty_scale_at_checkpoint"])
        checkpoint_step_by_run[row["run_name"]] = int(row["checkpoint_step"])
        for metric_idx, metric_name in enumerate(metric_names):
            metric_cube[run_idx, scenario_idx, metric_idx] = np.float32(row.get(metric_name, np.nan))

    np.savez_compressed(
        output_dir / "quiet_run_benchmarks.npz",
        run_names=np.asarray(run_names),
        scenario_names=np.asarray(scenario_names),
        metric_names=np.asarray(metric_names),
        categories=np.asarray([category_by_run[name] for name in run_names]),
        reward_weights=np.asarray([weight_by_run[name] for name in run_names], dtype=np.float32),
        effective_reward_weights=np.asarray([effective_weight_by_run[name] for name in run_names], dtype=np.float32),
        penalty_scales=np.asarray([penalty_scale_by_run[name] for name in run_names], dtype=np.float32),
        checkpoint_steps=np.asarray([checkpoint_step_by_run[name] for name in run_names], dtype=np.int32),
        metric_cube=metric_cube,
    )


def _generate_plots(results: list[dict[str, Any]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _plot_metric_heatmaps(results, plots_dir, plt)
    _plot_metric_vs_weight(results, plots_dir, plt)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("holosoma.analyze_quiet_runs")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def _maybe_empty_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        return


def _plot_metric_heatmaps(results: list[dict[str, Any]], plots_dir: Path, plt) -> None:
    metric_names = _metric_names_for_plots()
    scenario_names = sorted({row["scenario"] for row in results})
    run_names = sorted(
        {row["run_name"] for row in results},
        key=lambda name: (
            next(row["category"] for row in results if row["run_name"] == name),
            next(_plot_weight_value(row) for row in results if row["run_name"] == name),
            name,
        ),
    )
    row_index = {run_name: idx for idx, run_name in enumerate(run_names)}
    scenario_index = {scenario_name: idx for idx, scenario_name in enumerate(scenario_names)}

    for metric_name in metric_names:
        matrix = np.full((len(run_names), len(scenario_names)), np.nan, dtype=np.float32)
        for row in results:
            matrix[row_index[row["run_name"]], scenario_index[row["scenario"]]] = np.float32(
                row.get(metric_name, np.nan)
            )

        figure_height = max(6, len(run_names) * 0.35)
        figure_width = max(8, len(scenario_names) * 2.5)
        fig, ax = plt.subplots(figsize=(figure_width, figure_height))
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
        ax.set_title(metric_name)
        ax.set_xticks(np.arange(len(scenario_names)))
        ax.set_xticklabels(scenario_names, rotation=25, ha="right")
        ax.set_yticks(np.arange(len(run_names)))
        ax.set_yticklabels(run_names, fontsize=8)
        fig.colorbar(image, ax=ax, label=metric_name)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric_name}_heatmap.png", dpi=200)
        plt.close(fig)


def _plot_metric_vs_weight(results: list[dict[str, Any]], plots_dir: Path, plt) -> None:
    metric_names = _metric_names_for_plots()
    scenario_names = sorted({row["scenario"] for row in results})

    for metric_name in metric_names:
        fig, axes = plt.subplots(
            nrows=len(scenario_names),
            ncols=1,
            figsize=(9, max(4, 3 * len(scenario_names))),
            squeeze=False,
        )
        baseline_rows = [row for row in results if row["category"] == "baseline"]

        for axis, scenario_name in zip(axes.flatten(), scenario_names):
            scenario_rows = [row for row in results if row["scenario"] == scenario_name]
            for category, color in (
                ("not_gated", "tab:blue"),
                ("gated", "tab:orange"),
                ("olaf", "tab:red"),
            ):
                category_rows = sorted(
                    (row for row in scenario_rows if row["category"] == category),
                    key=_plot_weight_value,
                )
                if not category_rows:
                    continue
                weights = [_plot_weight_value(row) for row in category_rows]
                values = [float(row.get(metric_name, np.nan)) for row in category_rows]
                axis.plot(weights, values, marker="o", label=category, color=color)

            baseline_for_scenario = [row for row in baseline_rows if row["scenario"] == scenario_name]
            if baseline_for_scenario:
                baseline_value = float(baseline_for_scenario[0].get(metric_name, np.nan))
                axis.axhline(baseline_value, color="tab:green", linestyle="--", label="baseline")

            axis.set_title(f"{metric_name} | {scenario_name}")
            axis.set_xlabel("Effective Reward Weight Magnitude")
            axis.set_ylabel(metric_name)
            axis.grid(True, alpha=0.3)
            axis.legend()

        fig.tight_layout()
        fig.savefig(plots_dir / f"{metric_name}_by_weight.png", dpi=200)
        plt.close(fig)


def _plot_penalty_scale_history(resolved_runs: list[ResolvedRun], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    runs_with_history = [resolved_run for resolved_run in resolved_runs if resolved_run.penalty_scale_history]
    if not runs_with_history:
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8), squeeze=False)
    scale_axis, weight_axis = axes.flatten()
    for resolved_run in runs_with_history:
        points = resolved_run.penalty_scale_history
        x_values = [
            point.global_step if point.global_step is not None else index
            for index, point in enumerate(points)
        ]
        scale_values = [point.penalty_scale for point in points]
        effective_weight_values = [
            abs(resolved_run.spec.reward_weight * point.penalty_scale) for point in points
        ]
        scale_axis.plot(x_values, scale_values, marker="o", markersize=3, label=resolved_run.spec.run_name)
        weight_axis.plot(
            x_values,
            effective_weight_values,
            marker="o",
            markersize=3,
            label=resolved_run.spec.run_name,
        )

    scale_axis.set_title("penalty_scale")
    scale_axis.set_xlabel("global_step")
    scale_axis.set_ylabel("penalty_scale")
    scale_axis.grid(True, alpha=0.3)
    scale_axis.legend(fontsize=8)

    weight_axis.set_title("Effective Quiet Penalty Weight")
    weight_axis.set_xlabel("global_step")
    weight_axis.set_ylabel("|reward_weight * penalty_scale|")
    weight_axis.grid(True, alpha=0.3)
    weight_axis.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(plots_dir / "penalty_scale_history.png", dpi=200)
    plt.close(fig)


def _plot_weight_value(row: dict[str, Any]) -> float:
    effective_weight = _optional_float(row.get("effective_reward_weight_at_checkpoint"))
    if effective_weight is not None:
        return abs(effective_weight)
    return abs(float(row["reward_weight"]))


def _concat_samples(samples):
    from holosoma.utils.safe_torch_import import torch

    if not samples:
        return torch.empty(0)
    return torch.cat(samples)


def _stack_step_means(samples) -> np.ndarray:
    if not samples:
        return np.empty(0, dtype=np.float32)
    return np.asarray([sample.float().mean().item() for sample in samples], dtype=np.float32)


def _stack_step_vectors(samples) -> np.ndarray:
    if not samples:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack([sample.numpy() for sample in samples]).astype(np.float32)


def _stack_step_tensors(samples) -> np.ndarray:
    if not samples:
        return np.empty(0, dtype=np.float32)
    return np.stack([sample.numpy() for sample in samples]).astype(np.float32)


def _tensor_to_numpy(values) -> np.ndarray:
    if values.numel() == 0:
        return np.empty(0, dtype=np.float32)
    return values.detach().cpu().numpy().astype(np.float32)


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or "benchmark"


def _metric_names_for_plots() -> list[str]:
    return [
        "lin_vel_rmse",
        "yaw_rate_rmse",
        "fall_rate",
        "mean_episode_length_s",
        "lin_vel_error_mean",
        "yaw_rate_error_mean",
        "touchdown_vz_mean",
        "touchdown_vz_p95",
        "touchdown_fz_mean",
        "touchdown_fz_p95",
        "contact_fz_mean",
        "contact_fz_p95",
        "foot_fz_all_mean",
        "foot_fz_all_p95",
    ]


def _penalty_scale_at_checkpoint(resolved_run: ResolvedRun) -> float:
    if not resolved_run.spec.uses_penalty_scale_curriculum:
        return 1.0
    if not resolved_run.penalty_scale_history:
        return float("nan")

    points_with_step = [
        point
        for point in resolved_run.penalty_scale_history
        if point.global_step is not None and point.global_step <= resolved_run.checkpoint_step
    ]
    if points_with_step:
        return max(points_with_step, key=lambda point: point.global_step or 0).penalty_scale
    return resolved_run.penalty_scale_history[-1].penalty_scale


def _effective_reward_weight_at_checkpoint(resolved_run: ResolvedRun) -> float:
    penalty_scale = _penalty_scale_at_checkpoint(resolved_run)
    if math.isnan(penalty_scale):
        return float("nan")
    return resolved_run.spec.reward_weight * penalty_scale


def _empty_if_none(value: Any) -> Any:
    return "" if value is None else value


def _safe_mean(values) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.float().mean().item())


def _safe_quantile(values, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.float().quantile(q).item())


if __name__ == "__main__":
    main()
