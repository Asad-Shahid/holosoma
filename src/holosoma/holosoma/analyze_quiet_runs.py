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
    randomization_report: dict[str, Any]
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
    checkpoint_name_overrides: dict[str, str] | None = None,
) -> list[ResolvedRun]:
    import wandb

    checkpoint_name_overrides = checkpoint_name_overrides or {}
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
        if spec.run_name in checkpoint_name_overrides:
            checkpoint_name = checkpoint_name_overrides[spec.run_name]
            checkpoint_step = _checkpoint_step_from_name(checkpoint_name)
        else:
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
    num_steps: int | None,
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

            resolved_num_steps = num_steps
            if resolved_num_steps is None:
                resolved_num_steps = int(round(episode_length_s / float(runtime.env.dt)))

            algo = _load_algo_for_runtime(
                runtime=runtime,
                checkpoint_uri=resolved_run.checkpoint_uri,
                log_dir=per_run_dir,
            )

            for scenario in scenarios:
                benchmark_result = _evaluate_fast_sac_checkpoint(
                    algo=algo,
                    scenario=scenario,
                    num_steps=resolved_num_steps,
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
                    "physics_randomization_configured": runtime.randomization_report[
                        "physics_randomization_configured"
                    ],
                    "physics_randomization_detected_across_envs": runtime.randomization_report[
                        "physics_randomization_detected_across_envs"
                    ],
                    "detected_physics_variation_fields": runtime.randomization_report[
                        "detected_physics_variation_fields"
                    ],
                    "physics_randomization_terms": runtime.randomization_report["physics_randomization_terms"],
                    "randomization_setup_terms": runtime.randomization_report["setup_terms"],
                    "randomization_reset_terms": runtime.randomization_report["reset_terms"],
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
        "--checkpoint-name",
        action="append",
        default=None,
        help=(
            "Checkpoint file to evaluate. Use 'model_<step>.pt' when evaluating exactly one run, "
            "or '<run_name>=model_<step>.pt' when evaluating multiple runs. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Evaluation command scenario in the format '<label>:vx,vy,yaw'. Can be passed multiple times.",
    )
    parser.add_argument("--num-envs", type=int, default=64, help="Number of parallel environments for evaluation.")
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help=(
            "Number of policy/control steps per scenario. Defaults to episode-length-s / policy_dt."
        ),
    )
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
    if args.num_steps is not None and args.num_steps <= 0:
        parser.error("--num-steps must be greater than zero when provided.")

    scenarios = parse_command_scenarios(args.scenario)
    specs = parse_quiet_reward_commands_file(args.manifest)
    include_run_names = set(args.include_run_name) if args.include_run_name else None
    checkpoint_name_overrides = parse_checkpoint_name_overrides(
        args.checkpoint_name,
        specs=specs,
        include_run_names=include_run_names,
    )
    resolved_runs = resolve_wandb_runs(
        entity=args.wandb_entity,
        project=args.wandb_project,
        specs=specs,
        include_run_names=include_run_names,
        checkpoint_name_overrides=checkpoint_name_overrides,
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


def parse_checkpoint_name_overrides(
    values: list[str] | None,
    *,
    specs: list[RunSpec],
    include_run_names: set[str] | None,
) -> dict[str, str]:
    if not values:
        return {}

    requested_specs = [
        spec for spec in specs if include_run_names is None or spec.run_name in include_run_names
    ]
    requested_names = {spec.run_name for spec in requested_specs}
    overrides: dict[str, str] = {}

    for raw_value in values:
        if "=" in raw_value:
            run_name, checkpoint_name = raw_value.split("=", 1)
            run_name = run_name.strip()
            checkpoint_name = checkpoint_name.strip()
        else:
            if len(requested_names) != 1:
                raise ValueError(
                    "Bare --checkpoint-name can only be used when exactly one run is selected. "
                    "Use '<run_name>=model_<step>.pt' for multiple runs."
                )
            run_name = next(iter(requested_names))
            checkpoint_name = raw_value.strip()

        if run_name not in requested_names:
            raise ValueError(f"--checkpoint-name references a run that is not selected: {run_name}")
        _checkpoint_step_from_name(checkpoint_name)
        overrides[run_name] = checkpoint_name

    return overrides


def _checkpoint_step_from_name(checkpoint_name: str) -> int:
    match = re.match(r"model_(\d+)\.pt$", Path(checkpoint_name).name)
    if not match:
        raise ValueError(
            f"Invalid checkpoint name '{checkpoint_name}'. Expected a file named 'model_<step>.pt'."
        )
    return int(match.group(1))


def _select_latest_checkpoint(run: Any) -> tuple[str, int]:
    checkpoint_candidates: list[tuple[int, str]] = []

    try:
        run_files = run.files()
    except Exception as exc:
        raise RuntimeError(
            f"Could not list checkpoint files for W&B run '{run.id}'. "
            "Pass --checkpoint-name model_<step>.pt to evaluate a known checkpoint directly."
        ) from exc

    try:
        file_iterator = iter(run_files)
    except TypeError as exc:
        raise RuntimeError(
            f"Could not list checkpoint files for W&B run '{run.id}'. "
            "Pass --checkpoint-name model_<step>.pt to evaluate a known checkpoint directly."
        ) from exc

    while True:
        try:
            file_obj = next(file_iterator)
        except StopIteration:
            break
        except Exception as exc:
            raise RuntimeError(
                f"Could not list checkpoint files for W&B run '{run.id}'. "
                "Pass --checkpoint-name model_<step>.pt to evaluate a known checkpoint directly."
            ) from exc
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
    randomization_report = _physics_randomization_report(env)
    logger = _get_logger()
    logger.info(
        "Physics randomization configured=%s, detected_across_envs=%s, terms=%s",
        randomization_report["physics_randomization_configured"],
        randomization_report["physics_randomization_detected_across_envs"],
        randomization_report["physics_randomization_terms"],
    )
    return AnalysisRuntime(
        analysis_cfg=analysis_cfg,
        env=env,
        device=device,
        simulation_app=simulation_app,
        randomization_report=randomization_report,
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


def _build_analysis_config(
    saved_cfg,
    *,
    num_envs: int,
    episode_length_s: float,
    headless: bool,
):
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
    _reset_locomotion_to_default_pose(env=env, torch=torch)
    obs = _apply_fixed_command_and_rebuild_actor_obs(
        env=env,
        wrapped_env=wrapped_env,
        command_tensor=command_tensor,
        torch=torch,
    )

    touchdown_vz_samples: list[torch.Tensor] = []
    touchdown_fz_samples: list[torch.Tensor] = []
    touchdown_vz_left_samples: list[torch.Tensor] = []
    touchdown_vz_right_samples: list[torch.Tensor] = []
    touchdown_fz_left_samples: list[torch.Tensor] = []
    touchdown_fz_right_samples: list[torch.Tensor] = []
    contact_fz_samples: list[torch.Tensor] = []
    foot_fz_all_samples: list[torch.Tensor] = []
    touchdown_vz_trace_env: list[torch.Tensor] = []
    touchdown_fz_trace_env: list[torch.Tensor] = []
    lin_vel_error_trace_env: list[torch.Tensor] = []
    yaw_rate_error_trace_env: list[torch.Tensor] = []
    gravity_xy_trace_env: list[torch.Tensor] = []
    raw_tracking_lin_trace_env: list[torch.Tensor] = []
    raw_tracking_ang_trace_env: list[torch.Tensor] = []
    foot_fz_mean_trace_env: list[torch.Tensor] = []

    prev_contact = env.simulator.contact_forces[:, env.feet_indices, 2] > contact_threshold
    prev_foot_vz = env.simulator._rigid_body_vel[:, env.feet_indices, 2].clone()
    tracking_lin_sigma = _reward_term_param(env, "tracking_lin_vel", "tracking_sigma", 0.25)
    tracking_ang_sigma = _reward_term_param(env, "tracking_ang_vel", "tracking_sigma", 0.25)

    lin_vel_sq_error_sum_env = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    lin_vel_error_sum_env = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    yaw_rate_sq_error_sum_env = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    yaw_rate_error_sum_env = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    gravity_xy_norm_sum_env = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    raw_tracking_lin_sum_env = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    raw_tracking_ang_sum_env = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    touchdown_vz_sum_env = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    touchdown_fz_sum_env = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    touchdown_count_env = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
    recording_step_count = 0
    fall_event_count = 0
    timeout_event_count = 0
    done_event_count = 0
    fell_env_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    terminal_recorded_mask = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    terminal_step_counts = torch.full(
        (env.num_envs,),
        int(num_steps),
        device=env.device,
        dtype=torch.float32,
    )
    if env.feet_indices.numel() < 2:
        raise ValueError(f"Expected at least two feet for left/right analysis, got {env.feet_indices.numel()}.")

    with torch.inference_mode():
        for step_idx in range(num_steps):
            _apply_fixed_command(env=env, command_tensor=command_tensor)
            actions = policy({"actor_obs": obs})
            env._pre_physics_step(actions)
            env.render()

            for _ in range(env.simulator.simulator_config.sim.control_decimation):
                env._apply_force_in_physics_step()
                env.simulator.simulate_at_each_physics_step()
                env.simulator.refresh_sim_tensors()
                env._pre_compute_observations_callback()
                _apply_fixed_command(env=env, command_tensor=command_tensor)

                commands = env.command_manager.commands
                lin_vel = get_base_lin_vel(env)[:, :2]
                yaw_rate = get_base_ang_vel(env)[:, 2]
                gravity_xy_norm = torch.linalg.norm(get_projected_gravity(env)[:, :2], dim=1)
                lin_vel_error = torch.linalg.norm(commands[:, :2] - lin_vel, dim=1)
                yaw_rate_error = torch.abs(commands[:, 2] - yaw_rate)
                lin_vel_sq_error = torch.sum(torch.square(commands[:, :2] - lin_vel), dim=1)
                yaw_rate_sq_error = torch.square(commands[:, 2] - yaw_rate)
                raw_tracking_lin = torch.exp(-lin_vel_sq_error / tracking_lin_sigma)
                raw_tracking_ang = torch.exp(-yaw_rate_sq_error / tracking_ang_sigma)
                lin_vel_sq_error_sum_env += lin_vel_sq_error
                lin_vel_error_sum_env += lin_vel_error
                yaw_rate_sq_error_sum_env += yaw_rate_sq_error
                yaw_rate_error_sum_env += yaw_rate_error
                gravity_xy_norm_sum_env += gravity_xy_norm
                raw_tracking_lin_sum_env += raw_tracking_lin
                raw_tracking_ang_sum_env += raw_tracking_ang
                lin_vel_error_trace_env.append(lin_vel_error.detach().cpu())
                yaw_rate_error_trace_env.append(yaw_rate_error.detach().cpu())
                gravity_xy_trace_env.append(gravity_xy_norm.detach().cpu())
                raw_tracking_lin_trace_env.append(raw_tracking_lin.detach().cpu())
                raw_tracking_ang_trace_env.append(raw_tracking_ang.detach().cpu())
                recording_step_count += 1

                contact_fz = torch.clamp(env.simulator.contact_forces[:, env.feet_indices, 2], min=0.0)
                foot_fz_all_samples.append(contact_fz.reshape(-1).detach().cpu())
                foot_fz_mean_trace_env.append(contact_fz.mean(dim=1).detach().cpu())
                contact_now = contact_fz > contact_threshold
                touchdown_now = contact_now & ~prev_contact
                downward_speed = torch.clamp(-prev_foot_vz, min=0.0)
                touchdown_vz_by_env = torch.full(
                    (env.num_envs,), float("nan"), device=env.device, dtype=torch.float32
                )
                touchdown_fz_by_env = torch.full(
                    (env.num_envs,), float("nan"), device=env.device, dtype=torch.float32
                )

                if touchdown_now.any():
                    touchdown_env_ids, _touchdown_foot_ids = touchdown_now.nonzero(as_tuple=True)
                    touchdown_vz_values = downward_speed[touchdown_env_ids, _touchdown_foot_ids]
                    touchdown_fz_values = contact_fz[touchdown_env_ids, _touchdown_foot_ids]
                    touchdown_vz_step = touchdown_vz_values.detach().cpu()
                    touchdown_fz_step = touchdown_fz_values.detach().cpu()
                    touchdown_vz_samples.append(touchdown_vz_step)
                    touchdown_fz_samples.append(touchdown_fz_step)
                    ones = torch.ones_like(touchdown_vz_values, dtype=torch.float32)
                    touchdown_vz_sum_env.scatter_add_(0, touchdown_env_ids, touchdown_vz_values)
                    touchdown_fz_sum_env.scatter_add_(0, touchdown_env_ids, touchdown_fz_values)
                    touchdown_count_env.scatter_add_(0, touchdown_env_ids, ones)
                    touchdown_vz_step_sum = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
                    touchdown_fz_step_sum = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
                    touchdown_step_count = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
                    touchdown_vz_step_sum.scatter_add_(0, touchdown_env_ids, touchdown_vz_values)
                    touchdown_fz_step_sum.scatter_add_(0, touchdown_env_ids, touchdown_fz_values)
                    touchdown_step_count.scatter_add_(0, touchdown_env_ids, ones)
                    touchdown_has_sample = touchdown_step_count > 0
                    touchdown_vz_by_env[touchdown_has_sample] = (
                        touchdown_vz_step_sum[touchdown_has_sample] / touchdown_step_count[touchdown_has_sample]
                    )
                    touchdown_fz_by_env[touchdown_has_sample] = (
                        touchdown_fz_step_sum[touchdown_has_sample] / touchdown_step_count[touchdown_has_sample]
                    )
                touchdown_vz_trace_env.append(touchdown_vz_by_env.detach().cpu())
                touchdown_fz_trace_env.append(touchdown_fz_by_env.detach().cpu())
                touchdown_left = touchdown_now[:, 0]
                touchdown_right = touchdown_now[:, 1]
                if touchdown_left.any():
                    touchdown_vz_left_step = downward_speed[:, 0][touchdown_left].detach().cpu()
                    touchdown_fz_left_step = contact_fz[:, 0][touchdown_left].detach().cpu()
                    touchdown_vz_left_samples.append(touchdown_vz_left_step)
                    touchdown_fz_left_samples.append(touchdown_fz_left_step)
                if touchdown_right.any():
                    touchdown_vz_right_step = downward_speed[:, 1][touchdown_right].detach().cpu()
                    touchdown_fz_right_step = contact_fz[:, 1][touchdown_right].detach().cpu()
                    touchdown_vz_right_samples.append(touchdown_vz_right_step)
                    touchdown_fz_right_samples.append(touchdown_fz_right_step)
                if contact_now.any():
                    contact_fz_samples.append(contact_fz[contact_now].detach().cpu())

                prev_contact = contact_now.clone()
                prev_foot_vz = env.simulator._rigid_body_vel[:, env.feet_indices, 2].clone()

            env._post_physics_step()
            _apply_fixed_command(env=env, command_tensor=command_tensor)
            obs = _actor_obs_from_env_obs_dict(env=env, wrapped_env=wrapped_env, torch=torch)
            dones = env.reset_buf
            extras = env.extras

            done_mask = dones.bool()
            if done_mask.any():
                timeout_mask = extras["time_outs"].bool()
                fall_mask = done_mask & ~timeout_mask
                fall_events = int(fall_mask.sum().item())
                timeout_events = int((done_mask & timeout_mask).sum().item())
                done_events = int(done_mask.sum().item())
                fall_event_count += fall_events
                timeout_event_count += timeout_events
                done_event_count += done_events
                fell_env_mask |= fall_mask

                completed_lengths = getattr(env, "_pending_episode_lengths", None)
                if completed_lengths is None:
                    done_step_counts = torch.full_like(terminal_step_counts, float(step_idx + 1))
                else:
                    done_step_counts = completed_lengths.to(device=env.device, dtype=torch.float32)
                    fallback_step_counts = torch.full_like(done_step_counts, float(step_idx + 1))
                    done_step_counts = torch.where(done_step_counts > 0.0, done_step_counts, fallback_step_counts)
                newly_terminal = done_mask & ~terminal_recorded_mask
                terminal_step_counts[newly_terminal] = done_step_counts[newly_terminal]
                terminal_recorded_mask |= newly_terminal

            if done_mask.any():
                reset_env_ids = done_mask.nonzero(as_tuple=False).flatten()
                _reset_locomotion_to_default_pose(env=env, torch=torch, env_ids=reset_env_ids)
                fixed_obs = _rebuild_actor_obs_without_history_update(env=env, wrapped_env=wrapped_env, torch=torch)
                obs[done_mask] = fixed_obs[done_mask]
                prev_contact[done_mask] = (
                    env.simulator.contact_forces[done_mask][:, env.feet_indices, 2] > contact_threshold
                )
                prev_foot_vz[done_mask] = env.simulator._rigid_body_vel[done_mask][:, env.feet_indices, 2]

    env_sample_count = max(recording_step_count, 1)
    per_env_lin_vel_rmse = torch.sqrt(lin_vel_sq_error_sum_env / env_sample_count)
    per_env_lin_vel_error_mean = lin_vel_error_sum_env / env_sample_count
    per_env_yaw_rate_rmse = torch.sqrt(yaw_rate_sq_error_sum_env / env_sample_count)
    per_env_yaw_rate_error_mean = yaw_rate_error_sum_env / env_sample_count
    per_env_gravity_xy_mean = gravity_xy_norm_sum_env / env_sample_count
    per_env_raw_tracking_lin_mean = raw_tracking_lin_sum_env / env_sample_count
    per_env_raw_tracking_ang_mean = raw_tracking_ang_sum_env / env_sample_count
    per_env_touchdown_vz_mean = torch.full(
        (env.num_envs,), float("nan"), device=env.device, dtype=torch.float32
    )
    per_env_touchdown_fz_mean = torch.full(
        (env.num_envs,), float("nan"), device=env.device, dtype=torch.float32
    )
    touchdown_env_mask = touchdown_count_env > 0
    per_env_touchdown_vz_mean[touchdown_env_mask] = (
        touchdown_vz_sum_env[touchdown_env_mask] / touchdown_count_env[touchdown_env_mask]
    )
    per_env_touchdown_fz_mean[touchdown_env_mask] = (
        touchdown_fz_sum_env[touchdown_env_mask] / touchdown_count_env[touchdown_env_mask]
    )

    lin_vel_rmse = _safe_mean(per_env_lin_vel_rmse)
    lin_vel_error_mean = _safe_mean(per_env_lin_vel_error_mean)
    yaw_rate_rmse = _safe_mean(per_env_yaw_rate_rmse)
    yaw_rate_error_mean = _safe_mean(per_env_yaw_rate_error_mean)
    gravity_xy_mean = _safe_mean(per_env_gravity_xy_mean)
    raw_tracking_lin_mean = _safe_mean(per_env_raw_tracking_lin_mean)
    raw_tracking_ang_mean = _safe_mean(per_env_raw_tracking_ang_mean)
    episode_count = env.num_envs
    fall_count = int(fell_env_mask.sum().item())
    timeout_count = max(int(env.num_envs) - fall_count, 0)
    mean_episode_length_s = float(terminal_step_counts.mean().item()) * float(env.dt)

    touchdown_vz_tensor = _concat_samples(touchdown_vz_samples)
    touchdown_fz_tensor = _concat_samples(touchdown_fz_samples)
    touchdown_vz_left_tensor = _concat_samples(touchdown_vz_left_samples)
    touchdown_vz_right_tensor = _concat_samples(touchdown_vz_right_samples)
    touchdown_fz_left_tensor = _concat_samples(touchdown_fz_left_samples)
    touchdown_fz_right_tensor = _concat_samples(touchdown_fz_right_samples)
    contact_fz_tensor = _concat_samples(contact_fz_samples)
    foot_fz_all_tensor = _concat_samples(foot_fz_all_samples)
    touchdown_vz_trace_env_np = _stack_step_tensors(touchdown_vz_trace_env)
    touchdown_fz_trace_env_np = _stack_step_tensors(touchdown_fz_trace_env)
    lin_vel_error_trace_env_np = _stack_step_tensors(lin_vel_error_trace_env)
    yaw_rate_error_trace_env_np = _stack_step_tensors(yaw_rate_error_trace_env)
    gravity_xy_trace_env_np = _stack_step_tensors(gravity_xy_trace_env)
    raw_tracking_lin_trace_env_np = _stack_step_tensors(raw_tracking_lin_trace_env)
    raw_tracking_ang_trace_env_np = _stack_step_tensors(raw_tracking_ang_trace_env)
    foot_fz_mean_trace_env_np = _stack_step_tensors(foot_fz_mean_trace_env)
    metrics = {
        "num_envs": env.num_envs,
        "num_steps": num_steps,
        "num_recording_steps": recording_step_count,
        "lin_vel_rmse": lin_vel_rmse,
        "lin_vel_rmse_var": _safe_var(per_env_lin_vel_rmse),
        "lin_vel_rmse_std": _safe_std(per_env_lin_vel_rmse),
        "lin_vel_error_mean": lin_vel_error_mean,
        "lin_vel_error_mean_var": _safe_var(per_env_lin_vel_error_mean),
        "lin_vel_error_mean_std": _safe_std(per_env_lin_vel_error_mean),
        "yaw_rate_rmse": yaw_rate_rmse,
        "yaw_rate_rmse_var": _safe_var(per_env_yaw_rate_rmse),
        "yaw_rate_rmse_std": _safe_std(per_env_yaw_rate_rmse),
        "yaw_rate_error_mean": yaw_rate_error_mean,
        "yaw_rate_error_mean_var": _safe_var(per_env_yaw_rate_error_mean),
        "yaw_rate_error_mean_std": _safe_std(per_env_yaw_rate_error_mean),
        "gravity_xy_mean": gravity_xy_mean,
        "gravity_xy_mean_var": _safe_var(per_env_gravity_xy_mean),
        "gravity_xy_mean_std": _safe_std(per_env_gravity_xy_mean),
        "episode_count": episode_count,
        "fall_count": fall_count,
        "timeout_count": timeout_count,
        "fall_event_count": fall_event_count,
        "timeout_event_count": timeout_event_count,
        "done_event_count": done_event_count,
        "fall_rate": (fall_count / episode_count) if episode_count else float("nan"),
        "mean_episode_length_s": mean_episode_length_s,
        "touchdown_count": int(touchdown_vz_tensor.numel()),
        "touchdown_left_count": int(touchdown_vz_left_tensor.numel()),
        "touchdown_right_count": int(touchdown_vz_right_tensor.numel()),
        "touchdown_vz_mean": _safe_mean(touchdown_vz_tensor),
        "touchdown_vz_mean_across_envs": _safe_nanmean(per_env_touchdown_vz_mean),
        "touchdown_vz_mean_var_across_envs": _safe_nanvar(per_env_touchdown_vz_mean),
        "touchdown_vz_mean_std_across_envs": _safe_nanstd(per_env_touchdown_vz_mean),
        "touchdown_vz_p95": _safe_quantile(touchdown_vz_tensor, 0.95),
        "touchdown_vz_left_mean": _safe_mean(touchdown_vz_left_tensor),
        "touchdown_vz_left_p95": _safe_quantile(touchdown_vz_left_tensor, 0.95),
        "touchdown_vz_right_mean": _safe_mean(touchdown_vz_right_tensor),
        "touchdown_vz_right_p95": _safe_quantile(touchdown_vz_right_tensor, 0.95),
        "touchdown_fz_mean": _safe_mean(touchdown_fz_tensor),
        "touchdown_fz_mean_across_envs": _safe_nanmean(per_env_touchdown_fz_mean),
        "touchdown_fz_mean_var_across_envs": _safe_nanvar(per_env_touchdown_fz_mean),
        "touchdown_fz_mean_std_across_envs": _safe_nanstd(per_env_touchdown_fz_mean),
        "touchdown_fz_p95": _safe_quantile(touchdown_fz_tensor, 0.95),
        "touchdown_fz_left_mean": _safe_mean(touchdown_fz_left_tensor),
        "touchdown_fz_left_p95": _safe_quantile(touchdown_fz_left_tensor, 0.95),
        "touchdown_fz_right_mean": _safe_mean(touchdown_fz_right_tensor),
        "touchdown_fz_right_p95": _safe_quantile(touchdown_fz_right_tensor, 0.95),
        "contact_fz_mean": _safe_mean(contact_fz_tensor),
        "contact_fz_p95": _safe_quantile(contact_fz_tensor, 0.95),
        "foot_fz_all_mean": _safe_mean(foot_fz_all_tensor),
        "foot_fz_all_p95": _safe_quantile(foot_fz_all_tensor, 0.95),
        "raw_tracking_lin_mean": raw_tracking_lin_mean,
        "raw_tracking_lin_mean_var": _safe_var(per_env_raw_tracking_lin_mean),
        "raw_tracking_lin_mean_std": _safe_std(per_env_raw_tracking_lin_mean),
        "raw_tracking_ang_mean": raw_tracking_ang_mean,
        "raw_tracking_ang_mean_var": _safe_var(per_env_raw_tracking_ang_mean),
        "raw_tracking_ang_mean_std": _safe_std(per_env_raw_tracking_ang_mean),
        "analysis_dt_s": float(env.sim_dt),
        "recording_dt_s": float(env.sim_dt),
        "policy_dt_s": float(env.dt),
        "recording_frequency_hz": float(1.0 / float(env.sim_dt)),
        "policy_frequency_hz": float(1.0 / float(env.dt)),
        "rollout_duration_s": float(num_steps) * float(env.dt),
    }
    time_s = (np.arange(recording_step_count, dtype=np.float32) + np.float32(1.0)) * np.float32(env.sim_dt)
    arrays = {
        "command": np.asarray(scenario.command, dtype=np.float32),
        "time_s": time_s,
        "touchdown_vz_trace": _nanmean_over_env(touchdown_vz_trace_env_np),
        "touchdown_vz_trace_std": _nanstd_over_env(touchdown_vz_trace_env_np),
        "touchdown_vz_trace_env": touchdown_vz_trace_env_np,
        "touchdown_fz_trace": _nanmean_over_env(touchdown_fz_trace_env_np),
        "touchdown_fz_trace_std": _nanstd_over_env(touchdown_fz_trace_env_np),
        "touchdown_fz_trace_env": touchdown_fz_trace_env_np,
        "lin_vel_error_trace": _nanmean_over_env(lin_vel_error_trace_env_np),
        "lin_vel_error_trace_std": _nanstd_over_env(lin_vel_error_trace_env_np),
        "lin_vel_error_trace_env": lin_vel_error_trace_env_np,
        "yaw_rate_error_trace": _nanmean_over_env(yaw_rate_error_trace_env_np),
        "yaw_rate_error_trace_std": _nanstd_over_env(yaw_rate_error_trace_env_np),
        "yaw_rate_error_trace_env": yaw_rate_error_trace_env_np,
        "gravity_xy_trace": _nanmean_over_env(gravity_xy_trace_env_np),
        "gravity_xy_trace_std": _nanstd_over_env(gravity_xy_trace_env_np),
        "gravity_xy_trace_env": gravity_xy_trace_env_np,
        "raw_tracking_lin_trace": _nanmean_over_env(raw_tracking_lin_trace_env_np),
        "raw_tracking_lin_trace_std": _nanstd_over_env(raw_tracking_lin_trace_env_np),
        "raw_tracking_lin_trace_env": raw_tracking_lin_trace_env_np,
        "raw_tracking_ang_trace": _nanmean_over_env(raw_tracking_ang_trace_env_np),
        "raw_tracking_ang_trace_std": _nanstd_over_env(raw_tracking_ang_trace_env_np),
        "raw_tracking_ang_trace_env": raw_tracking_ang_trace_env_np,
        "foot_fz_mean_trace": _nanmean_over_env(foot_fz_mean_trace_env_np),
        "foot_fz_mean_trace_std": _nanstd_over_env(foot_fz_mean_trace_env_np),
        "foot_fz_mean_trace_env": foot_fz_mean_trace_env_np,
        "per_env_lin_vel_rmse": _tensor_to_numpy(per_env_lin_vel_rmse),
        "per_env_lin_vel_error_mean": _tensor_to_numpy(per_env_lin_vel_error_mean),
        "per_env_yaw_rate_rmse": _tensor_to_numpy(per_env_yaw_rate_rmse),
        "per_env_yaw_rate_error_mean": _tensor_to_numpy(per_env_yaw_rate_error_mean),
        "per_env_gravity_xy_mean": _tensor_to_numpy(per_env_gravity_xy_mean),
        "per_env_raw_tracking_lin_mean": _tensor_to_numpy(per_env_raw_tracking_lin_mean),
        "per_env_raw_tracking_ang_mean": _tensor_to_numpy(per_env_raw_tracking_ang_mean),
        "per_env_touchdown_count": _tensor_to_numpy(touchdown_count_env),
        "per_env_touchdown_vz_mean": _tensor_to_numpy(per_env_touchdown_vz_mean),
        "per_env_touchdown_fz_mean": _tensor_to_numpy(per_env_touchdown_fz_mean),
        "foot_fz_all": _tensor_to_numpy(foot_fz_all_tensor),
        "touchdown_vz": _tensor_to_numpy(touchdown_vz_tensor),
        "touchdown_fz": _tensor_to_numpy(touchdown_fz_tensor),
        "touchdown_vz_left": _tensor_to_numpy(touchdown_vz_left_tensor),
        "touchdown_vz_right": _tensor_to_numpy(touchdown_vz_right_tensor),
        "touchdown_fz_left": _tensor_to_numpy(touchdown_fz_left_tensor),
        "touchdown_fz_right": _tensor_to_numpy(touchdown_fz_right_tensor),
        "contact_fz": _tensor_to_numpy(contact_fz_tensor),
    }
    return BenchmarkResult(metrics=metrics, arrays=arrays)


def _reset_locomotion_to_default_pose(*, env, torch, env_ids=None) -> None:
    """Reset locomotion envs to the exact configured default pose before analysis starts."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return
    if not hasattr(env, "default_dof_pos"):
        raise TypeError("Default-pose analysis is currently implemented for locomotion environments only.")

    if hasattr(env, "default_dof_pos_base"):
        default_dof_pos = env.default_dof_pos_base.view(1, -1).expand(env.num_envs, -1)
    else:
        default_dof_pos = env.default_dof_pos
    env.simulator.dof_pos[env_ids] = default_dof_pos[env_ids]
    env.simulator.dof_vel[env_ids] = 0.0

    env.simulator.robot_root_states[env_ids] = env.base_init_state
    env.simulator.robot_root_states[env_ids, :3] += env.terrain_manager.get_state(
        "locomotion_terrain"
    ).env_origins[env_ids]
    env.simulator.robot_root_states[env_ids, 7:13] = 0.0

    env.episode_length_buf[env_ids] = 0
    env.reset_buf[env_ids] = 0
    env.time_out_buf[env_ids] = False

    env.simulator.set_actor_root_state_tensor_robots(env_ids, env.simulator.robot_root_states)
    env.simulator.set_dof_state_tensor_robots(env_ids, env.simulator.dof_state)
    env.simulator.clear_contact_forces_history(env_ids)
    env.simulator.refresh_sim_tensors()
    env._pre_compute_observations_callback()


def _apply_fixed_command_and_rebuild_actor_obs(*, env, wrapped_env, command_tensor, torch):
    _apply_fixed_command(env=env, command_tensor=command_tensor)
    return _rebuild_actor_obs_without_history_update(env=env, wrapped_env=wrapped_env, torch=torch)


def _actor_obs_from_env_obs_dict(*, env, wrapped_env, torch):
    clip_limit = env.observation_manager.cfg.clip_observations
    actor_obs_keys = getattr(wrapped_env, "_actor_obs_keys")
    return torch.cat(
        [torch.clip(env.obs_buf_dict[key], -clip_limit, clip_limit) for key in actor_obs_keys],
        dim=1,
    )


def _apply_fixed_command(*, env, command_tensor) -> None:
    env.command_manager.commands[:] = command_tensor.view(1, -1).expand_as(env.command_manager.commands)


def _reward_term_param(env, term_name: str, param_name: str, default: float) -> float:
    reward_manager = getattr(env, "reward_manager", None)
    if reward_manager is None or term_name not in reward_manager.active_terms:
        return float(default)
    term_cfg = reward_manager.get_term_cfg(term_name)
    params = getattr(term_cfg, "params", None) or {}
    return float(params.get(param_name, default))


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
        "analysis_dt_s": benchmark_result.metrics.get("analysis_dt_s"),
        "recording_dt_s": benchmark_result.metrics.get("recording_dt_s"),
        "policy_dt_s": benchmark_result.metrics.get("policy_dt_s"),
        "recording_frequency_hz": benchmark_result.metrics.get("recording_frequency_hz"),
        "policy_frequency_hz": benchmark_result.metrics.get("policy_frequency_hz"),
        "rollout_duration_s": benchmark_result.metrics.get("rollout_duration_s"),
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
    _plot_left_right_metric_vs_weight(results, plots_dir, plt)
    _plot_time_series_comparisons(results, output_dir, plots_dir, plt)


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


def _physics_randomization_report(env) -> dict[str, Any]:
    randomization_manager = getattr(env, "randomization_manager", None)
    cfg = getattr(randomization_manager, "cfg", None)
    setup_terms_cfg = getattr(cfg, "setup_terms", {}) or {}
    reset_terms_cfg = getattr(cfg, "reset_terms", {}) or {}
    setup_terms = list(setup_terms_cfg.keys())
    reset_terms = list(reset_terms_cfg.keys())
    physics_term_names = {
        "mass_randomizer",
        "randomize_friction_startup",
        "randomize_base_com_startup",
    }
    configured_terms = [
        term_name
        for term_name, term_cfg in setup_terms_cfg.items()
        if term_name in physics_term_names and _randomization_term_enabled(term_name, term_cfg)
    ]
    detected_fields = _detect_physics_variation_fields(env)
    return {
        "physics_randomization_configured": bool(configured_terms),
        "physics_randomization_detected_across_envs": bool(detected_fields),
        "physics_randomization_terms": ",".join(configured_terms),
        "detected_physics_variation_fields": ",".join(detected_fields),
        "setup_terms": ",".join(setup_terms),
        "reset_terms": ",".join(reset_terms),
    }


def _randomization_term_enabled(term_name: str, term_cfg: Any) -> bool:
    params = getattr(term_cfg, "params", None) or {}
    if not bool(params.get("enabled", True)):
        return False
    if term_name == "mass_randomizer":
        return bool(params.get("enable_link_mass", True) or params.get("enable_base_mass", True))
    return True


def _detect_physics_variation_fields(env) -> list[str]:
    if int(getattr(env, "num_envs", 1)) <= 1:
        return []

    simulator = getattr(env, "simulator", None)
    detected: list[str] = []
    if simulator is None:
        return detected

    if hasattr(simulator, "gym"):
        detected.extend(_detect_isaacgym_physics_variation(env, simulator))

    detected.extend(_detect_tensor_physics_variation(simulator))
    if _tensor_has_cross_env_variation(getattr(simulator, "_base_com_bias", None)):
        detected.append("base_com_bias")
    if _tensor_has_cross_env_variation(getattr(simulator, "base_com_bias", None)):
        detected.append("base_com_bias")

    return sorted(set(detected))


def _detect_isaacgym_physics_variation(env, simulator) -> list[str]:
    detected: list[str] = []
    gym = simulator.gym
    envs = getattr(simulator, "envs", [])
    robot_handles = getattr(simulator, "robot_handles", [])
    if not envs or not robot_handles:
        return detected

    mass_rows: list[list[float]] = []
    com_rows: list[list[float]] = []
    friction_rows: list[list[float]] = []
    body_names = list(getattr(simulator, "_body_list", []) or [])
    randomized_body_names = list(getattr(env.robot_config, "randomize_link_body_names", None) or [])
    torso_name = getattr(env.robot_config, "torso_name", None)
    if torso_name:
        randomized_body_names.append(torso_name)
    body_indices = [body_names.index(name) for name in randomized_body_names if name in body_names]

    for env_ptr, actor in zip(envs, robot_handles):
        if body_indices:
            body_props = gym.get_actor_rigid_body_properties(env_ptr, actor)
            mass_rows.append([float(body_props[index].mass) for index in body_indices])
            com_row: list[float] = []
            for index in body_indices:
                com = body_props[index].com
                com_row.extend([float(com.x), float(com.y), float(com.z)])
            com_rows.append(com_row)

        shape_props = gym.get_actor_rigid_shape_properties(env_ptr, actor)
        if shape_props:
            friction_rows.append([float(prop.friction) for prop in shape_props])

    if _rows_have_cross_env_variation(mass_rows):
        detected.append("body_mass")
    if _rows_have_cross_env_variation(com_rows):
        detected.append("body_com")
    if _rows_have_cross_env_variation(friction_rows):
        detected.append("geom_friction")
    return detected


def _detect_tensor_physics_variation(simulator) -> list[str]:
    detected: list[str] = []
    bridge = getattr(getattr(simulator, "backend", None), "warp_model_bridge", None)
    if bridge is None:
        return detected
    for field_name in ("body_mass", "geom_friction", "body_ipos"):
        if _tensor_has_cross_env_variation(getattr(bridge, field_name, None)):
            detected.append(field_name)
    return detected


def _rows_have_cross_env_variation(rows: list[list[float]], *, atol: float = 1e-7) -> bool:
    if len(rows) <= 1:
        return False
    try:
        values = np.asarray(rows, dtype=np.float64)
    except ValueError:
        return False
    if values.ndim != 2 or values.shape[0] <= 1 or values.size == 0:
        return False
    deltas = np.abs(values - values[0:1])
    return bool(np.isfinite(deltas).any() and np.nanmax(deltas) > atol)


def _tensor_has_cross_env_variation(values: Any, *, atol: float = 1e-7) -> bool:
    if values is None:
        return False
    try:
        if hasattr(values, "detach"):
            array = values.detach().cpu().numpy()
        elif hasattr(values, "numpy"):
            array = values.numpy()
        else:
            array = np.asarray(values)
    except Exception:
        return False
    if array.ndim == 0 or array.shape[0] <= 1 or array.size == 0:
        return False
    deltas = np.abs(array - array[0:1])
    return bool(np.isfinite(deltas).any() and np.nanmax(deltas) > atol)


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


def _plot_left_right_metric_vs_weight(results: list[dict[str, Any]], plots_dir: Path, plt) -> None:
    scenario_names = sorted({row["scenario"] for row in results})
    category_colors = {
        "baseline": "tab:green",
        "gated": "tab:blue",
        "not_gated": "tab:orange",
        "olaf": "tab:red",
    }
    metric_specs = [
        (
            "touchdown_vz_p95",
            "touchdown_vz_left_p95",
            "touchdown_vz_right_p95",
            "Touchdown Vertical Speed P95",
            "m/s",
        ),
        (
            "touchdown_fz_p95",
            "touchdown_fz_left_p95",
            "touchdown_fz_right_p95",
            "Touchdown Vertical Force P95",
            "N",
        ),
    ]

    for metric_base, left_key, right_key, title, ylabel in metric_specs:
        fig, axes = plt.subplots(
            nrows=len(scenario_names),
            ncols=1,
            figsize=(10, max(4, 3.5 * len(scenario_names))),
            squeeze=False,
        )
        plotted_any = False

        for axis, scenario_name in zip(axes[:, 0], scenario_names):
            scenario_rows = [row for row in results if row["scenario"] == scenario_name]
            categories = sorted({str(row["category"]) for row in scenario_rows})
            for category in categories:
                category_rows = sorted(
                    (row for row in scenario_rows if str(row["category"]) == category),
                    key=_plot_weight_value,
                )
                if not category_rows:
                    continue
                weights = [_plot_weight_value(row) for row in category_rows]
                left_values = [float(row.get(left_key, np.nan)) for row in category_rows]
                right_values = [float(row.get(right_key, np.nan)) for row in category_rows]
                color = category_colors.get(category, None)
                axis.plot(weights, left_values, marker="o", linestyle="-", color=color, label=f"{category} left")
                axis.plot(weights, right_values, marker="s", linestyle="--", color=color, label=f"{category} right")
                plotted_any = True

            axis.set_title(f"{title} | {scenario_name}")
            axis.set_xlabel("Effective Reward Weight Magnitude")
            axis.set_ylabel(ylabel)
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=7)

        if plotted_any:
            fig.tight_layout()
            fig.savefig(plots_dir / f"{metric_base}_left_right_by_weight.png", dpi=200)
        plt.close(fig)


def _plot_time_series_comparisons(
    results: list[dict[str, Any]],
    output_dir: Path,
    plots_dir: Path,
    plt,
) -> None:
    scenario_names = sorted({row["scenario"] for row in results})
    trace_specs = [
        ("lin_vel_error_trace", "Linear Velocity Error", "m/s"),
        ("yaw_rate_error_trace", "Yaw Rate Error", "rad/s"),
        ("gravity_xy_trace", "Projected Gravity XY Norm", ""),
        ("raw_tracking_lin_trace", "Raw Linear Tracking Reward", ""),
        ("raw_tracking_ang_trace", "Raw Angular Tracking Reward", ""),
        ("foot_fz_mean_trace", "Mean Vertical Foot Force", "N"),
        ("touchdown_vz_trace", "Touchdown Vertical Speed", "m/s"),
        ("touchdown_fz_trace", "Touchdown Vertical Force", "N"),
    ]

    for scenario_name in scenario_names:
        scenario_rows = sorted(
            (row for row in results if row["scenario"] == scenario_name),
            key=lambda row: (
                str(row["category"]),
                _plot_weight_value(row),
                str(row["run_name"]),
                int(row["checkpoint_step"]),
            ),
        )
        scenario_stem = _slugify(scenario_name)

        for trace_name, title, ylabel in trace_specs:
            fig, ax = plt.subplots(figsize=(12, 6))
            plotted_any = False

            for row in scenario_rows:
                data_path = output_dir / str(row["run_name"]) / f"{scenario_stem}_benchmark_data.npz"
                if not data_path.exists():
                    continue
                with np.load(data_path, allow_pickle=False) as data:
                    if "time_s" not in data or trace_name not in data:
                        continue
                    time_s = data["time_s"]
                    values = data[trace_name]
                    std_values = data[f"{trace_name}_std"] if f"{trace_name}_std" in data else None

                valid = np.isfinite(values)
                if not valid.any():
                    continue

                label = f"{row['run_name']} | {row['checkpoint_name']}"
                ax.plot(time_s[valid], values[valid], marker="o", markersize=2, linewidth=0.8, label=label)
                if std_values is not None:
                    std_valid = valid & np.isfinite(std_values)
                    if std_valid.any():
                        ax.fill_between(
                            time_s[std_valid],
                            values[std_valid] - std_values[std_valid],
                            values[std_valid] + std_values[std_valid],
                            alpha=0.12,
                        )
                plotted_any = True

            if not plotted_any:
                plt.close(fig)
                continue

            ax.set_title(f"{title} | {scenario_name}")
            ax.set_xlabel("time [s]")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(plots_dir / f"{scenario_stem}_{trace_name}_timeseries.png", dpi=200)
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


def _stack_step_tensors(samples) -> np.ndarray:
    if not samples:
        return np.empty((0, 0), dtype=np.float32)
    return np.stack([sample.numpy() for sample in samples]).astype(np.float32)


def _nanmean_over_env(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    sums = np.where(finite, values, 0.0).sum(axis=1)
    output = np.full(values.shape[0], np.nan, dtype=np.float32)
    valid = counts > 0
    output[valid] = (sums[valid] / counts[valid]).astype(np.float32)
    return output


def _nanstd_over_env(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return np.empty(0, dtype=np.float32)
    means = _nanmean_over_env(values)
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    centered = np.where(finite, values - means[:, None], 0.0)
    variances = np.full(values.shape[0], np.nan, dtype=np.float32)
    valid = counts > 0
    variances[valid] = (np.square(centered[valid]).sum(axis=1) / counts[valid]).astype(np.float32)
    return np.sqrt(variances).astype(np.float32)


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
        "lin_vel_rmse_var",
        "lin_vel_rmse_std",
        "yaw_rate_rmse",
        "yaw_rate_rmse_var",
        "yaw_rate_rmse_std",
        "fall_rate",
        "mean_episode_length_s",
        "lin_vel_error_mean",
        "lin_vel_error_mean_var",
        "lin_vel_error_mean_std",
        "yaw_rate_error_mean",
        "yaw_rate_error_mean_var",
        "yaw_rate_error_mean_std",
        "gravity_xy_mean",
        "gravity_xy_mean_var",
        "gravity_xy_mean_std",
        "touchdown_vz_mean",
        "touchdown_vz_mean_across_envs",
        "touchdown_vz_mean_var_across_envs",
        "touchdown_vz_mean_std_across_envs",
        "touchdown_vz_p95",
        "touchdown_vz_left_mean",
        "touchdown_vz_left_p95",
        "touchdown_vz_right_mean",
        "touchdown_vz_right_p95",
        "touchdown_fz_mean",
        "touchdown_fz_mean_across_envs",
        "touchdown_fz_mean_var_across_envs",
        "touchdown_fz_mean_std_across_envs",
        "touchdown_fz_p95",
        "touchdown_fz_left_mean",
        "touchdown_fz_left_p95",
        "touchdown_fz_right_mean",
        "touchdown_fz_right_p95",
        "contact_fz_mean",
        "contact_fz_p95",
        "foot_fz_all_mean",
        "foot_fz_all_p95",
        "raw_tracking_lin_mean",
        "raw_tracking_lin_mean_var",
        "raw_tracking_lin_mean_std",
        "raw_tracking_ang_mean",
        "raw_tracking_ang_mean_var",
        "raw_tracking_ang_mean_std",
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


def _safe_std(values) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.float().std(unbiased=False).item())


def _safe_var(values) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.float().var(unbiased=False).item())


def _safe_nanmean(values) -> float:
    finite_values = values[values.isfinite()]
    if finite_values.numel() == 0:
        return float("nan")
    return float(finite_values.float().mean().item())


def _safe_nanstd(values) -> float:
    finite_values = values[values.isfinite()]
    if finite_values.numel() == 0:
        return float("nan")
    return float(finite_values.float().std(unbiased=False).item())


def _safe_nanvar(values) -> float:
    finite_values = values[values.isfinite()]
    if finite_values.numel() == 0:
        return float("nan")
    return float(finite_values.float().var(unbiased=False).item())


def _safe_quantile(values, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(values.float().quantile(q).item())


if __name__ == "__main__":
    main()
