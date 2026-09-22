"""Node 3: Programmatic validation of candidate proposals (deduplication & cycle bounds)."""

from __future__ import annotations

import json
import logging
import sqlite3
from typing import Any, Mapping

from chia.base.ChiaFunction import ChiaFunction

logger = logging.getLogger("agentic_ml_analog_sim.node3")


def _get_val(obj: Any, key: str, default: Any = None) -> Any:
    """Extract attribute or dictionary key safely."""
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_dict(val: Any) -> dict[str, Any]:
    """Normalize dictionary, JSON string, or None into a clean comparable dict."""
    if val is None:
        return {}
    if isinstance(val, dict):
        return val
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, dict):
                return parsed
            return {"value": parsed}
        except Exception:
            return {"raw": val}
    return {"value": val}


def _get_proposal_record(db: Any, candidate_id: str) -> dict[str, Any] | None:
    """Retrieve proposal record by candidate_id from DB across diverse implementations."""
    # 1. Specialized get_proposal method on ResultDatabase
    if hasattr(db, "get_proposal") and callable(db.get_proposal):
        try:
            prop = db.get_proposal(candidate_id)
            if prop is not None:
                if isinstance(prop, Mapping):
                    return dict(prop)
                return {
                    "candidate_id": _get_val(prop, "candidate_id"),
                    "solver": _get_val(prop, "solver"),
                    "noise_model": _get_val(prop, "noise_model"),
                    "sparsity_config": _get_val(prop, "sparsity_config"),
                    "solver_params": _get_val(prop, "solver_params"),
                    "noise_params": _get_val(prop, "noise_params"),
                    "status": _get_val(prop, "status"),
                }
        except Exception as exc:
            logger.debug("db.get_proposal(%s) raised: %s", candidate_id, exc)

    # 2. Direct SQLite query
    conn = getattr(db, "conn", None) or getattr(db, "connection", None) or getattr(db, "_conn", None)
    if conn is not None and isinstance(conn, sqlite3.Connection):
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT candidate_id, solver, noise_model, sparsity_config,
                       solver_params, noise_params, status, is_reference
                FROM proposals
                WHERE candidate_id = ?
                """,
                (candidate_id,),
            )
            row = cursor.fetchone()
            if row is not None:
                return dict(row)
        except Exception as exc:
            logger.debug("Direct SQLite query for proposal %s raised: %s", candidate_id, exc)

    # 3. Handle dictionary or list mocks
    if isinstance(db, dict):
        if candidate_id in db:
            return db[candidate_id]
        for v in db.values():
            if isinstance(v, Mapping) and v.get("candidate_id") == candidate_id:
                return v
    elif isinstance(db, list):
        for item in db:
            if _get_val(item, "candidate_id") == candidate_id:
                return item

    return None


def _get_all_proposals(db: Any) -> list[dict[str, Any]]:
    """Retrieve all proposal records from DB for duplicate checking."""
    if hasattr(db, "get_all_proposals") and callable(db.get_all_proposals):
        try:
            res = db.get_all_proposals()
            if res:
                return [dict(p) if isinstance(p, Mapping) else p for p in res]
        except Exception as exc:
            logger.debug("db.get_all_proposals() in _get_all_proposals raised: %s", exc)

    if hasattr(db, "query_proposals") and callable(db.query_proposals):
        try:
            res = db.query_proposals()
            if res:
                return [dict(p) if isinstance(p, Mapping) else {
                    "candidate_id": _get_val(p, "candidate_id"),
                    "solver": _get_val(p, "solver"),
                    "noise_model": _get_val(p, "noise_model"),
                    "sparsity_config": _get_val(p, "sparsity_config"),
                    "solver_params": _get_val(p, "solver_params"),
                    "noise_params": _get_val(p, "noise_params"),
                    "status": _get_val(p, "status"),
                } for p in res]
        except Exception as exc:
            logger.debug("db.query_proposals() in _get_all_proposals raised: %s", exc)

    conn = getattr(db, "conn", None) or getattr(db, "connection", None) or getattr(db, "_conn", None)
    if conn is not None and isinstance(conn, sqlite3.Connection):
        try:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT candidate_id, solver, noise_model, sparsity_config,
                       solver_params, noise_params, status, is_reference
                FROM proposals
                """
            )
            return [dict(r) for r in cursor.fetchall()]
        except Exception as exc:
            logger.debug("Direct SQLite fetchall raised: %s", exc)

    if isinstance(db, dict):
        return [v for v in db.values() if isinstance(v, (dict, object))]
    elif isinstance(db, list):
        return [item for item in db if isinstance(item, (dict, object))]

    return []


def _update_proposal_status(db: Any, candidate_id: str, new_status: str) -> None:
    """Update status of a proposal in DB."""
    if hasattr(db, "update_proposal_status") and callable(db.update_proposal_status):
        try:
            db.update_proposal_status(candidate_id, new_status)
            return
        except Exception as exc:
            logger.debug("db.update_proposal_status() raised: %s", exc)

    conn = getattr(db, "conn", None) or getattr(db, "connection", None) or getattr(db, "_conn", None)
    if conn is not None and isinstance(conn, sqlite3.Connection):
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE proposals SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE candidate_id = ?",
                (new_status, candidate_id),
            )
            conn.commit()
            return
        except Exception as exc:
            logger.debug("Direct SQLite update proposals raised: %s", exc)

    if isinstance(db, dict) and candidate_id in db:
        if isinstance(db[candidate_id], dict):
            db[candidate_id]["status"] = new_status
        else:
            setattr(db[candidate_id], "status", new_status)


def _add_comment(db: Any, candidate_id: str, phase: str, agent_name: str, comment: str) -> None:
    """Add comment record to DB."""
    if hasattr(db, "add_comment") and callable(db.add_comment):
        try:
            db.add_comment(candidate_id=candidate_id, phase=phase, comment=comment, agent_name=agent_name)
            return
        except TypeError:
            try:
                db.add_comment(candidate_id=candidate_id, phase=phase, comment=comment)
                return
            except Exception:
                pass
        except Exception as exc:
            logger.debug("db.add_comment() raised: %s", exc)

    conn = getattr(db, "conn", None) or getattr(db, "connection", None) or getattr(db, "_conn", None)
    if conn is not None and isinstance(conn, sqlite3.Connection):
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO comments (candidate_id, phase, agent_name, comment)
                VALUES (?, ?, ?, ?)
                """,
                (candidate_id, phase, agent_name, comment),
            )
            conn.commit()
            return
        except Exception as exc:
            logger.debug("Direct SQLite insert into comments raised: %s", exc)


def _get_max_retries(config: Any, default: int = 3) -> int:
    """Extract max_node_retries from budget configuration."""
    budget = _get_val(config, "budget")
    if budget is not None:
        retries = _get_val(budget, "max_node_retries")
        if retries is not None:
            return int(retries)
    return default


def _validate_solver(solver: str, solver_params: dict[str, Any], config: Any) -> tuple[bool, str | None]:
    """Verify solver name and parameters against config.solvers."""
    configured_solvers = _get_val(config, "solvers", [])
    allowed_solver_names: set[str] = set()
    solver_spec_map: dict[str, Any] = {}

    if isinstance(configured_solvers, dict):
        for name, spec in configured_solvers.items():
            allowed_solver_names.add(name)
            solver_spec_map[name] = spec
    else:
        for s in configured_solvers:
            name = _get_val(s, "name") if not isinstance(s, str) else s
            if name:
                allowed_solver_names.add(name)
                solver_spec_map[name] = s

    if solver not in allowed_solver_names:
        return (
            False,
            f"INVALID_SOLVER: Solver '{solver}' is not allowed. Allowed solvers: {sorted(allowed_solver_names)}",
        )

    # Check allowed_steps or steps if specified
    spec = solver_spec_map.get(solver)
    if spec is not None:
        allowed_steps = _get_val(spec, "allowed_steps") or _get_val(spec, "steps")
        if allowed_steps is not None and isinstance(allowed_steps, (list, tuple)):
            num_steps = solver_params.get("num_steps")
            if num_steps is None:
                num_steps = solver_params.get("steps")
            if num_steps is not None and num_steps not in allowed_steps:
                return (
                    False,
                    f"INVALID_SOLVER_STEPS: num_steps={num_steps} is invalid for solver '{solver}'. You MUST choose from allowed step counts: {allowed_steps}.",
                )

    return (True, None)


def _validate_noise_model(noise_model: str, noise_params: dict[str, Any], config: Any) -> tuple[bool, str | None]:
    """Verify noise model and parameters against config.noise_models."""
    # "none" is always allowed for baseline / noise-free evaluations
    allowed_noise_names: set[str] = {"none"}
    noise_spec_map: dict[str, Any] = {}

    configured_noise = _get_val(config, "noise_models", [])
    if isinstance(configured_noise, dict):
        for name, spec in configured_noise.items():
            allowed_noise_names.add(name)
            noise_spec_map[name] = spec
    else:
        for n in configured_noise:
            name = _get_val(n, "name") if not isinstance(n, str) else n
            if name:
                allowed_noise_names.add(name)
                noise_spec_map[name] = n

    if noise_model not in allowed_noise_names:
        return (
            False,
            f"INVALID_NOISE_MODEL: Noise model '{noise_model}' is not configured. Allowed noise models are: {sorted(allowed_noise_names)}. Select from this list.",
        )

    # Check sigma_range if specified
    spec = noise_spec_map.get(noise_model)
    if spec is not None:
        sigma_range = _get_val(spec, "sigma_range")
        if sigma_range is not None and len(sigma_range) == 2:
            sigma = noise_params.get("sigma")
            if sigma is not None:
                min_s, max_s = sigma_range[0], sigma_range[1]
                if not (min_s <= sigma <= max_s):
                    return (
                        False,
                        f"INVALID_NOISE_SIGMA: sigma={sigma} is out of bounds {sigma_range} for noise model '{noise_model}'. Choose sigma within [{min_s}, {max_s}].",
                    )

    return (True, None)


def _validate_sparsity_config(sparsity_config: dict[str, Any], config: Any) -> tuple[bool, str | None]:
    """Verify sparsity configuration against config.sparsity_configs."""
    configured_sparsity = _get_val(config, "sparsity_configs")
    if configured_sparsity is None:
        # Check if topology is set to modular (Revision 3 style)
        topology = _get_val(config, "topology")
        if topology is not None:
            return (True, None)
        return (True, None)

    allowed_ratios: set[float] = set()
    allowed_names: set[str] = set()

    # Support SparsityConfigs Pydantic model (with dense, threshold_pruned, degree_bounded attributes)
    if hasattr(configured_sparsity, "threshold_pruned") or hasattr(configured_sparsity, "dense"):
        dense_obj = getattr(configured_sparsity, "dense", None)
        if dense_obj is not None:
            allowed_names.add("dense")
            allowed_ratios.add(float(getattr(dense_obj, "sparsity_ratio", 0.0)))

        tp_obj = getattr(configured_sparsity, "threshold_pruned", None)
        if tp_obj is not None:
            allowed_names.add("threshold_pruned")
            for r in getattr(tp_obj, "allowed_sparsity_ratios", []):
                allowed_ratios.add(float(r))

        db_obj = getattr(configured_sparsity, "degree_bounded", None)
        if db_obj is not None:
            allowed_names.add("degree_bounded")
    elif isinstance(configured_sparsity, dict):
        if "threshold_pruned" in configured_sparsity or "dense" in configured_sparsity:
            if "dense" in configured_sparsity:
                allowed_names.add("dense")
                allowed_ratios.add(float(_get_val(configured_sparsity["dense"], "sparsity_ratio", 0.0)))
            if "threshold_pruned" in configured_sparsity:
                allowed_names.add("threshold_pruned")
                for r in _get_val(configured_sparsity["threshold_pruned"], "allowed_sparsity_ratios", []):
                    allowed_ratios.add(float(r))
            if "degree_bounded" in configured_sparsity:
                allowed_names.add("degree_bounded")
        else:
            for name, sc in configured_sparsity.items():
                allowed_names.add(name)
                ratio = _get_val(sc, "sparsity_ratio")
                if ratio is not None:
                    allowed_ratios.add(float(ratio))
                ratios = _get_val(sc, "allowed_sparsity_ratios")
                if ratios:
                    for r in ratios:
                        allowed_ratios.add(float(r))
    elif isinstance(configured_sparsity, (list, tuple)):
        for sc in configured_sparsity:
            name = _get_val(sc, "name") if not isinstance(sc, str) else sc
            if name:
                allowed_names.add(name)
            ratio = _get_val(sc, "sparsity_ratio")
            if ratio is not None:
                allowed_ratios.add(float(ratio))
            ratios = _get_val(sc, "allowed_sparsity_ratios")
            if ratios:
                for r in ratios:
                    allowed_ratios.add(float(r))

    prop_ratio = sparsity_config.get("sparsity_ratio")
    prop_name = sparsity_config.get("name") or sparsity_config.get("type")

    # If proposal specifies a sparsity ratio, check against allowed ratios
    if prop_ratio is not None and allowed_ratios:
        # Use floating point tolerance for ratio comparison
        if not any(abs(float(prop_ratio) - r) < 1e-5 for r in allowed_ratios):
            return (
                False,
                f"INVALID_SPARSITY_RATIO: sparsity_ratio={prop_ratio} is invalid. You MUST select one of the allowed sparsity ratios from configuration: {sorted(allowed_ratios)} (allowed sparsity types: {sorted(allowed_names)}).",
            )

    if prop_name and allowed_names:
        clean_name = str(prop_name).lower().replace("sparsityconfig", "").replace("config", "").strip()
        matched = any(
            clean_name == name.lower() or name.lower() in clean_name or clean_name in name.lower()
            for name in allowed_names
        )
        if not matched and clean_name not in ("dense", "threshold_pruned", "degree_bounded"):
            return (
                False,
                f"INVALID_SPARSITY_TYPE: Sparsity type '{prop_name}' is not allowed. You MUST select one of the allowed sparsity types from configuration: {sorted(allowed_names)} with ratios: {sorted(allowed_ratios)}.",
            )

    return (True, None)


def _canonicalize_sparsity_config(sc: Any) -> dict[str, Any]:
    """Canonicalize sparsity configuration into a standardized semantic dictionary."""
    d = _normalize_dict(sc)
    s_type = str(d.get("type", d.get("method", ""))).lower()
    max_deg = d.get("max_degree")
    ratio = d.get("sparsity_ratio")

    # 1. Degree-bounded coupling
    if max_deg is not None or "degree" in s_type:
        return {
            "type": "degree_bounded",
            "max_degree": int(max_deg) if max_deg is not None else 64,
            "sparsity_ratio": 0.0,
        }

    # 2. Dense coupling
    ratio_val = float(ratio) if ratio is not None else 0.0
    if abs(ratio_val) < 1e-6 or "dense" in s_type:
        return {"type": "dense", "sparsity_ratio": 0.0}

    # 3. Threshold pruned
    return {"type": "threshold_pruned", "sparsity_ratio": round(ratio_val, 4)}


def _canonicalize_solver_params(solver: str, params: Any) -> dict[str, Any]:
    """Canonicalize solver parameters into standardized steps and dt."""
    d = _normalize_dict(params)
    num_steps = d.get("num_steps") or d.get("steps") or 10
    num_steps = int(num_steps)
    dt = d.get("dt")
    if dt is None:
        t = float(d.get("integration_time", 1.0))
        dt = round(t / num_steps, 6)
    else:
        dt = round(float(dt), 6)
    return {"num_steps": num_steps, "dt": dt}


def _canonicalize_noise_params(noise_model: str, params: Any) -> dict[str, Any]:
    """Canonicalize noise model parameters."""
    if str(noise_model).strip().lower() in ("none", "null", ""):
        return {}
    d = _normalize_dict(params)
    sigma = d.get("sigma", 0.01)
    return {"sigma": round(float(sigma), 6)}


def _check_duplicate(candidate_id: str, proposal: dict[str, Any], db: Any) -> tuple[bool, str | None]:
    """Check if an identical proposal was previously evaluated or pending in DB using semantic equivalence."""
    all_proposals = _get_all_proposals(db)

    solver = str(proposal.get("solver", "")).strip().lower()
    noise_model = str(proposal.get("noise_model", "")).strip().lower()
    solver_params = _canonicalize_solver_params(solver, proposal.get("solver_params"))
    noise_params = _canonicalize_noise_params(noise_model, proposal.get("noise_params"))
    sparsity_config = _canonicalize_sparsity_config(proposal.get("sparsity_config"))

    for p in all_proposals:
        p_id = _get_val(p, "candidate_id")
        p_status = str(_get_val(p, "status", "")).upper()

        # Skip self and previously rejected proposals
        if p_id == candidate_id or p_status == "REJECTED":
            continue

        p_solver = str(_get_val(p, "solver", "")).strip().lower()
        p_noise = str(_get_val(p, "noise_model", "")).strip().lower()
        p_spar = _canonicalize_sparsity_config(_get_val(p, "sparsity_config"))
        p_solv_params = _canonicalize_solver_params(p_solver, _get_val(p, "solver_params"))
        p_noise_params = _canonicalize_noise_params(p_noise, _get_val(p, "noise_params"))

        if (
            p_solver == solver
            and p_noise == noise_model
            and p_spar == sparsity_config
            and p_solv_params == solver_params
            and p_noise_params == noise_params
        ):
            p_solv = _get_val(p, "solver") or p_solver
            p_noise = _get_val(p, "noise_model") or p_noise
            p_steps = p_solv_params.get("num_steps")
            spar_type = p_spar.get("type", "dense")
            if spar_type == "threshold_pruned":
                spar_desc = f"threshold_pruned(ratio={p_spar.get('sparsity_ratio', 0.0)})"
            elif spar_type == "degree_bounded":
                spar_desc = f"degree_bounded(max_degree={p_spar.get('max_degree', 64)})"
            elif spar_type == "dense":
                spar_desc = "dense"
            else:
                spar_desc = str(p_spar)

            return (
                True,
                f"DUPLICATE_PROPOSAL: A candidate with solver '{p_solv}', steps={p_steps}, noise='{p_noise}', and sparsity '{spar_desc}' already exists in the database ({p_id}). You MUST propose a distinct parameter combination.",
            )

    return (False, None)


@ChiaFunction(resources={"control_worker": 1})
def node3_phase0_validate_proposal(
    candidate_id: str,
    db: Any,
    config: Any,
    cycle_count: int,
    target_noise_model: Optional[str] = None,
) -> tuple[bool, str, int]:
    """Programmatically validate the latest candidate proposal in DB.

    Performs the following verification checks:
      0. Verifies proposal matches `target_noise_model` if specified.
      1. Verifies that `solver` is in `config.solvers`.
      2. Verifies that `noise_model` is in `config.noise_models`.
      3. Verifies that `sparsity_config` is allowed by `config.sparsity_configs`.
      4. Checks for duplicate proposals against prior evaluations in DB.
      5. Enforces cycle bound:
         - If duplicate or invalid and `cycle_count >= config.budget.max_node_retries`:
           Marks proposal as REJECTED, adds termination comment, returns (False, "MAX_CYCLES_EXCEEDED", cycle_count).
         - If duplicate or invalid and `cycle_count < config.budget.max_node_retries`:
           Marks proposal as REJECTED, adds rejection comment, returns (False, reason, cycle_count + 1).
      6. If valid and unique:
         Returns (True, "VALID", cycle_count).

    Args:
        candidate_id: Unique candidate identifier in DB.
        db: ResultDatabase instance (or compatible database/mock).
        config: SimulationConfig instance (or compatible configuration object).
        cycle_count: Current retry cycle counter (starting at 0 or 1).
        target_noise_model: Optional target noise model to enforce.

    Returns:
        tuple[bool, str, int]: (is_valid, reason, updated_cycle_count)
    """
    proposal = _get_proposal_record(db, candidate_id)
    max_retries = _get_max_retries(config, default=3)

    if proposal is None:
        reason = f"PROPOSAL_NOT_FOUND: Proposal '{candidate_id}' not found in database"
        logger.error("Node 3: %s", reason)
        return (False, reason, cycle_count + 1 if cycle_count < max_retries else cycle_count)

    solver = str(proposal.get("solver", ""))
    noise_model = str(proposal.get("noise_model", ""))
    solver_params = _normalize_dict(proposal.get("solver_params"))
    noise_params = _normalize_dict(proposal.get("noise_params"))
    sparsity_config = _normalize_dict(proposal.get("sparsity_config"))

    # 0. Verify target noise model constraint if specified
    if target_noise_model is not None and proposal.get("noise_model") != target_noise_model:
        return _handle_rejection(
            candidate_id,
            db,
            "REJECTED_WRONG_NOISE_MODEL",
            cycle_count,
            max_retries,
        )

    # 1. Verify solver
    valid_solver, solver_err = _validate_solver(solver, solver_params, config)
    if not valid_solver:
        return _handle_rejection(candidate_id, db, solver_err or "INVALID_SOLVER", cycle_count, max_retries)

    # 2. Verify noise model
    valid_noise, noise_err = _validate_noise_model(noise_model, noise_params, config)
    if not valid_noise:
        return _handle_rejection(candidate_id, db, noise_err or "INVALID_NOISE_MODEL", cycle_count, max_retries)

    # 3. Verify sparsity configuration
    valid_spar, spar_err = _validate_sparsity_config(sparsity_config, config)
    if not valid_spar:
        return _handle_rejection(candidate_id, db, spar_err or "INVALID_SPARSITY_CONFIG", cycle_count, max_retries)

    # 4. Check for duplicate proposals
    is_duplicate, dup_err = _check_duplicate(candidate_id, proposal, db)
    if is_duplicate:
        return _handle_rejection(candidate_id, db, dup_err or "DUPLICATE_PROPOSAL", cycle_count, max_retries)

    # 6. Proposal is valid and unique!
    logger.info("Node 3: Proposal %s is valid and unique (cycle=%d)", candidate_id, cycle_count)
    return (True, "VALID", cycle_count)


def _handle_rejection(
    candidate_id: str,
    db: Any,
    reason: str,
    cycle_count: int,
    max_retries: int,
) -> tuple[bool, str, int]:
    """Handle proposal rejection according to cycle count and max retries."""
    _update_proposal_status(db, candidate_id, "REJECTED")

    reason_str = reason.rstrip(".") + "."

    if cycle_count >= max_retries:
        comment = (
            f"Proposal rejected: {reason_str} "
            f"Terminated execution because cycle_count ({cycle_count}) reached max_node_retries ({max_retries})."
        )
        _add_comment(db, candidate_id, phase="phase_0", agent_name="node3_validate_proposal", comment=comment)
        logger.warning("Node 3: %s", comment)
        return (False, "MAX_CYCLES_EXCEEDED", cycle_count)
    else:
        next_cycle = cycle_count + 1
        comment = f"Proposal rejected: {reason_str} Retry cycle {next_cycle}/{max_retries}."
        _add_comment(db, candidate_id, phase="phase_0", agent_name="node3_validate_proposal", comment=comment)
        logger.info("Node 3: %s", comment)
        return (False, reason, next_cycle)


# Backward compatibility aliases
validate_proposal = node3_phase0_validate_proposal
node3_validate_proposal = node3_phase0_validate_proposal

__all__ = [
    "node3_phase0_validate_proposal",
    "validate_proposal",
    "node3_validate_proposal",
    "_canonicalize_sparsity_config",
    "_canonicalize_solver_params",
    "_canonicalize_noise_params",
    "_check_duplicate",
]

