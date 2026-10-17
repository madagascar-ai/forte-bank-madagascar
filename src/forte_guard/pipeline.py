from __future__ import annotations

import base64
import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .contracts import AnalysisResult, Evidence, Frame, ObjectJournal, Violation, ZoneProfile
from .journal.builder import build_journal
from .journal.zones import CORE_ZONES, estimate_zones
from .reasoning.catalog import load_catalog
from .reasoning.llm import auto_zones, llm_candidates
from .reasoning.rules import rule_candidates
from .reasoning.validation import merge_candidates, validate_candidates
from .settings import Settings
from .storage import write_json
from .video import Deadline, sample_video
from .vision.detector import ReplayDetector, YoloTracker


def _evidence(candidate, frame_map: dict[str, Frame]) -> list[Evidence]:
    frames = sorted(
        (frame_map[key] for key in set(candidate.evidence_frame_ids)), key=lambda f: f.timestamp_sec
    )
    anchor = min(frames, key=lambda frame: abs(frame.timestamp_sec - candidate.timecode_sec))
    # Preserve the anchor plus before/after context, maximum three JPEG frames.
    selected = {anchor.frame_id: anchor}
    for frame in (frames[0], frames[-1]):
        selected[frame.frame_id] = frame
    return [
        Evidence(
            timestamp_sec=frame.timestamp_sec,
            image_base64=base64.b64encode(Path(frame.image_path).read_bytes()).decode("ascii"),
        )
        for frame in sorted(selected.values(), key=lambda f: f.timestamp_sec)
    ]


def _busiest_frame(frames: list[Frame], objects: ObjectJournal) -> Frame:
    # The vision fallback sees the frame with the most people, where the layout is clearest.
    counts = {f.frame_id: sum(d.label in {"person", "cashier", "client"} for d in f.detections) for f in objects.frames}
    return max(frames, key=lambda frame: counts.get(frame.frame_id, 0))


def run_pipeline(
    source: Path,
    output: Path,
    settings: Settings,
    job_id: str,
    progress: Callable[[str, float], None] = lambda *_: None,
    zones: ZoneProfile | None = None,
    replay: Path | None = None,
    remaining_seconds: float | None = None,
) -> AnalysisResult:
    settings.validate()
    deadline = Deadline(remaining_seconds if remaining_seconds is not None else settings.job_timeout_sec)
    timings, warnings = {}, []
    output.mkdir(parents=True, exist_ok=True)
    # Snapshot inputs for reproducibility. Credentials are deliberately excluded.
    rules, scenarios = load_catalog(settings.config_dir)
    write_json(
        output / "catalog_snapshot.json", {"rules": [r.model_dump() for r in rules], "scenarios": scenarios}
    )
    write_json(
        output / "run_config.json",
        {
            "schema_version": "1.0",
            "sample_fps": settings.sample_fps,
            "detector": "replay" if replay else settings.detector,
            "model_path": settings.model_path.name,
            "llm_model": settings.llm_model,
            "zone_model": settings.zone_model,
            "llm_enabled": settings.llm_enabled,
            "cashier_presence": settings.cashier_presence,
            "verification_mode": "structural_only",
            "labels_used": False,
        },
    )
    started = time.monotonic()
    progress("frames", 0.03)
    frames, metadata = sample_video(
        source, output / "frames", settings.sample_fps, deadline, settings.max_video_sec
    )
    write_json(output / "frames.json", [frame.model_dump() for frame in frames])
    write_json(output / "metadata.json", metadata)
    timings["frames"] = time.monotonic() - started
    if metadata["timestamp_fallback_used"]:
        warnings.append("Часть таймкодов восстановлена по номеру исходного кадра и FPS.")
    if zones is None:
        raw_profile = json.loads((settings.config_dir / "zones.json").read_text())
        zones = ZoneProfile.model_validate({key: raw_profile[key] for key in ("source", "zones")})
    progress("objects", 0.12)
    started = time.monotonic()
    detector = ReplayDetector(replay) if replay else YoloTracker(settings)
    objects = detector.run(frames, deadline, lambda value: progress("objects", 0.12 + 0.48 * value))
    write_json(output / "object_journal.json", objects)
    timings["objects"] = time.monotonic() - started
    if not zones.zones and settings.auto_zones:
        progress("zones", 0.6)
        started = time.monotonic()
        # Zones come from where people stay and where items rest; nobody has to draw them.
        zones, notes = estimate_zones(objects)
        warnings.extend(notes)
        missing = CORE_ZONES - {zone.name for zone in zones.zones}
        if missing and settings.llm_enabled and replay is None:
            try:
                guess = auto_zones(settings, _busiest_frame(frames, objects), deadline)
                known = {zone.name for zone in zones.zones}
                extra = [zone for zone in guess.zones if zone.name not in known]
                if extra:
                    source = "auto_tracks+vision" if zones.zones else guess.source
                    zones = ZoneProfile(source=source, zones=[*zones.zones, *extra])
            except Exception as exc:
                # Do not print model content or frames. Surface a clear degraded-mode warning.
                warnings.append(f"Уточнение зон vision-моделью недоступно ({type(exc).__name__}).")
        timings["zones"] = time.monotonic() - started
    write_json(output / "zones.json", zones)
    deadline.remaining()
    progress("actions", 0.64)
    started = time.monotonic()
    journal = build_journal(
        objects, zones, metadata["duration_sec"], settings.sample_fps, settings.cashier_presence
    )
    write_json(output / "action_journal.json", journal)
    warnings.extend(journal.warnings)
    timings["actions"] = time.monotonic() - started
    progress("candidates", 0.73)
    started = time.monotonic()
    # Both providers receive exactly the same immutable-by-convention journal snapshot.
    with ThreadPoolExecutor(max_workers=2) as pool:
        code_future = pool.submit(rule_candidates, journal, rules, scenarios)
        llm_future = pool.submit(llm_candidates, journal, rules, scenarios, settings, deadline)
        from_rules = code_future.result()
        try:
            from_llm, llm_warnings = llm_future.result()
            warnings.extend(llm_warnings)
        except Exception as exc:
            from_llm = []
            warnings.append(f"Ветка LLM недоступна ({type(exc).__name__}); показаны кандидаты правил в коде.")
    write_json(output / "candidates.rules.json", [c.model_dump() for c in from_rules])
    write_json(output / "candidates.llm.json", [c.model_dump() for c in from_llm])
    timings["candidates"] = time.monotonic() - started
    deadline.remaining()
    progress("validation", 0.9)
    combined = merge_candidates([*from_rules, *from_llm])
    accepted, validation_warnings = validate_candidates(combined, journal, frames, rules)
    warnings.extend(validation_warnings)
    warnings.append("Верификация MVP проверяет ссылки и таймкоды; повторная визуальная проверка отключена.")
    write_json(output / "candidates.json", [c.model_dump() for c in combined])
    frame_map = {frame.frame_id: frame for frame in frames}
    rule_map = {rule.id: rule for rule in rules}
    violations = []
    for candidate in accepted:
        rule = rule_map[candidate.rule_id]
        document = rule.document
        observed = (
            rule.observation.replace("{t}", f"{candidate.timecode_sec:.1f}")
            if rule.observation
            else candidate.reasoning
        )
        description = (
            observed + f" Основание для проверки: {document.get('title', '')}; "
            f"{document.get('clause', '')}."
        )
        if rule.source_status != "verified":
            description += " Нормативная привязка предварительная."
        violations.append(
            Violation(
                type=rule.title,
                timecode_sec=candidate.timecode_sec,
                description=description,
                evidence_frames=_evidence(candidate, frame_map),
                rule_id=rule.id,
                sources=candidate.sources,
                document=document,
            )
        )
    result = AnalysisResult(
        job_id=job_id,
        verdict="violation" if violations else "no_violation",
        violations=violations,
        warnings=list(dict.fromkeys(warnings)),
        timings_sec={key: round(value, 3) for key, value in timings.items()},
        stats={
            "frames": len(frames),
            "detections": sum(len(f.detections) for f in objects.frames),
            "actions": len(journal.actions),
            "states": len(journal.states),
            "rule_candidates": len(from_rules),
            "llm_candidates": len(from_llm),
            "merged_candidates": len(combined),
            "accepted_candidates": len(accepted),
        },
    )
    deadline.remaining()
    write_json(output / "result.json", result)
    progress("done", 1.0)
    return result
