#!/usr/bin/env python3
"""Small-team face enrollment and recognition demo built on InsightFace.

The demo deliberately keeps identity management outside the recognition model:
InsightFace produces normalized embeddings, SQLite stores them, and this script
performs open-set 1:N matching with an UNKNOWN result below a threshold.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from insightface.app import FaceAnalysis


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_MODEL = "buffalo_l"
DEFAULT_THRESHOLD = 0.40
DEFAULT_MARGIN = 0.05


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def normalize(vector: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 0:
        raise ValueError("Cannot normalize an empty embedding")
    return array / norm


def read_image(path: Path) -> np.ndarray:
    """Read paths containing non-ASCII characters on Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Unable to decode image: {path}")
    return image


def write_image(path: Path, image: np.ndarray, jpeg_quality: int = 95) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower() or ".jpg"
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if suffix in {".jpg", ".jpeg"} else []
    ok, encoded = cv2.imencode(suffix, image, params)
    if not ok:
        raise ValueError(f"Unable to encode image: {path}")
    encoded.tofile(str(path))


def image_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES)


def identity_directories(dataset: Path) -> list[Path]:
    if not dataset.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset}")
    direct_images = [item for item in dataset.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES]
    if direct_images:
        return [dataset]
    directories = [
        item
        for item in sorted(dataset.iterdir())
        if item.is_dir() and item.name.upper().startswith("FACE_TEAM_") and image_files(item)
    ]
    if not directories:
        raise ValueError(f"No FACE_TEAM_* identity folders or images found under: {dataset}")
    return directories


def create_face_app(
    model_name: str,
    det_size: int,
    providers: Sequence[str] | None = None,
    root: str | None = None,
) -> FaceAnalysis:
    options: dict[str, Any] = {
        "name": model_name,
        "allowed_modules": ["detection", "recognition"],
        "providers": list(providers or ["CPUExecutionProvider"]),
    }
    if root:
        options["root"] = root
    app = FaceAnalysis(**options)
    app.prepare(ctx_id=0, det_size=(det_size, det_size))
    return app


def resize_long_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)
    if max_side <= 0 or longest <= max_side:
        return image
    scale = max_side / float(longest)
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


def face_bbox(face: Any) -> np.ndarray:
    return np.asarray(face.bbox, dtype=np.float32).reshape(4)


def choose_primary_face(faces: Sequence[Any], image_shape: Sequence[int]) -> Any:
    """Prefer the dominant, central face while ignoring small background faces."""
    if not faces:
        raise ValueError("No face detected")
    height, width = image_shape[:2]
    image_area = float(max(1, height * width))
    diagonal = float(max(1.0, np.hypot(width, height)))
    image_center = np.array([width / 2.0, height / 2.0], dtype=np.float32)

    def rank(face: Any) -> float:
        x1, y1, x2, y2 = face_bbox(face)
        area_ratio = max(0.0, (x2 - x1) * (y2 - y1)) / image_area
        center = np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=np.float32)
        center_distance = float(np.linalg.norm(center - image_center)) / diagonal
        detection_score = float(getattr(face, "det_score", 0.0) or 0.0)
        return area_ratio * 5.0 + detection_score * 0.05 - center_distance * 0.10

    return max(faces, key=rank)


def square_face_crop(image: np.ndarray, bbox: Sequence[float], scale: float, output_size: int) -> np.ndarray:
    x1, y1, x2, y2 = np.asarray(bbox, dtype=np.float32)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(float(x2 - x1), float(y2 - y1)) * scale
    side = max(side, 2.0)
    left = int(np.floor(center_x - side / 2.0))
    top = int(np.floor(center_y - side / 2.0))
    right = int(np.ceil(center_x + side / 2.0))
    bottom = int(np.ceil(center_y + side / 2.0))

    height, width = image.shape[:2]
    pad_left = max(0, -left)
    pad_top = max(0, -top)
    pad_right = max(0, right - width)
    pad_bottom = max(0, bottom - height)
    if any((pad_left, pad_top, pad_right, pad_bottom)):
        image = cv2.copyMakeBorder(
            image,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_REPLICATE,
        )
        left += pad_left
        right += pad_left
        top += pad_top
        bottom += pad_top
    crop = image[top:bottom, left:right]
    if crop.size == 0:
        raise ValueError("Computed face crop is empty")
    return cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)


def blur_score(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def augment_face_crop(image: np.ndarray, variant: int) -> tuple[str, np.ndarray]:
    """Apply a deterministic, mild transformation suitable for enrollment crops."""
    height, width = image.shape[:2]
    mode = variant % 8
    if mode == 0:
        return "flip", cv2.flip(image, 1)
    if mode in {1, 2}:
        angle = -5.0 if mode == 1 else 5.0
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        augmented = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return ("rotate_m5" if angle < 0 else "rotate_p5"), augmented
    if mode in {3, 4}:
        factor = 0.90 if mode == 3 else 1.10
        augmented = np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)
        return ("brightness_m10" if factor < 1 else "brightness_p10"), augmented
    if mode in {5, 6}:
        factor = 0.92 if mode == 5 else 1.08
        mean = np.mean(image, axis=(0, 1), keepdims=True)
        augmented = np.clip((image.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
        return ("contrast_m8" if factor < 1 else "contrast_p8"), augmented

    inset_x = max(1, round(width * 0.025))
    inset_y = max(1, round(height * 0.025))
    zoomed = image[inset_y : height - inset_y, inset_x : width - inset_x]
    return "zoom_p5", cv2.resize(zoomed, (width, height), interpolation=cv2.INTER_LINEAR)


class FaceDatabase:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def __enter__(self) -> "FaceDatabase":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS people (
                person_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS face_samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                person_id TEXT NOT NULL REFERENCES people(person_id) ON DELETE CASCADE,
                source_path TEXT NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dim INTEGER NOT NULL,
                det_score REAL NOT NULL,
                blur_score REAL NOT NULL,
                model_name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                UNIQUE(person_id, source_path, model_name)
            );

            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_face_samples_person ON face_samples(person_id);
            """
        )
        self.connection.commit()

    def upsert_person(self, person_id: str, display_name: str | None = None) -> None:
        now = utc_now()
        self.connection.execute(
            """
            INSERT INTO people(person_id, display_name, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(person_id) DO UPDATE SET
                display_name=excluded.display_name,
                updated_at=excluded.updated_at
            """,
            (person_id, display_name or person_id, now, now),
        )

    def clear_person_samples(self, person_id: str) -> None:
        self.connection.execute("DELETE FROM face_samples WHERE person_id=?", (person_id,))

    def add_sample(
        self,
        person_id: str,
        source_path: str,
        embedding: np.ndarray,
        det_score: float,
        sample_blur_score: float,
        model_name: str,
    ) -> None:
        vector = normalize(embedding).astype(np.float32)
        self.connection.execute(
            """
            INSERT INTO face_samples(
                person_id, source_path, embedding, embedding_dim,
                det_score, blur_score, model_name, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(person_id, source_path, model_name) DO UPDATE SET
                embedding=excluded.embedding,
                embedding_dim=excluded.embedding_dim,
                det_score=excluded.det_score,
                blur_score=excluded.blur_score,
                created_at=excluded.created_at
            """,
            (
                person_id,
                source_path,
                vector.tobytes(),
                int(vector.size),
                float(det_score),
                float(sample_blur_score),
                model_name,
                utc_now(),
            ),
        )

    def set_metadata(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def metadata(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM metadata WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def samples(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            """
            SELECT fs.*, p.display_name
            FROM face_samples fs
            JOIN people p ON p.person_id=fs.person_id
            ORDER BY fs.person_id, fs.id
            """
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            vector = np.frombuffer(item.pop("embedding"), dtype=np.float32).copy()
            if vector.size != int(item["embedding_dim"]):
                raise ValueError(f"Corrupt embedding in sample {item['id']}")
            item["embedding"] = normalize(vector)
            result.append(item)
        return result


def build_gallery(
    samples: Iterable[dict[str, Any]],
    exclude_sample_ids: set[int] | None = None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if exclude_sample_ids and int(sample["id"]) in exclude_sample_ids:
            continue
        grouped[str(sample["person_id"])].append(sample)
    gallery: list[dict[str, Any]] = []
    for person_id, items in grouped.items():
        centroid = normalize(np.mean(np.vstack([item["embedding"] for item in items]), axis=0))
        gallery.append(
            {
                "person_id": person_id,
                "display_name": str(items[0]["display_name"]),
                "centroid": centroid,
                "sample_embeddings": [item["embedding"] for item in items],
                "sample_count": len(items),
            }
        )
    return sorted(gallery, key=lambda item: item["person_id"])


def match_embedding(
    embedding: np.ndarray,
    gallery: Sequence[dict[str, Any]],
    threshold: float,
    min_margin: float,
) -> dict[str, Any]:
    query = normalize(embedding)
    candidates: list[dict[str, Any]] = []
    for identity in gallery:
        centroid_similarity = float(np.dot(query, identity["centroid"]))
        best_sample_similarity = max(float(np.dot(query, sample)) for sample in identity["sample_embeddings"])
        candidates.append(
            {
                "person_id": identity["person_id"],
                "display_name": identity["display_name"],
                "similarity": centroid_similarity,
                "best_sample_similarity": best_sample_similarity,
                "sample_count": identity["sample_count"],
            }
        )
    candidates.sort(key=lambda item: item["similarity"], reverse=True)
    if not candidates:
        return {"status": "UNKNOWN", "person_id": None, "similarity": 0.0, "reason": "empty_gallery"}

    best = candidates[0]
    runner_up = candidates[1]["similarity"] if len(candidates) > 1 else None
    margin = float(best["similarity"] - runner_up) if runner_up is not None else None
    passes_threshold = float(best["similarity"]) >= threshold
    passes_margin = margin is None or margin >= min_margin
    known = passes_threshold and passes_margin
    reason = "matched" if known else ("below_threshold" if not passes_threshold else "ambiguous_margin")
    return {
        "status": "KNOWN" if known else "UNKNOWN",
        "person_id": best["person_id"] if known else None,
        "display_name": best["display_name"] if known else None,
        "candidate_person_id": best["person_id"],
        "similarity": float(best["similarity"]),
        "best_sample_similarity": float(best["best_sample_similarity"]),
        "runner_up_similarity": float(runner_up) if runner_up is not None else None,
        "margin": margin,
        "threshold": threshold,
        "min_margin": min_margin,
        "reason": reason,
    }


def command_preprocess(args: argparse.Namespace) -> int:
    source_root = args.input.resolve()
    output_root = args.output.resolve()
    app = create_face_app(args.model, args.det_size)
    records: list[dict[str, Any]] = []
    successful = 0
    augmented_count = 0
    failed = 0

    if source_root == output_root:
        raise ValueError("Preprocess output must be different from the source dataset")

    for identity_dir in identity_directories(source_root):
        person_id = identity_dir.name
        identity_output = output_root / person_id
        if identity_output.exists():
            for stale_image in image_files(identity_output):
                stale_image.unlink()
        processed_crops: list[tuple[Path, Path, np.ndarray, float, float]] = []
        for source in image_files(identity_dir):
            record: dict[str, Any] = {
                "person_id": person_id,
                "source": str(source.resolve()),
                "source_group": source.stem,
                "augmented": False,
            }
            try:
                image = resize_long_side(read_image(source), args.max_side)
                faces = app.get(image)
                primary = choose_primary_face(faces, image.shape)
                crop = square_face_crop(image, face_bbox(primary), args.crop_scale, args.output_size)
                destination = output_root / person_id / f"{source.stem}.jpg"
                write_image(destination, crop)
                score = blur_score(crop)
                record.update(
                    {
                        "status": "ok",
                        "output": str(destination),
                        "faces_detected": len(faces),
                        "det_score": float(primary.det_score),
                        "blur_score": score,
                        "bbox": [float(value) for value in face_bbox(primary)],
                        "quality_warning": "low_blur" if score < args.blur_warning else None,
                    }
                )
                successful += 1
                processed_crops.append((source, destination, crop, float(primary.det_score), score))
                print(f"[OK] {person_id}/{source.name} -> {destination.name} ({len(faces)} face(s))")
            except Exception as exc:  # keep processing the remaining enrollment set
                record.update({"status": "error", "error": str(exc)})
                failed += 1
                print(f"[ERROR] {person_id}/{source.name}: {exc}", file=sys.stderr)
            records.append(record)

        needed = max(0, args.augment_to - len(processed_crops))
        if needed and processed_crops:
            for index in range(needed):
                source, _destination, crop, detection_score, _original_blur = processed_crops[index % len(processed_crops)]
                label, augmented = augment_face_crop(crop, index)
                destination = identity_output / f"{source.stem}__aug{index + 1:02d}_{label}.jpg"
                write_image(destination, augmented)
                score = blur_score(augmented)
                records.append(
                    {
                        "person_id": person_id,
                        "source": str(source.resolve()),
                        "source_group": source.stem,
                        "augmented": True,
                        "augmentation": label,
                        "status": "ok",
                        "output": str(destination),
                        "faces_detected": 1,
                        "det_score": detection_score,
                        "blur_score": score,
                        "bbox": None,
                        "quality_warning": "low_blur" if score < args.blur_warning else None,
                    }
                )
                successful += 1
                augmented_count += 1
                print(f"[AUGMENTED] {person_id}/{destination.name} from {source.name}")

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "processed": successful,
                "augmented": augmented_count,
                "failed": failed,
                "manifest": str(manifest_path),
            },
            ensure_ascii=False,
        )
    )
    return 0 if failed == 0 else 2


def command_enroll(args: argparse.Namespace) -> int:
    app = create_face_app(args.model, args.det_size)
    total = 0
    errors = 0
    with FaceDatabase(args.db.resolve()) as database:
        database.set_metadata("model_name", args.model)
        database.set_metadata("updated_at", utc_now())
        for identity_dir in identity_directories(args.input.resolve()):
            person_id = identity_dir.name
            database.upsert_person(person_id)
            if args.replace:
                database.clear_person_samples(person_id)
            identity_count = 0
            for source in image_files(identity_dir):
                try:
                    image = read_image(source)
                    faces = app.get(image)
                    primary = choose_primary_face(faces, image.shape)
                    embedding = getattr(primary, "normed_embedding", None)
                    if embedding is None:
                        raise ValueError("Recognition embedding is unavailable")
                    database.add_sample(
                        person_id=person_id,
                        source_path=str(source.resolve()),
                        embedding=embedding,
                        det_score=float(primary.det_score),
                        sample_blur_score=blur_score(image),
                        model_name=args.model,
                    )
                    identity_count += 1
                    total += 1
                except Exception as exc:
                    errors += 1
                    print(f"[ERROR] {person_id}/{source.name}: {exc}", file=sys.stderr)
            print(f"[ENROLLED] {person_id}: {identity_count} sample(s)")
        database.connection.commit()
    print(json.dumps({"database": str(args.db.resolve()), "enrolled": total, "failed": errors}, ensure_ascii=False))
    return 0 if errors == 0 else 2


def command_recognize(args: argparse.Namespace) -> int:
    with FaceDatabase(args.db.resolve()) as database:
        samples = database.samples()
        model_name = args.model or database.metadata("model_name") or DEFAULT_MODEL
    gallery = build_gallery(samples)
    app = create_face_app(model_name, args.det_size)
    image = read_image(args.image.resolve())
    faces = app.get(image)
    results: list[dict[str, Any]] = []
    annotated = image.copy()
    for index, face in enumerate(faces):
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            continue
        match = match_embedding(embedding, gallery, args.threshold, args.min_margin)
        bbox = face_bbox(face)
        result = {
            "face_index": index,
            "bbox": [float(value) for value in bbox],
            "det_score": float(face.det_score),
            **match,
        }
        results.append(result)
        x1, y1, x2, y2 = bbox.astype(int)
        known = match["status"] == "KNOWN"
        color = (0, 180, 0) if known else (0, 0, 220)
        label_id = match["person_id"] if known else "UNKNOWN"
        label = f"{label_id} {match['similarity']:.3f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
        cv2.putText(annotated, label, (x1, max(24, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    if args.output:
        write_image(args.output.resolve(), annotated)
    payload = {
        "image": str(args.image.resolve()),
        "database": str(args.db.resolve()),
        "model_name": model_name,
        "gallery_people": len(gallery),
        "faces": results,
        "output": str(args.output.resolve()) if args.output else None,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def percentile(values: Sequence[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


def command_validate(args: argparse.Namespace) -> int:
    with FaceDatabase(args.db.resolve()) as database:
        samples = database.samples()
    if len(samples) < 2:
        raise ValueError("At least two samples are required for leave-one-out validation")

    def source_group(sample: dict[str, Any]) -> str:
        return Path(str(sample["source_path"])).stem.split("__aug", 1)[0]

    validation_samples = [sample for sample in samples if "__aug" not in Path(str(sample["source_path"])).stem]
    if not validation_samples:
        raise ValueError("No original samples are available for validation")

    correct = 0
    false_unknown = 0
    wrong_identity = 0
    scores: list[float] = []
    impostor_scores: list[float] = []
    rows: list[dict[str, Any]] = []
    for sample in validation_samples:
        group = source_group(sample)
        excluded_ids = {
            int(candidate["id"])
            for candidate in samples
            if candidate["person_id"] == sample["person_id"] and source_group(candidate) == group
        }
        gallery = build_gallery(samples, exclude_sample_ids=excluded_ids)
        match = match_embedding(sample["embedding"], gallery, args.threshold, args.min_margin)
        predicted = match["person_id"]
        expected = sample["person_id"]
        if predicted == expected:
            correct += 1
        elif match["status"] == "UNKNOWN":
            false_unknown += 1
        else:
            wrong_identity += 1
        scores.append(float(match["similarity"]))
        other_candidates = [
            float(np.dot(sample["embedding"], identity["centroid"]))
            for identity in gallery
            if identity["person_id"] != expected
        ]
        if other_candidates:
            impostor_scores.append(max(other_candidates))
        rows.append(
            {
                "sample_id": sample["id"],
                "person_id": expected,
                "source_path": sample["source_path"],
                "source_group": group,
                "predicted_person_id": predicted,
                "correct": predicted == expected,
                "status": match["status"],
                "similarity": match["similarity"],
                "reason": match["reason"],
            }
        )

    per_person: dict[str, dict[str, Any]] = {}
    for person_id in sorted({str(sample["person_id"]) for sample in validation_samples}):
        identity_rows = [row for row in rows if row["person_id"] == person_id]
        identity_scores = [float(row["similarity"]) for row in identity_rows]
        identity_correct = sum(bool(row["correct"]) for row in identity_rows)
        per_person[person_id] = {
            "samples": len(identity_rows),
            "correct": identity_correct,
            "correct_rate": identity_correct / len(identity_rows),
            "false_unknown": sum(row["status"] == "UNKNOWN" for row in identity_rows),
            "wrong_identity": sum(
                row["predicted_person_id"] is not None and row["predicted_person_id"] != person_id
                for row in identity_rows
            ),
            "similarity_min": min(identity_scores),
            "similarity_mean": float(np.mean(identity_scores)),
        }

    report = {
        "database": str(args.db.resolve()),
        "validation": "leave-one-source-group-out",
        "samples": len(validation_samples),
        "gallery_samples": len(samples),
        "augmented_samples": len(samples) - len(validation_samples),
        "people": len({sample["person_id"] for sample in samples}),
        "threshold": args.threshold,
        "min_margin": args.min_margin,
        "correct": correct,
        "correct_rate": correct / len(validation_samples),
        "false_unknown": false_unknown,
        "wrong_identity": wrong_identity,
        "per_person": per_person,
        "genuine_similarity": {
            "min": min(scores),
            "p10": percentile(scores, 10),
            "median": percentile(scores, 50),
            "mean": float(np.mean(scores)),
            "max": max(scores),
        },
        "impostor_similarity": {
            "samples": len(impostor_scores),
            "max": max(impostor_scores) if impostor_scores else None,
            "note": None if impostor_scores else "Add at least one different person to measure false matches.",
        },
        "details": rows if args.details else None,
    }
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if wrong_identity == 0 else 3


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser("preprocess", help="Detect and crop the primary face in enrollment photos")
    preprocess_parser.add_argument("--input", type=Path, default=Path("face_datasets"))
    preprocess_parser.add_argument("--output", type=Path, default=Path(".face_demo/preprocessed"))
    preprocess_parser.add_argument("--model", default=DEFAULT_MODEL)
    preprocess_parser.add_argument("--det-size", type=int, default=640)
    preprocess_parser.add_argument("--max-side", type=int, default=1600)
    preprocess_parser.add_argument("--output-size", type=int, default=640)
    preprocess_parser.add_argument("--crop-scale", type=float, default=1.55)
    preprocess_parser.add_argument("--blur-warning", type=float, default=60.0)
    preprocess_parser.add_argument(
        "--augment-to",
        type=int,
        default=15,
        help="Generate mild augmentations until each identity has this many crops; 0 disables augmentation",
    )
    preprocess_parser.set_defaults(handler=command_preprocess)

    enroll_parser = subparsers.add_parser("enroll", help="Extract embeddings and write them to SQLite")
    enroll_parser.add_argument("--input", type=Path, default=Path(".face_demo/preprocessed"))
    enroll_parser.add_argument("--db", type=Path, default=Path(".face_demo/team_faces.sqlite3"))
    enroll_parser.add_argument("--model", default=DEFAULT_MODEL)
    enroll_parser.add_argument("--det-size", type=int, default=640)
    enroll_parser.add_argument("--replace", action="store_true", help="Replace all existing samples for identities in --input")
    enroll_parser.set_defaults(handler=command_enroll)

    recognize_parser = subparsers.add_parser("recognize", help="Recognize every face in one image")
    recognize_parser.add_argument("--image", type=Path, required=True)
    recognize_parser.add_argument("--db", type=Path, default=Path(".face_demo/team_faces.sqlite3"))
    recognize_parser.add_argument("--output", type=Path)
    recognize_parser.add_argument("--json-output", type=Path, help="Write machine-readable results to this JSON file")
    recognize_parser.add_argument("--model", help="Override the model recorded in the database")
    recognize_parser.add_argument("--det-size", type=int, default=640)
    recognize_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    recognize_parser.add_argument("--min-margin", type=float, default=DEFAULT_MARGIN)
    recognize_parser.set_defaults(handler=command_recognize)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate original photos while excluding each source and its augmentations from the gallery",
    )
    validate_parser.add_argument("--db", type=Path, default=Path(".face_demo/team_faces.sqlite3"))
    validate_parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    validate_parser.add_argument("--min-margin", type=float, default=DEFAULT_MARGIN)
    validate_parser.add_argument("--details", action="store_true")
    validate_parser.add_argument("--json-output", type=Path)
    validate_parser.set_defaults(handler=command_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ValueError, sqlite3.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
