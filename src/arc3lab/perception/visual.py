from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from math import log1p
from typing import Any, Iterable

import numpy as np

from arc3lab.types import ActionSpec, Component, Scene


VisualSignature = tuple[int, str, int]


def visual_signature(component: Component) -> VisualSignature:
    """Translation-invariant visual identity used inside one game.

    Color is retained because ARC environments frequently use color as an intrinsic
    visual attribute. Absolute coordinates are deliberately excluded.
    """
    return (int(component.color), str(component.shape_hash), int(component.pixels))


def _center_distance(a: Component, b: Component) -> int:
    ar, ac = a.center_cell
    br, bc = b.center_cell
    return abs(ar - br) + abs(ac - bc)


def _component_match_cost(a: Component, b: Component) -> float | None:
    """Conservative cost for matching an object across adjacent frames.

    Exact color+shape matches may move arbitrarily. Shape-changing matches are allowed
    only locally and only when color/size are still compatible, which avoids inventing
    object permanence across unrelated scene changes.
    """
    if visual_signature(a) == visual_signature(b):
        return float(_center_distance(a, b))
    if int(a.color) != int(b.color):
        return None
    ratio = max(a.pixels, b.pixels) / max(1, min(a.pixels, b.pixels))
    if ratio > 1.8:
        return None
    dh = abs(a.height - b.height)
    dw = abs(a.width - b.width)
    dist = _center_distance(a, b)
    if dist > 4 or dh > 2 or dw > 2:
        return None
    return 8.0 + dist + 1.5 * (ratio - 1.0) + 0.5 * (dh + dw)


@dataclass(slots=True)
class TrackState:
    track_id: int
    signature: VisualSignature
    first_step: int
    last_step: int
    seen: int = 1
    motion_events: int = 0
    transform_events: int = 0
    last_center: tuple[int, int] = (0, 0)
    last_delta: tuple[int, int] = (0, 0)
    current_index: int | None = None
    last_action: int | None = None
    disappeared: bool = False

    @property
    def age(self) -> int:
        return max(1, self.last_step - self.first_step + 1)

    @property
    def persistence(self) -> float:
        return self.seen / max(self.age, 1)


class VisualTracker:
    """Object-centric temporal tracker over ARC connected components.

    Tracks are intentionally conservative: exact shape/color components are associated
    across arbitrary translations; mild local shape transformations may remain linked;
    everything else becomes an appearance/disappearance event rather than a fabricated
    correspondence.
    """

    def __init__(self, *, event_history: int = 96) -> None:
        self.next_track_id = 1
        self.tracks: dict[int, TrackState] = {}
        self.current_by_component: dict[int, int] = {}
        self.last_scene: Scene | None = None
        self.last_step = -1
        self.recent_events: deque[dict[str, Any]] = deque(maxlen=max(16, int(event_history)))

    def reset_link(self) -> None:
        """Break frame-to-frame association while retaining accumulated track history."""
        self.last_scene = None
        self.current_by_component = {}

    def _new_track(self, component: Component, index: int, step: int) -> int:
        tid = self.next_track_id
        self.next_track_id += 1
        self.tracks[tid] = TrackState(
            track_id=tid,
            signature=visual_signature(component),
            first_step=step,
            last_step=step,
            last_center=component.center_cell,
            current_index=index,
        )
        return tid

    def _match(self, before: Scene, after: Scene) -> list[tuple[int, int, float]]:
        candidates: list[tuple[float, int, int]] = []
        for i, a in enumerate(before.components):
            for j, b in enumerate(after.components):
                cost = _component_match_cost(a, b)
                if cost is not None:
                    candidates.append((cost, i, j))
        left = set(range(len(before.components)))
        right = set(range(len(after.components)))
        out: list[tuple[int, int, float]] = []
        for cost, i, j in sorted(candidates):
            if i in left and j in right:
                left.remove(i)
                right.remove(j)
                out.append((i, j, cost))
        return out

    def observe(
        self,
        scene: Scene,
        *,
        step: int,
        action: ActionSpec | None = None,
        level_changed: bool = False,
    ) -> list[dict[str, Any]]:
        step = int(step)
        self.last_step = step
        events: list[dict[str, Any]] = []

        if self.last_scene is None or level_changed:
            self.current_by_component = {}
            for j, comp in enumerate(scene.components):
                tid = self._new_track(comp, j, step)
                self.current_by_component[j] = tid
                event = {
                    "kind": "appear",
                    "step": step,
                    "track": tid,
                    "object": j,
                    "center": list(comp.center_cell),
                    "signature": list(visual_signature(comp)),
                }
                events.append(event)
                self.recent_events.append(event)
            self.last_scene = scene
            return events

        before = self.last_scene
        previous_map = dict(self.current_by_component)
        matches = self._match(before, scene)
        matched_before = {i for i, _, _ in matches}
        matched_after = {j for _, j, _ in matches}
        new_map: dict[int, int] = {}

        for i, j, _ in matches:
            tid = previous_map.get(i)
            if tid is None:
                tid = self._new_track(before.components[i], i, max(0, step - 1))
            track = self.tracks[tid]
            before_comp = before.components[i]
            after_comp = scene.components[j]
            br, bc = before_comp.center_cell
            ar, ac = after_comp.center_cell
            delta = (ar - br, ac - bc)
            transformed = visual_signature(before_comp) != visual_signature(after_comp)
            moved = delta != (0, 0)
            track.last_step = step
            track.seen += 1
            track.last_center = after_comp.center_cell
            track.last_delta = delta
            track.current_index = j
            track.last_action = int(action.action_id) if action is not None else None
            track.disappeared = False
            if moved:
                track.motion_events += 1
            if transformed:
                track.transform_events += 1
                track.signature = visual_signature(after_comp)
            new_map[j] = tid
            if moved or transformed:
                event = {
                    "kind": "transform" if transformed else "move",
                    "step": step,
                    "track": tid,
                    "object": j,
                    "from": [br, bc],
                    "to": [ar, ac],
                    "delta": [delta[0], delta[1]],
                    "action": int(action.action_id) if action is not None else None,
                    "transformed": transformed,
                }
                events.append(event)
                self.recent_events.append(event)

        for i, comp in enumerate(before.components):
            if i in matched_before:
                continue
            tid = previous_map.get(i)
            if tid is None:
                continue
            track = self.tracks[tid]
            track.current_index = None
            track.disappeared = True
            event = {
                "kind": "disappear",
                "step": step,
                "track": tid,
                "from": list(comp.center_cell),
                "signature": list(visual_signature(comp)),
                "action": int(action.action_id) if action is not None else None,
            }
            events.append(event)
            self.recent_events.append(event)

        for j, comp in enumerate(scene.components):
            if j in matched_after:
                continue
            tid = self._new_track(comp, j, step)
            new_map[j] = tid
            event = {
                "kind": "appear",
                "track": tid,
                "object": j,
                "center": list(comp.center_cell),
                "signature": list(visual_signature(comp)),
                "action": int(action.action_id) if action is not None else None,
            }
            events.append(event)
            self.recent_events.append(event)

        self.current_by_component = new_map
        self.last_scene = scene
        return events

    def track_for_component(self, component_index: int) -> int | None:
        return self.current_by_component.get(int(component_index))

    def component_for_track(self, track_id: int) -> int | None:
        track = self.tracks.get(int(track_id))
        return None if track is None else track.current_index

    def tracks_with_signature(self, signature: VisualSignature) -> list[int]:
        return [tid for tid, track in self.tracks.items() if track.signature == signature]

    def recent_motion_groups(self, *, limit: int = 8) -> list[dict[str, Any]]:
        grouped: dict[tuple[int, int | None, tuple[int, int]], list[int]] = {}
        for event in self.recent_events:
            if event.get("kind") not in {"move", "transform"}:
                continue
            delta_raw = event.get("delta")
            if not isinstance(delta_raw, list) or len(delta_raw) != 2:
                continue
            key = (
                int(event.get("step", -1)),
                event.get("action"),
                (int(delta_raw[0]), int(delta_raw[1])),
            )
            grouped.setdefault(key, []).append(int(event["track"]))
        records = []
        for (step, action, delta), tracks in grouped.items():
            if len(tracks) < 2:
                continue
            records.append(
                {
                    "step": step,
                    "action": action,
                    "delta": [delta[0], delta[1]],
                    "tracks": sorted(set(tracks)),
                    "parts": len(set(tracks)),
                }
            )
        records.sort(key=lambda x: (x["step"], x["parts"]), reverse=True)
        return records[: max(1, int(limit))]

    @staticmethod
    def salience(component: Component, color_count: int, track: TrackState | None) -> float:
        rarity = 1.0 / max(1, color_count)
        size = 1.0 / max(1.0, component.pixels ** 0.5)
        motion = 0.0 if track is None else min(1.0, 0.2 * track.motion_events)
        transform = 0.0 if track is None else min(0.8, 0.25 * track.transform_events)
        persistence = 0.0 if track is None else 0.25 * track.persistence
        return 0.65 * rarity + 0.25 * size + motion + transform + persistence

    def current_objects(self, scene: Scene, *, limit: int = 24) -> list[dict[str, Any]]:
        color_counts = Counter(int(c.color) for c in scene.components)
        records = []
        for i, comp in enumerate(scene.components):
            tid = self.track_for_component(i)
            track = self.tracks.get(tid) if tid is not None else None
            record = {
                "object": i,
                "track": tid,
                "signature": list(visual_signature(comp)),
                "color": int(comp.color),
                "pixels": int(comp.pixels),
                "bbox": list(comp.bbox),
                "center": list(comp.center_cell),
                "edge": bool(comp.edge_touch),
                "same_color_count": int(color_counts[int(comp.color)]),
                "age": track.age if track is not None else 1,
                "seen": track.seen if track is not None else 1,
                "motion_events": track.motion_events if track is not None else 0,
                "transform_events": track.transform_events if track is not None else 0,
                "last_delta": list(track.last_delta) if track is not None else [0, 0],
                "salience": round(self.salience(comp, color_counts[int(comp.color)], track), 4),
            }
            records.append(record)
        records.sort(key=lambda x: (-x["salience"], x["object"]))
        return records[: max(1, int(limit))]

    def summary(self, scene: Scene, *, limit: int = 24) -> dict[str, Any]:
        return {
            "tracked_objects": self.current_objects(scene, limit=limit),
            "active_tracks": len(self.current_by_component),
            "historical_tracks": len(self.tracks),
            "co_moving_groups": self.recent_motion_groups(limit=8),
            "recent_events": list(self.recent_events)[-16:],
        }


def temporal_visual_packet(
    grids: list[np.ndarray],
    *,
    background: int = 0,
    vanished_marker: int = 6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build one lossless 2x2 temporal packet for a VLM.

    top-left=t-2, top-right=t-1, bottom-left=current, bottom-right=delta. The delta
    quadrant is diagnostic rather than an environment frame: unchanged cells are the
    supplied background, newly changed cells keep their *current* color, and cells that
    vanished to background use a marker color. A one-cell background gutter separates
    panels. This lets an existing single-image adapter carry recent visual history.
    """
    if not grids:
        raise ValueError("temporal_visual_packet requires at least one grid")
    arrs = [np.asarray(g, dtype=np.int8) for g in grids[-3:]]
    shape = arrs[-1].shape
    if any(a.shape != shape for a in arrs):
        # Scene-size changes are rare; only the current shape is semantically valid.
        arrs = [a for a in arrs if a.shape == shape]
    while len(arrs) < 3:
        arrs.insert(0, arrs[0].copy())
    older, previous, current = arrs[-3], arrs[-2], arrs[-1]
    delta = np.full_like(current, int(background))
    changed = previous != current
    delta[changed] = current[changed]
    vanished = changed & (current == int(background)) & (previous != int(background))
    marker = int(vanished_marker) % 16
    if marker == int(background):
        marker = (marker + 1) % 16
    delta[vanished] = marker

    h, w = shape
    packet = np.full((2 * h + 1, 2 * w + 1), int(background), dtype=np.int8)
    packet[:h, :w] = older
    packet[:h, w + 1 :] = previous
    packet[h + 1 :, :w] = current
    packet[h + 1 :, w + 1 :] = delta
    meta = {
        "layout": {
            "top_left": "t-2",
            "top_right": "t-1",
            "bottom_left": "current",
            "bottom_right": f"delta(t-1,current); vanished_marker={marker}",
        },
        "panel_shape": [h, w],
        "packet_shape": list(packet.shape),
    }
    return packet, meta
