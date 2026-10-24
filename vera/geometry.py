"""Aircraft coordinate frames from detected parts: alignment axis, origin, half-wingspan and canvases."""

import numpy as np


def invert_part_polygon(polygon, source_to_crop):
    vertices = np.asarray(polygon, dtype=np.float64).reshape(-1, 2)
    matrix = np.asarray(source_to_crop, dtype=np.float64).reshape(3, 3)
    if len(vertices) < 3 or not np.isfinite(vertices).all() or not np.isfinite(matrix).all():
        raise ValueError("invalid part polygon or transform")
    inverse = np.linalg.inv(matrix)
    mapped = np.c_[vertices, np.ones(len(vertices))] @ inverse.T
    if np.any(np.abs(mapped[:, 2]) < 1e-12):
        raise ValueError("part polygon maps to infinity")
    result = mapped[:, :2] / mapped[:, 2:3]
    if not np.isfinite(result).all():
        raise ValueError("non-finite source polygon")
    return result


def polygon_centroid(vertices):
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
    following = np.roll(points, -1, axis=0)
    cross = points[:, 0] * following[:, 1] - following[:, 0] * points[:, 1]
    twice_area = cross.sum()
    if abs(twice_area) < 1e-10:
        raise ValueError("degenerate polygon")
    center = ((points + following) * cross[:, None]).sum(axis=0) / (3 * twice_area)
    if not np.isfinite(center).all():
        raise ValueError("non-finite polygon centroid")
    return center


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm < 1e-10:
        raise ValueError("degenerate aircraft axis")
    return vector / norm


def _clip_to_side(polygon, anchor, transverse, side):
    """Clip a polygon to one side of a line through anchor."""
    vertices = np.asarray(polygon, dtype=np.float64)
    kept = []
    previous = vertices[-1]
    previous_distance = side * float(np.dot(previous - anchor, transverse))
    for current in vertices:
        current_distance = side * float(np.dot(current - anchor, transverse))
        previous_inside = previous_distance >= -1e-9
        current_inside = current_distance >= -1e-9
        if previous_inside != current_inside:
            fraction = previous_distance / (previous_distance - current_distance)
            kept.append(previous + fraction * (current - previous))
        if current_inside:
            kept.append(current)
        previous, previous_distance = current, current_distance
    if len(kept) < 3:
        raise ValueError("both-side wing clip is degenerate")
    clipped = np.asarray(kept, dtype=np.float64)
    polygon_centroid(clipped)
    return clipped


def _temporary_axis(wings, tails):
    wing_mean = np.mean([centroid for _, centroid in wings], axis=0)
    if tails:
        tail_mean = np.mean([centroid for _, centroid in tails], axis=0)
        try:
            return _unit(tail_mean - wing_mean)
        except ValueError:
            pass
    vertices = np.concatenate([polygon for polygon, _ in wings])
    _, singular_values, directions = np.linalg.svd(vertices - vertices.mean(0),
                                                    full_matrices=False)
    if singular_values[0] < 1e-10:
        raise ValueError("degenerate wing span direction")
    span_axis = _unit(directions[0])
    return np.array([-span_axis[1], span_axis[0]])


def align_from_parts(parts, source_to_crop, proposal_box, score_threshold=.30):
    """Split wing evidence by side and return robust aircraft coordinates."""
    wings, tails = [], []
    for prediction in parts:
        if float(prediction["score"]) < score_threshold:
            continue
        kind = str(prediction["refined_type"]).split("|", 1)[0]
        if kind not in ("main_wing", "horizontal_tail"):
            continue
        polygon = invert_part_polygon(prediction["polygon"], source_to_crop)
        try:
            centroid = polygon_centroid(polygon)
        except ValueError:
            continue
        (wings if kind == "main_wing" else tails).append((polygon, centroid))
    if not wings:
        return {"status": "unaligned", "reason": "no_main_wing", "wing_count": 0,
                "tail_count": len(tails), "alignment_mode": "no_main_wing",
                "tail_mode": "tail_direction" if tails else "no_tail"}
    x0, y0, x1, y1 = map(float, proposal_box)
    if not np.isfinite([x0, y0, x1, y1]).all() or x1 <= x0 or y1 <= y0:
        raise ValueError("invalid proposal box for temporary axis")
    anchor = np.array([(x0 + x1) / 2, (y0 + y1) / 2])
    tail_mode = "tail_direction" if tails else "no_tail"
    try:
        temporary_axis = _temporary_axis(wings, tails)
        temporary_transverse = np.array([-temporary_axis[1], temporary_axis[0]])
        all_vertices = np.concatenate([polygon for polygon, _ in wings])
        preliminary = float(np.max(np.abs((all_vertices - anchor) @ temporary_transverse)))
        if preliminary <= 1e-8:
            raise ValueError("degenerate temporary half wingspan")
        sides = {-1: [], 1: []}
        crossing_count = 0
        for polygon, centroid in wings:
            signed = (polygon - anchor) @ temporary_transverse
            negative, positive = max(0., -float(signed.min())), max(0., float(signed.max()))
            if negative >= .3 * preliminary and positive >= .3 * preliminary:
                for side in (-1, 1):
                    clipped = _clip_to_side(polygon, anchor, temporary_transverse, side)
                    sides[side].append((clipped, polygon_centroid(clipped)))
                crossing_count += 1
            else:
                side = -1 if np.dot(centroid - anchor, temporary_transverse) < 0 else 1
                sides[side].append((polygon, centroid))
        if not sides[-1] or not sides[1]:
            return {"status": "unaligned", "reason": "single_side_main_wing",
                    "alignment_mode": "single_side_unaligned", "tail_mode": tail_mode,
                    "wing_count": len(wings), "tail_count": len(tails),
                    "crossing_polygon_count": crossing_count}
        left = np.mean([centroid for _, centroid in sides[-1]], axis=0)
        right = np.mean([centroid for _, centroid in sides[1]], axis=0)
        origin = (left + right) / 2
        span_axis = _unit(right - left)
        axis = np.array([-span_axis[1], span_axis[0]])
        if tails:
            tail = np.mean([centroid for _, centroid in tails], axis=0)
            if np.dot(tail - origin, axis) < 0:
                axis = -axis
            axes = [axis]
        else:
            axes = [axis, -axis]
        left_extent = max(float(np.max(-((polygon - origin) @ span_axis)))
                          for polygon, _ in sides[-1])
        right_extent = max(float(np.max((polygon - origin) @ span_axis))
                           for polygon, _ in sides[1])
        half_wingspan = (left_extent + right_extent) / 2
        if not np.isfinite(half_wingspan) or half_wingspan <= 1e-8:
            raise ValueError("degenerate half wingspan")
    except ValueError as error:
        return {"status": "unaligned", "reason": str(error), "wing_count": len(wings),
                "tail_count": len(tails), "alignment_mode": "geometry_degenerate",
                "tail_mode": tail_mode}
    return {"status": "aligned", "origin": origin.tolist(),
            "axes": [direction.tolist() for direction in axes],
            "half_wingspan": float(half_wingspan),
            "axis_source": "wing_centroids_perpendicular",
            "alignment_mode": "crossing_single_polygon" if len(wings) == 1 else "two_sided",
            "tail_mode": tail_mode, "left_wing_centroid": left.tolist(),
            "right_wing_centroid": right.tolist(),
            "crossing_polygon_count": crossing_count,
            "wing_count": len(wings), "tail_count": len(tails)}


def canvas_affine(alignment, axis, mirror=False, width=448, height=224):
    """Homogeneous source-to-canvas transform; tail points right."""
    if alignment["status"] != "aligned":
        raise ValueError("cannot render unaligned aircraft")
    origin = np.asarray(alignment["origin"], dtype=np.float64)
    axis = _unit(axis)
    transverse = np.array([-axis[1], axis[0]]) * (-1 if mirror else 1)
    span = float(alignment["half_wingspan"])
    pixels_per_halfspan = width / 4.8
    if not np.isclose(pixels_per_halfspan, height / 2.4):
        raise ValueError("canvas changes aircraft aspect ratio")
    scale = pixels_per_halfspan / span
    result = np.eye(3, dtype=np.float64)
    result[0, :2] = axis * scale
    result[1, :2] = transverse * scale
    result[0, 2] = width / 2 - np.dot(result[0, :2], origin)
    result[1, 2] = height / 2 - np.dot(result[1, :2], origin)
    return result
