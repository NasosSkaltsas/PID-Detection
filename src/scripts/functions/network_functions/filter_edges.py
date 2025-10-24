import numpy as np
import math

def _is_horizontal(p1, p2, tol):
    return abs(p1[1] - p2[1]) <= tol

def _is_vertical(p1, p2, tol):
    return abs(p1[0] - p2[0]) <= tol

def _interval_overlap(a, b):
    """Return overlap length between intervals a=[a0,a1], b=[b0,b1]."""
    return min(a[1], b[1]) - max(a[0], b[0])

def prune_overlapping_edges(G, axis_tol=1e-6, overlap_tol=1e-9, len_tol=1e-9):
    """
    Remove edges that are horizontal or vertical (within axis_tol) and
    either overlap or fully contain smaller edges. For any overlapping pair,
    the longer edge is removed (if equal, drop one deterministically).
    """
    # Collect horizontal and vertical edges with their projections
    horiz = []  # (u,v,y, x0,x1, length)
    vert  = []  # (u,v,x, y0,y1, length)

    for u, v, d in G.edges(data=True):
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']
        p1, p2 = (x1, y1), (x2, y2)
        L = float(d.get('weight', np.hypot(x1 - x2, y1 - y2)))

        if _is_horizontal(p1, p2, axis_tol):
            y = 0.5 * (y1 + y2)
            x0, x1p = sorted((x1, x2))
            horiz.append((u, v, y, x0, x1p, L))
        elif _is_vertical(p1, p2, axis_tol):
            x = 0.5 * (x1 + x2)
            y0, y1p = sorted((y1, y2))
            vert.append((u, v, x, y0, y1p, L))

    to_remove = set()

    def sweep(items, axis='h'):
        # Pairwise compare edges that lie on (approximately) the same line
        # For horizontals: match by y within axis_tol; for verticals: by x within axis_tol.
        n = len(items)
        for i in range(n):
            ui, vi, line_i, a0i, a1i, Li = items[i]
            for j in range(i + 1, n):
                uj, vj, line_j, a0j, a1j, Lj = items[j]

                # Must be on the same line within tolerance
                if abs(line_i - line_j) > axis_tol:
                    continue

                # Compute projection overlap on the axis
                ov = _interval_overlap((a0i, a1i), (a0j, a1j))

                # Consider overlaps or touching (>= overlap_tol)
                if ov >= overlap_tol:
                    # If one contains the other or they partially overlap,
                    # drop the longer one (redundant "including" edge).
                    if Li > Lj + len_tol:
                        to_remove.add(tuple(sorted((ui, vi))))
                    elif Lj > Li + len_tol:
                        to_remove.add(tuple(sorted((uj, vj))))
                    else:
                        # Equal (within len_tol): remove one deterministically
                        # to avoid removing both.
                        cand_i = tuple(sorted((ui, vi)))
                        cand_j = tuple(sorted((uj, vj)))
                        to_remove.add(max(cand_i, cand_j))

    sweep(horiz, axis='h')
    sweep(vert, axis='v')

    # Apply removals
    G.remove_edges_from(list(to_remove))

def prune_long_diagonals(G, threshold=64.0, axis_tol=1.0):
    """
    Remove edges that are NOT approximately horizontal or vertical
    (i.e., both dx > axis_tol and dy > axis_tol)
    AND are longer than the given threshold.

    Parameters
    ----------
    G : networkx.Graph
        Graph with nodes having 'x' and 'y' attributes.
    threshold : float
        Maximum allowed length for diagonal edges.
    axis_tol : float
        Tolerance for deciding what counts as axis-aligned.
        (dx <= axis_tol or dy <= axis_tol means horizontal/vertical)
    """
    edges_to_remove = []

    for u, v, d in G.edges(data=True):
        x1, y1 = G.nodes[u]['x'], G.nodes[u]['y']
        x2, y2 = G.nodes[v]['x'], G.nodes[v]['y']

        dx, dy = abs(x1 - x2), abs(y1 - y2)
        w = d.get('weight', (dx**2 + dy**2)**0.5)

        # Remove if not horizontal/vertical and longer than threshold
        if (dx > axis_tol and dy > axis_tol and w > threshold):
            edges_to_remove.append((u, v))

    G.remove_edges_from(edges_to_remove)


def prune_nodes_with_multi_edges(G, angle_tol_deg=5.0, len_tol=1e-9):
    """
    For any node with degree > 3, keep at most:
      - one edge ~horizontal (angle ≈ 0° or 180° within angle_tol),
      - one edge with positive angle (>0°),
      - one edge with negative angle (<0°).
    Among candidates in each category, keep the shortest (by edge weight).
    Ties within len_tol are broken deterministically.

    Angles are measured from the node toward its neighbor, relative to +x axis,
    using atan2(dy, dx) in (-π, π].
    """
    tol = math.radians(angle_tol_deg)

    def _is_near_horizontal(theta):
        # near 0 or π (±π) within tolerance
        return min(abs(theta), abs(abs(theta) - math.pi)) <= tol

    def _pick_best(group):
        # group: list of (weight, edge_tuple)
        if not group:
            return None
        best_w, best_e = group[0]
        for w, e in group[1:]:
            if w + len_tol < best_w:
                best_w, best_e = w, e
            elif abs(w - best_w) <= len_tol:
                # deterministic tie-breaker
                best_e = min(best_e, e)
        return best_e

    # Collect which edges to keep across all high-degree nodes
    to_keep = set()
    high_nodes = [n for n in G.nodes if G.degree(n) > 3]

    for n in high_nodes:
        cx, cy = G.nodes[n]['x'], G.nodes[n]['y']

        horiz, pos, neg = [], [], []  # lists of (weight, edge)

        for nbr in list(G.neighbors(n)):
            x2, y2 = G.nodes[nbr]['x'], G.nodes[nbr]['y']
            dx, dy = x2 - cx, y2 - cy
            theta = math.atan2(dy, dx)          # (-π, π]
            w = G[n][nbr].get('weight', math.hypot(dx, dy))
            edge = tuple(sorted((n, nbr)))

            if _is_near_horizontal(theta):
                horiz.append((w, edge))
            elif theta > 0:
                pos.append((w, edge))
            else:  # theta < 0
                neg.append((w, edge))

        # Choose one per category (if available)
        for group in (horiz, pos, neg):
            chosen = _pick_best(group)
            if chosen is not None:
                to_keep.add(chosen)

    # Remove other incident edges for high-degree nodes (but keep anything selected)
    incident = set()
    for n in high_nodes:
        for nbr in list(G.neighbors(n)):
            incident.add(tuple(sorted((n, nbr))))

    to_remove = [e for e in incident if e not in to_keep]
    G.remove_edges_from(to_remove)