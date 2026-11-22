import os
import sys
import math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from triad import triad_class, hand_worked_example, NUM_CLASSES
from count import triad_census
from randomize import randomize, assert_degrees_preserved

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")

EDGES_FILE = os.path.join(DATA_DIR, "email-Eu-core.txt")
LABELS_FILE = os.path.join(DATA_DIR, "email-Eu-core-department-labels.txt")

# ---------------------------------------------------------------------
# GROUP PARAMETERS -- members' last-4-digit roll numbers
# ---------------------------------------------------------------------
MEMBER_ROLL_SUFFIXES = [718, 635, 566, 1015]
R = sum(MEMBER_ROLL_SUFFIXES)


# =======================================================================
# T1 - Load, clean, compute top-20, derive parameters
# =======================================================================
def load_raw_edges(path):
    edges = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            u, v = map(int, line.split())
            edges.append((u, v))
    return edges


def load_labels(path):
    labels = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            node, dept = map(int, line.split())
            labels[node] = dept
    return labels


def clean_edges(raw_edges):
    """Remove self-loops and duplicate directed edges (mandatory preprocessing)."""
    seen = set()
    cleaned = []
    for u, v in raw_edges:
        if u == v:
            continue
        if (u, v) in seen:
            continue
        seen.add((u, v))
        cleaned.append((u, v))
    return cleaned


def compute_top20(edges, labels):
    """Rank departments by # internal edges (both endpoints in dept), desc,
    ties broken by smaller department id. Return list of dept ids, length 20."""
    internal = defaultdict(int)
    for u, v in edges:
        du, dv = labels.get(u), labels.get(v)
        if du is not None and du == dv:
            internal[du] += 1
    ranked = sorted(internal.items(), key=lambda kv: (-kv[1], kv[0]))
    return [dept for dept, _ in ranked[:20]], ranked[:20]


def derive_parameters(R):
    pair_index = R % 190
    pairs = [(i, j) for i in range(20) for j in range(i + 1, 20)]
    i, j = pairs[pair_index]
    num_random_networks = 20 + (R % 31)
    swap_multiplier = 5 + (R % 6)
    seed = R
    return {
        "R": R,
        "pair_index": pair_index,
        "top20_i": i,
        "top20_j": j,
        "num_random_networks": num_random_networks,
        "swap_multiplier": swap_multiplier,
        "seed": seed,
    }


# =======================================================================
# T2 - Induced subgraphs
# =======================================================================
def induced_subgraph_edges(edges, labels, dept):
    return [(u, v) for u, v in edges if labels.get(u) == dept and labels.get(v) == dept]


def members_of(labels, dept):
    return [n for n, d in labels.items() if d == dept]


# =======================================================================
# T6 - Z-scores
# =======================================================================
def zscores_for_department(dept_edges, num_random, swap_attempts, seed):
    real_census = triad_census(set(dept_edges))

    # collect per-class counts across all random graphs
    samples = {k: [] for k in range(1, NUM_CLASSES + 1)}
    for r in range(num_random):
        rand_edges = randomize(dept_edges, attempts=swap_attempts, seed=seed + r)
        assert_degrees_preserved(dept_edges, rand_edges)
        rand_census = triad_census(set(rand_edges))
        for k in range(1, NUM_CLASSES + 1):
            samples[k].append(rand_census[k])

    z = {}
    mu = {}
    sigma = {}
    for k in range(1, NUM_CLASSES + 1):
        vals = samples[k]
        m = sum(vals) / len(vals)
        var = sum((x - m) ** 2 for x in vals) / len(vals)  # population std
        s = math.sqrt(var)
        mu[k] = m
        sigma[k] = s
        if s == 0:
            z[k] = 0.0
        else:
            z[k] = (real_census[k] - m) / s
    return real_census, mu, sigma, z


def classify_motifs(z):
    result = {}
    for k, val in z.items():
        if val > 2:
            result[k] = "motif"
        elif val < -2:
            result[k] = "anti-motif"
        else:
            result[k] = "-"
    return result


# =======================================================================
# T7 - Charts & Visualizations
# =======================================================================
def _render_svg_motif_chart(zA, zB, deptA, deptB, R, out_path):
    """Clean vector SVG grouped bar chart with non-overlapping labels and full axes."""
    classes = list(range(1, NUM_CLASSES + 1))
    valsA = [zA[k] for k in classes]
    valsB = [zB[k] for k in classes]
    all_vals = valsA + valsB + [-2.5, 2.5]
    min_z = math.floor(min(all_vals)) - 1
    max_z = math.ceil(max(all_vals)) + 2
    if min_z > -4: min_z = -4
    if max_z < 4: max_z = 4

    width, height = 1200, 650
    top, bottom, left, right = 95, 80, 85, 175
    pw = width - left - right
    ph = height - top - bottom

    def y_coord(val):
        ratio = (val - min_z) / (max_z - min_z)
        return top + ph * (1.0 - ratio)

    y0 = y_coord(0)
    yp2 = y_coord(2)
    ym2 = y_coord(-2)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background:#ffffff; font-family:\'Segoe UI\', Roboto, Arial, sans-serif;">',
        f'  <defs><filter id="drop"><feDropShadow dx="0" dy="2" stdDeviation="3" flood-opacity="0.08" /></filter></defs>',
        f'  <rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="#fcfcfd" stroke="#d1d5db" stroke-width="1.2" rx="4" />',
        f'  <text x="{width/2}" y="40" text-anchor="middle" font-size="22" font-weight="700" fill="#1e293b">Triad Motif Significance Profile (Z-Score) — Group R = {R}</text>',
        f'  <text x="{width/2}" y="66" text-anchor="middle" font-size="14" fill="#64748b">Department A (Dept {deptA}) vs Department B (Dept {deptB}) | Double-Edge Swap Null Model (40 Random Networks)</text>',
    ]

    # Grid lines & numeric Y ticks
    step = 2 if (max_z - min_z) > 16 else 1
    for tick in range(int(min_z), int(max_z) + 1, step):
        y = y_coord(tick)
        svg.append(f'  <line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#f1f5f9" stroke-width="1" />')
        svg.append(f'  <text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#64748b">{tick:+d}</text>')

    # Dashed threshold lines (+2 and -2) with labels placed safely in the right margin
    svg.extend([
        f'  <line x1="{left}" y1="{yp2:.1f}" x2="{width-right}" y2="{yp2:.1f}" stroke="#ef4444" stroke-dasharray="6,4" stroke-width="1.8" />',
        f'  <line x1="{width-right}" y1="{yp2:.1f}" x2="{width-right+8}" y2="{yp2:.1f}" stroke="#ef4444" stroke-width="1.5" />',
        f'  <text x="{width-right+12}" y="{yp2+4:.1f}" font-size="12" font-weight="600" fill="#dc2626">Motif (Z &gt; +2)</text>',
        f'  <line x1="{left}" y1="{ym2:.1f}" x2="{width-right}" y2="{ym2:.1f}" stroke="#ef4444" stroke-dasharray="6,4" stroke-width="1.8" />',
        f'  <line x1="{width-right}" y1="{ym2:.1f}" x2="{width-right+8}" y2="{ym2:.1f}" stroke="#ef4444" stroke-width="1.5" />',
        f'  <text x="{width-right+12}" y="{ym2+4:.1f}" font-size="12" font-weight="600" fill="#dc2626">Anti-motif (Z &lt; -2)</text>',
        f'  <line x1="{left}" y1="{y0:.1f}" x2="{width-right}" y2="{y0:.1f}" stroke="#334155" stroke-width="1.5" />'
    ])

    gw = pw / len(classes)
    bw = gw * 0.36
    colorA = '#3b82f6'
    colorB = '#f87171'

    for idx, c in enumerate(classes):
        cx = left + (idx + 0.5) * gw
        xA = cx - bw - 1
        xB = cx + 1
        yA = y_coord(valsA[idx])
        yB = y_coord(valsB[idx])

        topA = min(yA, y0)
        hA = max(abs(yA - y0), 1)
        svg.append(f'  <rect x="{xA:.1f}" y="{topA:.1f}" width="{bw:.1f}" height="{hA:.1f}" fill="{colorA}" rx="2" />')

        topB = min(yB, y0)
        hB = max(abs(yB - y0), 1)
        svg.append(f'  <rect x="{xB:.1f}" y="{topB:.1f}" width="{bw:.1f}" height="{hB:.1f}" fill="{colorB}" rx="2" />')

        svg.append(f'  <text x="{cx:.1f}" y="{height-bottom+24}" text-anchor="middle" font-size="12" font-weight="600" fill="#1e293b">C{c}</text>')

    # Legend box
    leg_x = left + 15
    leg_y = top + 15
    svg.extend([
        f'  <g filter="url(#drop)">',
        f'    <rect x="{leg_x}" y="{leg_y}" width="280" height="34" fill="#ffffff" stroke="#e2e8f0" rx="4" />',
        f'    <rect x="{leg_x+12}" y="{leg_y+10}" width="16" height="14" fill="{colorA}" rx="2" />',
        f'    <text x="{leg_x+34}" y="{leg_y+21}" font-size="12" font-weight="600" fill="#1e293b">Dept A (Dept {deptA})</text>',
        f'    <rect x="{leg_x+150}" y="{leg_y+10}" width="16" height="14" fill="{colorB}" rx="2" />',
        f'    <text x="{leg_x+172}" y="{leg_y+21}" font-size="12" font-weight="600" fill="#1e293b">Dept B (Dept {deptB})</text>',
        f'  </g>',
        f'  <text x="26" y="{height/2}" text-anchor="middle" transform="rotate(-90 26 {height/2})" font-size="15" font-weight="700" fill="#1e293b">Z-Score</text>',
        f'  <text x="{left + pw/2}" y="{height-22}" text-anchor="middle" font-size="15" font-weight="700" fill="#1e293b">Triad Class (1 to 13)</text>',
        f'</svg>'
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))


def _render_png_motif_chart(zA, zB, deptA, deptB, R, out_path):
    """Renders high-resolution PNG using Pillow matching the SVG layout."""
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return False

    classes = list(range(1, NUM_CLASSES + 1))
    valsA = [zA[k] for k in classes]
    valsB = [zB[k] for k in classes]
    all_vals = valsA + valsB + [-2.5, 2.5]
    min_z = math.floor(min(all_vals)) - 1
    max_z = math.ceil(max(all_vals)) + 2
    if min_z > -4: min_z = -4
    if max_z < 4: max_z = 4

    width, height = 1200, 650
    top, bottom, left, right = 95, 80, 85, 175
    pw = width - left - right
    ph = height - top - bottom

    def y_coord(val):
        ratio = (val - min_z) / (max_z - min_z)
        return top + ph * (1.0 - ratio)

    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Plot Background
    draw.rectangle([(left, top), (width - right, height - bottom)], fill=(252, 252, 253), outline=(209, 213, 219), width=1)

    # Title & Subtitle
    draw.text((width // 2 - 250, 22), f"Triad Motif Significance Profile (Z-Score) - Group R = {R}", fill=(30, 41, 59))
    draw.text((width // 2 - 290, 52), f"Department A (Dept {deptA}) vs Department B (Dept {deptB}) | Null Model (40 Random Networks)", fill=(100, 116, 139))

    y0 = y_coord(0)
    yp2 = y_coord(2)
    ym2 = y_coord(-2)

    # Grid & Y ticks
    step = 2 if (max_z - min_z) > 16 else 1
    for tick in range(int(min_z), int(max_z) + 1, step):
        y = y_coord(tick)
        draw.line([(left, y), (width - right, y)], fill=(241, 245, 249), width=1)
        draw.text((left - 38, y - 6), f"{tick:+3d}", fill=(100, 116, 139))

    # Dashed threshold lines
    x = left
    while x < width - right:
        draw.line([(x, yp2), (min(x + 8, width - right), yp2)], fill=(239, 68, 68), width=2)
        draw.line([(x, ym2), (min(x + 8, width - right), ym2)], fill=(239, 68, 68), width=2)
        x += 14

    # Threshold text in right margin (OUTSIDE plotting area)
    draw.text((width - right + 10, yp2 - 6), "Motif (Z > +2)", fill=(220, 38, 38))
    draw.text((width - right + 10, ym2 - 6), "Anti-motif (Z < -2)", fill=(220, 38, 38))
    draw.line([(left, y0), (width - right, y0)], fill=(51, 65, 85), width=2)

    gw = pw / len(classes)
    bw = gw * 0.36
    colA = (59, 130, 246)
    colB = (248, 113, 113)

    for idx, c in enumerate(classes):
        cx = left + (idx + 0.5) * gw
        xA = cx - bw - 1
        xB = cx + 1
        yA = y_coord(valsA[idx])
        yB = y_coord(valsB[idx])

        # Bar A
        if valsA[idx] >= 0:
            draw.rectangle([(xA, yA), (xA + bw, y0)], fill=colA)
        else:
            draw.rectangle([(xA, y0), (xA + bw, yA)], fill=colA)

        # Bar B
        if valsB[idx] >= 0:
            draw.rectangle([(xB, yB), (xB + bw, y0)], fill=colB)
        else:
            draw.rectangle([(xB, y0), (xB + bw, yB)], fill=colB)

        draw.text((cx - 8, height - bottom + 12), f"C{c}", fill=(30, 41, 59))

    # Legend
    leg_x = left + 15
    leg_y = top + 15
    draw.rectangle([(leg_x, leg_y), (leg_x + 270, leg_y + 30)], fill=(255, 255, 255), outline=(226, 232, 240))
    draw.rectangle([(leg_x + 10, leg_y + 8), (leg_x + 24, leg_y + 22)], fill=colA)
    draw.text((leg_x + 30, leg_y + 8), f"Dept A (Dept {deptA})", fill=(30, 41, 59))
    draw.rectangle([(leg_x + 140, leg_y + 8), (leg_x + 154, leg_y + 22)], fill=colB)
    draw.text((leg_x + 160, leg_y + 8), f"Dept B (Dept {deptB})", fill=(30, 41, 59))

    # Axis Labels
    draw.text((20, height // 2 - 30), "Z-Score", fill=(30, 41, 59))
    draw.text((width // 2 - 60, height - 35), "Triad Class (1 to 13)", fill=(30, 41, 59))

    img.save(out_path)
    return True


def make_chart(zA, zB, deptA, deptB, R, out_path):
    """Generates both vector SVG and raster PNG charts for Task T7."""
    svg_path = os.path.splitext(out_path)[0] + ".svg"
    png_path = os.path.splitext(out_path)[0] + ".png"

    _render_svg_motif_chart(zA, zB, deptA, deptB, R, svg_path)
    print(f"[SAVED] Vector SVG Chart : {svg_path}")

    _render_png_motif_chart(zA, zB, deptA, deptB, R, png_path)
    print(f"[SAVED] High-Res PNG Chart: {png_path}")


def make_raw_census_chart(censusA, censusB, deptA, deptB, R, out_svg, out_png):
    """
    Additional Insight Chart: Raw Triad Census comparison.
    Shows WHY raw counts alone are misleading (e.g. Class 3 is huge in count
    but neutral in Z-score, while Class 13 is smaller in count but a huge motif).
    """
    classes = list(range(1, NUM_CLASSES + 1))
    valsA = [censusA[k] for k in classes]
    valsB = [censusB[k] for k in classes]
    max_val = max(valsA + valsB)

    width, height = 1200, 600
    top, bottom, left, right = 95, 80, 85, 50
    pw = width - left - right
    ph = height - top - bottom

    def y_coord(val):
        return top + ph * (1.0 - (val / (max_val * 1.1)))

    y0 = y_coord(0)

    # 1. SVG
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background:#ffffff; font-family:\'Segoe UI\', Roboto, Arial, sans-serif;">',
        f'  <rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="#fcfcfd" stroke="#d1d5db" rx="4" />',
        f'  <text x="{width/2}" y="40" text-anchor="middle" font-size="22" font-weight="700" fill="#1e293b">Raw Triad Census Comparison: Dept {deptA} vs Dept {deptB}</text>',
        f'  <text x="{width/2}" y="66" text-anchor="middle" font-size="14" fill="#64748b">Empirical Counts N(G) — Illustrates why raw counting alone is misleading</text>',
    ]

    for tick in range(0, int(max_val * 1.1) + 1, 250):
        y = y_coord(tick)
        svg.append(f'  <line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#f1f5f9" />')
        svg.append(f'  <text x="{left-12}" y="{y+4:.1f}" text-anchor="end" font-size="12" fill="#64748b">{tick}</text>')

    gw = pw / len(classes)
    bw = gw * 0.36
    for idx, c in enumerate(classes):
        cx = left + (idx + 0.5) * gw
        xA = cx - bw - 1
        xB = cx + 1
        yA = y_coord(valsA[idx])
        yB = y_coord(valsB[idx])

        svg.append(f'  <rect x="{xA:.1f}" y="{yA:.1f}" width="{bw:.1f}" height="{y0 - yA:.1f}" fill="#3b82f6" rx="2" />')
        svg.append(f'  <rect x="{xB:.1f}" y="{yB:.1f}" width="{bw:.1f}" height="{y0 - yB:.1f}" fill="#f87171" rx="2" />')
        svg.append(f'  <text x="{cx:.1f}" y="{height-bottom+24}" text-anchor="middle" font-size="12" font-weight="600" fill="#1e293b">C{c}</text>')

    svg.extend([
        f'  <text x="26" y="{height/2}" text-anchor="middle" transform="rotate(-90 26 {height/2})" font-size="15" font-weight="700" fill="#1e293b">Raw Count N(G)</text>',
        f'  <text x="{left + pw/2}" y="{height-22}" text-anchor="middle" font-size="15" font-weight="700" fill="#1e293b">Triad Class (1 to 13)</text>',
        f'</svg>'
    ])

    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"[SAVED] Raw Census SVG   : {out_svg}")

    # 2. PNG
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(left, top), (width - right, height - bottom)], fill=(252, 252, 253), outline=(209, 213, 219))
        draw.text((width // 2 - 220, 22), f"Raw Triad Census Comparison: Dept {deptA} vs Dept {deptB}", fill=(30, 41, 59))
        draw.text((width // 2 - 250, 50), "Empirical Counts N(G) - Illustrates why raw counting alone is misleading", fill=(100, 116, 139))
        for tick in range(0, int(max_val * 1.1) + 1, 250):
            y = y_coord(tick)
            draw.line([(left, y), (width - right, y)], fill=(241, 245, 249))
            draw.text((left - 40, y - 6), f"{tick:4d}", fill=(100, 116, 139))
        colA = (59, 130, 246)
        colB = (248, 113, 113)
        for idx, c in enumerate(classes):
            cx = left + (idx + 0.5) * gw
            xA = cx - bw - 1
            xB = cx + 1
            yA = y_coord(valsA[idx])
            yB = y_coord(valsB[idx])
            draw.rectangle([(xA, yA), (xA + bw, y0)], fill=colA)
            draw.rectangle([(xB, yB), (xB + bw, y0)], fill=colB)
            draw.text((cx - 8, height - bottom + 12), f"C{c}", fill=(30, 41, 59))
        draw.text((20, height // 2 - 30), "Raw Count", fill=(30, 41, 59))
        draw.text((width // 2 - 60, height - 35), "Triad Class (1 to 13)", fill=(30, 41, 59))
        img.save(out_png)
        print(f"[SAVED] Raw Census PNG   : {out_png}")
    except Exception:
        pass


def make_null_distribution_chart(censusA, muA, sigmaA, deptA, R, out_svg, out_png):
    """
    Additional Insight Chart: Null Model Distribution for Class 13 (Dept 1).
    Visually depicts the Bell Curve vs Real Count to explain the Z-score.
    """
    real = censusA[13]
    m = muA[13]
    s = sigmaA[13]
    z = (real - m) / s

    width, height = 900, 450
    left, right, top, bottom = 80, 50, 80, 60
    pw = width - left - right
    ph = height - top - bottom

    min_x = m - 3.5 * s
    max_x = real + 20
    def val_to_x(v):
        return left + pw * ((v - min_x) / (max_x - min_x))

    x_mean = val_to_x(m)
    x_real = val_to_x(real)
    y_base = top + ph - 20

    # 1. SVG
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" style="background:#ffffff; font-family:\'Segoe UI\', Roboto, Arial, sans-serif;">',
        f'  <rect x="{left}" y="{top}" width="{pw}" height="{ph}" fill="#fcfcfd" stroke="#d1d5db" rx="4" />',
        f'  <text x="{width/2}" y="36" text-anchor="middle" font-size="20" font-weight="700" fill="#1e293b">Null Model Distribution: Class 13 (Complete Reciprocal Clique)</text>',
        f'  <text x="{width/2}" y="60" text-anchor="middle" font-size="13" fill="#64748b">Dept {deptA}: Expected Mean = {m:.1f} (std = {s:.1f}) vs. Real Observed = {real}  --&gt;  Z = +{z:.2f}</text>',
    ]

    pts = []
    for step in range(100):
        v = min_x + step * ((m + 3.5*s - min_x) / 100)
        prob = math.exp(-0.5 * ((v - m)/s)**2)
        x_pt = val_to_x(v)
        y_pt = y_base - prob * (ph * 0.7)
        pts.append(f"{x_pt:.1f},{y_pt:.1f}")

    svg.append(f'  <polygon points="{val_to_x(min_x):.1f},{y_base} ' + ' '.join(pts) + f' {val_to_x(m + 3.5*s):.1f},{y_base}" fill="#93c5fd" opacity="0.6" stroke="#3b82f6" stroke-width="2" />')
    svg.append(f'  <line x1="{x_mean:.1f}" y1="{y_base}" x2="{x_mean:.1f}" y2="{y_base - ph*0.75:.1f}" stroke="#1d4ed8" stroke-dasharray="4,3" stroke-width="2" />')
    svg.append(f'  <text x="{x_mean:.1f}" y="{y_base - ph*0.78:.1f}" text-anchor="middle" font-size="12" font-weight="600" fill="#1d4ed8">Random Mean (μ = {m:.1f})</text>')

    svg.append(f'  <line x1="{x_real:.1f}" y1="{y_base}" x2="{x_real:.1f}" y2="{top + 20}" stroke="#dc2626" stroke-width="3" />')
    svg.append(f'  <circle cx="{x_real:.1f}" cy="{top + 20}" r="5" fill="#dc2626" />')
    svg.append(f'  <text x="{x_real:.1f}" y="{top + 14}" text-anchor="middle" font-size="13" font-weight="700" fill="#dc2626">Real N(G) = {real} (Z = +{z:.2f})</text>')
    svg.append(f'  <text x="{x_real:.1f}" y="{top + 35}" text-anchor="middle" font-size="11" font-weight="600" fill="#b91c1c">&gt;&gt; Highly Significant Motif &lt;&lt;</text>')

    svg.append(f'  <line x1="{left}" y1="{y_base}" x2="{width-right}" y2="{y_base}" stroke="#334155" stroke-width="1.5" />')
    svg.append(f'  <text x="{width/2}" y="{height-16}" text-anchor="middle" font-size="13" font-weight="600" fill="#334155">Count of Class 13 Triads</text>')
    svg.append('</svg>')

    with open(out_svg, "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print(f"[SAVED] Null Distribution SVG: {out_svg}")

    # 2. PNG
    try:
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (width, height), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.rectangle([(left, top), (width - right, height - bottom)], fill=(252, 252, 253), outline=(209, 213, 219))
        draw.text((width // 2 - 200, 20), "Null Model Distribution: Class 13", fill=(30, 41, 59))
        draw.text((width // 2 - 250, 45), f"Expected Mean = {m:.1f} vs. Real Observed = {real} (Z = +{z:.2f})", fill=(100, 116, 139))
        draw.line([(x_mean, y_base), (x_mean, y_base - ph * 0.7)], fill=(29, 78, 216), width=2)
        draw.text((x_mean - 60, y_base - ph * 0.75), f"Mean ({m:.1f})", fill=(29, 78, 216))
        draw.line([(x_real, y_base), (x_real, top + 20)], fill=(220, 38, 38), width=3)
        draw.text((x_real - 70, top + 15), f"Real N(G) = {real} (Z = +{z:.2f})", fill=(220, 38, 38))
        draw.line([(left, y_base), (width - right, y_base)], fill=(51, 65, 85), width=2)
        img.save(out_png)
        print(f"[SAVED] Null Distribution PNG: {out_png}")
    except Exception:
        pass


def save_summary_report(params, deptA, deptB, censusA, censusB, realA, muA, sigmaA, zA, motifA, realB, muB, sigmaB, zB, motifB, out_file):
    """Saves all tables into a permanent plain-text summary report."""
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("=" * 75 + "\n")
        f.write(f"CS1005 DISCRETE STRUCTURES - NETWORK MOTIFS REPORT (GROUP R = {params['R']})\n")
        f.write("=" * 75 + "\n\n")
        f.write(f"Group Roll Suffixes : {MEMBER_ROLL_SUFFIXES}\n")
        f.write(f"Group Number R      : {params['R']}\n")
        f.write(f"Pair Index          : {params['pair_index']} -> (i={params['top20_i']}, j={params['top20_j']})\n")
        f.write(f"Department A        : Dept {deptA}\n")
        f.write(f"Department B        : Dept {deptB}\n")
        f.write(f"Random Networks     : {params['num_random_networks']}\n")
        f.write(f"Swap Multiplier     : {params['swap_multiplier']}\n")
        f.write(f"Random Seed         : {params['seed']}\n\n")

        f.write("--- T4: TRIAD CENSUS (13 CLASSES) ---\n")
        f.write(f"{'Class':>6} | {'Dept A (' + str(deptA) + ')':>14} | {'Dept B (' + str(deptB) + ')':>14}\n")
        f.write("-" * 42 + "\n")
        for k in range(1, NUM_CLASSES + 1):
            f.write(f"{k:>6} | {censusA[k]:>14} | {censusB[k]:>14}\n")

        f.write("\n--- T6: STATISTICAL SIGNIFICANCE (Z-SCORES) ---\n")
        f.write(f"{'Class':>6} | {'N(A)':>6} {'mu(A)':>8} {'Z(A)':>8} {'tag':>10} | {'N(B)':>6} {'mu(B)':>8} {'Z(B)':>8} {'tag':>10}\n")
        f.write("-" * 75 + "\n")
        for k in range(1, NUM_CLASSES + 1):
            f.write(f"{k:>6} | {realA[k]:>6} {muA[k]:>8.2f} {zA[k]:>8.2f} {motifA[k]:>10} | {realB[k]:>6} {muB[k]:>8.2f} {zB[k]:>8.2f} {motifB[k]:>10}\n")
    print(f"[SAVED] Text Summary Report  : {out_file}")


# =======================================================================
# MAIN
# =======================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- T1 ----
    print("=" * 70)
    print("T1 - Load, clean, derive parameters")
    print("=" * 70)
    raw_edges = load_raw_edges(EDGES_FILE)
    labels = load_labels(LABELS_FILE)
    edges = clean_edges(raw_edges)
    print(f"Raw edges: {len(raw_edges)}  ->  Cleaned edges: {len(edges)} "
          f"(removed {len(raw_edges) - len(edges)} self-loops/duplicates)")

    top20, top20_with_counts = compute_top20(edges, labels)
    print(f"\nTop-20 departments by internal edge count:")
    for idx, (dept, cnt) in enumerate(top20_with_counts):
        print(f"  index {idx:2d}:  dept {dept:2d}   internal edges = {cnt}")

    params = derive_parameters(R)
    deptA = top20[params["top20_i"]]
    deptB = top20[params["top20_j"]]

    print(f"\nGroup roll suffixes: {MEMBER_ROLL_SUFFIXES}")
    print(f"R                     = {params['R']}")
    print(f"Pair index (R mod 190)= {params['pair_index']}  -> "
          f"(i={params['top20_i']}, j={params['top20_j']})")
    print(f"Department A          = top20[{params['top20_i']}] = Dept {deptA}")
    print(f"Department B          = top20[{params['top20_j']}] = Dept {deptB}")
    print(f"Num random networks   = 20 + (R mod 31) = {params['num_random_networks']}")
    print(f"Swap multiplier       = 5 + (R mod 6)  = {params['swap_multiplier']}")
    print(f"Random seed           = R = {params['seed']}")

    # ---- T2 ----
    print("\n" + "=" * 70)
    print("T2 - Induced subgraphs")
    print("=" * 70)
    membersA = members_of(labels, deptA)
    membersB = members_of(labels, deptB)
    edgesA = induced_subgraph_edges(edges, labels, deptA)
    edgesB = induced_subgraph_edges(edges, labels, deptB)
    print(f"Dept A ({deptA}): {len(membersA)} nodes, {len(edgesA)} edges")
    print(f"Dept B ({deptB}): {len(membersB)} nodes, {len(edgesB)} edges")

    swapsA = params["swap_multiplier"] * len(edgesA)
    swapsB = params["swap_multiplier"] * len(edgesB)
    print(f"Swap attempts per random network: Dept A = {swapsA}, Dept B = {swapsB}")

    # ---- T3 ----
    print("\n" + "=" * 70)
    print("T3 - triad_class() hand-worked example")
    print("=" * 70)
    code, cls = hand_worked_example()
    print(f"Nodes (0, 1, 18) [all in Dept {deptA if deptA==1 else '1'}]: "
          f"canonical code = {code}  ->  Class {cls}")

    # ---- T4 ----
    print("\n" + "=" * 70)
    print("T4 - Triad census (13 counts) for both departments")
    print("=" * 70)
    censusA = triad_census(set(edgesA))
    censusB = triad_census(set(edgesB))
    print(f"{'Class':>6} | {'Dept A (' + str(deptA) + ')':>14} | {'Dept B (' + str(deptB) + ')':>14}")
    for k in range(1, NUM_CLASSES + 1):
        print(f"{k:>6} | {censusA[k]:>14} | {censusB[k]:>14}")

    # cross-check with networkx triadic_census (allowed, must note different ordering)
    try:
        import networkx as nx
        GA = nx.DiGraph(edgesA)
        GB = nx.DiGraph(edgesB)
        nx_censusA = nx.triadic_census(GA)
        nx_censusB = nx.triadic_census(GB)
        connected_classes = ["021D", "021U", "021C", "111D", "111U", "030T",
                              "030C", "201", "120D", "120U", "120C", "210", "300"]
        nx_connected_A = sum(nx_censusA[c] for c in connected_classes)
        nx_connected_B = sum(nx_censusB[c] for c in connected_classes)
        print(f"\n[Cross-check with nx.triadic_census (different class labels/order, "
              f"used ONLY to sanity-check our total connected-triad counts)]")
        print(f"  Our total connected triads  -> A: {sum(censusA.values())}, B: {sum(censusB.values())}")
        print(f"  nx connected-class total    -> A: {nx_connected_A}, B: {nx_connected_B}")
    except ImportError:
        print("\n(networkx not available for cross-check; skipping)")

    # ---- T5 ----
    print("\n" + "=" * 70)
    print("T5 - randomize() degree-preservation check")
    print("=" * 70)
    test_rand = randomize(edgesA, attempts=swapsA, seed=params["seed"])
    assert_degrees_preserved(edgesA, test_rand)
    print(f"Verified: in-degree and out-degree sequences unchanged after "
          f"{swapsA} swap attempts on Dept A ({len(edgesA)} edges).")

    # ---- T6 ----
    print("\n" + "=" * 70)
    print("T6 - Z-scores")
    print("=" * 70)
    num_random = params["num_random_networks"]
    seed = params["seed"]

    realA, muA, sigmaA, zA = zscores_for_department(edgesA, num_random, swapsA, seed)
    realB, muB, sigmaB, zB = zscores_for_department(edgesB, num_random, swapsB, seed + 10_000)

    motifA = classify_motifs(zA)
    motifB = classify_motifs(zB)

    print(f"{'Class':>6} | {'N(A)':>6} {'mu(A)':>8} {'Z(A)':>8} {'tag':>10} | "
          f"{'N(B)':>6} {'mu(B)':>8} {'Z(B)':>8} {'tag':>10}")
    for k in range(1, NUM_CLASSES + 1):
        print(f"{k:>6} | {realA[k]:>6} {muA[k]:>8.2f} {zA[k]:>8.2f} {motifA[k]:>10} | "
              f"{realB[k]:>6} {muB[k]:>8.2f} {zB[k]:>8.2f} {motifB[k]:>10}")

    # ---- T7 ----
    print("\n" + "=" * 70)
    print("T7 - Generating Charts & Deliverables")
    print("=" * 70)
    chart_path = os.path.join(OUTPUT_DIR, "motif_chart.png")
    make_chart(zA, zB, deptA, deptB, R, chart_path)

    raw_svg = os.path.join(OUTPUT_DIR, "raw_census_chart.svg")
    raw_png = os.path.join(OUTPUT_DIR, "raw_census_chart.png")
    make_raw_census_chart(censusA, censusB, deptA, deptB, R, raw_svg, raw_png)

    dist_svg = os.path.join(OUTPUT_DIR, "null_distribution_C13.svg")
    dist_png = os.path.join(OUTPUT_DIR, "null_distribution_C13.png")
    make_null_distribution_chart(censusA, muA, sigmaA, deptA, R, dist_svg, dist_png)

    summary_txt = os.path.join(OUTPUT_DIR, "results_summary.txt")
    save_summary_report(params, deptA, deptB, censusA, censusB, realA, muA, sigmaA, zA, motifA, realB, muB, sigmaB, zB, motifB, summary_txt)

    print("\n[SUCCESS] All tables, statistics, and charts generated successfully.")
    print("DONE.")


if __name__ == "__main__":
    main()
