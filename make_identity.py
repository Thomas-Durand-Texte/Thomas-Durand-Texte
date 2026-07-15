"""GitHub profile identity for Thomas Durand-Texte — avatar + README banner.

Same visual language as the project cards (dark, amber accent). A monogram
(TDT) plus a composable signal->deep-learning->vision motif built from four
segments: waveform (acoustics), spectrogram bars (signal processing), a neural
net (deep learning), and a pixel/tensor grid (computer vision).

  avatar : wave -> NN -> grid          (3 icons, reads at small size)
  banner : wave -> bars -> NN -> grid  (full pipeline; wide format has room)

Writes avatar_{theme}.png (1000x1000) and banner_{theme}.png (1200x300).
"""
import os, argparse, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle, FancyArrow, Polygon
from matplotlib.lines import Line2D

SP = os.path.dirname(__file__)

DARK = dict(bg='#0B0F14', ink='#EAF0F6', muted='#8592A0', accent='#F2A65A',
            ring='#243040', edge='#5A4633', cell='#F2A65A')
LIGHT = dict(bg='#FBFCFD', ink='#1B2230', muted='#5A6672', accent='#D9772E',
             ring='#D3DAE1', edge='#D6B48C', cell='#D9772E')


def _rng():
    return np.random.default_rng(7)


# ---- motif segments: each fills [x0,x1] x (yc +/- h/2) ----
def seg_wave(ov, x0, x1, yc, h, pal, alpha, lw):
    t = np.linspace(0, 1, 240)
    y = (np.sin(2*np.pi*3*t)*0.6 + np.sin(2*np.pi*7*t+1)*0.3) * np.hanning(240)
    ov.add_line(Line2D(x0 + t*(x1-x0), yc + y*h*0.5, color=pal['accent'],
                       lw=2.2*lw, alpha=alpha, solid_capstyle='round'))


def seg_bars(ov, x0, x1, yc, h, pal, alpha, lw):
    n = 9
    hs = (0.35 + 0.65*_rng().random(n)) * np.exp(-(np.linspace(-1, 1, n))**2 / 0.6)
    bw = (x1-x0)/n * 0.6
    for i, hh in enumerate(hs):
        cx = x0 + (i+0.5)*(x1-x0)/n
        ov.add_patch(FancyBboxPatch((cx-bw/2, yc-hh*h*0.5), bw, hh*h,
                     boxstyle=f'round,pad=0,rounding_size={bw*0.4}',
                     linewidth=0, facecolor=pal['accent'], alpha=alpha))


def seg_nn(ov, x0, x1, yc, h, pal, alpha, lw):
    cols = [3, 4, 3]
    xs = np.linspace(x0, x1, len(cols))
    pts = [[(cx, yy) for yy in np.linspace(yc-h*0.5, yc+h*0.5, nc)]
           for cx, nc in zip(xs, cols)]
    for a, b in zip(pts, pts[1:]):
        for (ax, ay) in a:
            for (bx, by) in b:
                ov.add_line(Line2D([ax, bx], [ay, by], color=pal['edge'],
                                   lw=0.9*lw, alpha=alpha, zorder=2))
    r = min(h*0.07, (x1-x0)/12) + 2
    for col in pts:
        for (px, py) in col:
            ov.add_patch(Circle((px, py), r, facecolor=pal['accent'], alpha=alpha, zorder=3))


def seg_grid(ov, x0, x1, yc, h, pal, alpha, lw):
    gsz = min(x1-x0, h); g = 5
    vals = _rng().random((g, g)); cell = gsz/g
    for r in range(g):
        for c in range(g):
            ov.add_patch(Rectangle((x0 + c*cell, yc - gsz/2 + r*cell),
                         cell*0.86, cell*0.86, linewidth=0,
                         facecolor=pal['cell'], alpha=alpha*(0.25 + 0.75*vals[r, c])))


# ---- vision-icon segments (pick one to represent computer vision) ----
def _box(x0, x1, yc, h):
    s = min(x1-x0, h)
    return (x0+x1)/2, yc, s


def _u(cx, cy, s, ux, uy):          # unit-box [0,1] -> absolute
    return cx + (ux-0.5)*s, cy + (uy-0.5)*s


def _clip_line(p, d, c, R):         # segment of line p+t*d inside circle(c,R)
    px, py = p[0]-c[0], p[1]-c[1]; dx, dy = d
    aa = dx*dx+dy*dy; bb = 2*(px*dx+py*dy); cc = px*px+py*py-R*R
    disc = bb*bb-4*aa*cc
    if disc < 0:
        return None
    t1 = (-bb-disc**0.5)/(2*aa); t2 = (-bb+disc**0.5)/(2*aa)
    return (p[0]+t1*dx, p[1]+t1*dy), (p[0]+t2*dx, p[1]+t2*dy)


def _eye(ov, ex, ey, w, pal, alpha, L, iris=False, pupil=0.15):
    """Almond (lens) eye: two circular arcs meeting at pointed corners, + pupil."""
    a = w*0.5; b = w*0.30                      # half-width, apex height
    k = (b*b - a*a)/(2*b); R = (a*a + b*b)/(2*b)
    xs = np.linspace(-a, a, 48)
    arc = k + np.sqrt(np.clip(R*R - xs*xs, 0, None))
    up = [(ex+x, ey+y) for x, y in zip(xs, arc)]
    lo = [(ex+x, ey-y) for x, y in zip(xs[::-1], arc[::-1])]
    ov.add_patch(Polygon(up+lo, closed=True, fill=False, ec=pal['accent'],
                         lw=L, alpha=alpha, joinstyle='round'))
    if iris:
        ov.add_patch(Circle((ex, ey), w*0.19, fill=False, ec=pal['accent'], lw=L, alpha=alpha))
    ov.add_patch(Circle((ex, ey), w*pupil, facecolor=pal['accent'], ec='none', alpha=alpha))


def seg_eye(ov, x0, x1, yc, h, pal, alpha, lw):
    cx, cy, s = _box(x0, x1, yc, h)
    _eye(ov, cx, cy, 0.9*s, pal, alpha, 2.6*lw, iris=True, pupil=0.09)


def seg_image(ov, x0, x1, yc, h, pal, alpha, lw):
    cx, cy, s = _box(x0, x1, yc, h); L = 2.6*lw
    p = lambda ux, uy: _u(cx, cy, s, ux, uy)
    x, y = p(0.13, 0.16)
    ov.add_patch(FancyBboxPatch((x, y), 0.74*s, 0.68*s,
                 boxstyle=f'round,pad=0,rounding_size={0.08*s}', fill=False,
                 ec=pal['accent'], lw=L, alpha=alpha))
    sx, sy = p(0.34, 0.63)
    ov.add_patch(Circle((sx, sy), 0.075*s, facecolor=pal['accent'], ec='none', alpha=alpha))
    pts = [p(0.17, 0.24), p(0.40, 0.52), p(0.55, 0.40), p(0.83, 0.60)]
    ov.add_line(Line2D([q[0] for q in pts], [q[1] for q in pts], color=pal['accent'],
                       lw=L, alpha=alpha, solid_joinstyle='round'))


def seg_aperture(ov, x0, x1, yc, h, pal, alpha, lw):
    """Camera iris: pinwheel blades closing to a central hexagonal hole."""
    cx, cy, s = _box(x0, x1, yc, h); L = 2.6*lw; c = (cx, cy)
    R = 0.40*s; r = 0.13*s          # smaller r -> more closed iris
    ov.add_patch(Circle(c, R, fill=False, ec=pal['accent'], lw=L, alpha=alpha))
    H = [(cx+r*np.cos(k*np.pi/3), cy+r*np.sin(k*np.pi/3)) for k in range(6)]
    for k in range(6):
        p, q = H[k], H[(k+1) % 6]
        d = (q[0]-p[0], q[1]-p[1]); n = np.hypot(*d); d = (d[0]/n, d[1]/n)
        seg = _clip_line(p, d, c, R)
        if not seg:
            continue
        far = seg[0] if np.hypot(seg[0][0]-q[0], seg[0][1]-q[1]) > \
            np.hypot(seg[1][0]-q[0], seg[1][1]-q[1]) else seg[1]
        ov.add_line(Line2D([far[0], q[0]], [far[1], q[1]], color=pal['accent'],
                           lw=L, alpha=alpha, solid_capstyle='round'))


def seg_bbox(ov, x0, x1, yc, h, pal, alpha, lw):
    cx, cy, s = _box(x0, x1, yc, h); L = 2.6*lw
    p = lambda ux, uy: _u(cx, cy, s, ux, uy)
    x, y = p(0.12, 0.14)
    ov.add_patch(Rectangle((x, y), 0.76*s, 0.72*s, fill=False, ec=pal['muted'], lw=L*0.8, alpha=alpha))
    for (ax0, ay0, ax1, ay1) in [(0.20, 0.30, 0.52, 0.70), (0.56, 0.22, 0.84, 0.54)]:
        for (ux, uy, dx, dy) in [(ax0, ay0, 1, 1), (ax1, ay0, -1, 1), (ax0, ay1, 1, -1), (ax1, ay1, -1, -1)]:
            px, py = p(ux, uy)
            ov.add_line(Line2D([px, px+dx*0.07*s], [py, py], color=pal['accent'], lw=L, alpha=alpha))
            ov.add_line(Line2D([px, px], [py, py+dy*0.07*s], color=pal['accent'], lw=L, alpha=alpha))


def seg_field(ov, x0, x1, yc, h, pal, alpha, lw):
    cx, cy, s = _box(x0, x1, yc, h)
    for ux in np.linspace(0.22, 0.78, 3):
        for uy in np.linspace(0.22, 0.78, 3):
            dx, dy = -(uy-0.5), (ux-0.5)
            n = np.hypot(dx, dy)
            if n < 1e-6:
                continue
            dx, dy = dx/n*0.13*s, dy/n*0.13*s
            px, py = _u(cx, cy, s, ux, uy)
            ov.add_patch(FancyArrow(px-dx/2, py-dy/2, dx, dy, width=0.008*s,
                         head_width=0.05*s, head_length=0.05*s, length_includes_head=True,
                         facecolor=pal['accent'], ec='none', alpha=alpha))


def seg_imageye(ov, x0, x1, yc, h, pal, alpha, lw):
    """The image-frame icon (border + sun + landscape) with an eye in the top-right."""
    cx, cy, s = _box(x0, x1, yc, h); L = 2.6*lw
    p = lambda ux, uy: _u(cx, cy, s, ux, uy)
    x, y = p(0.13, 0.16)
    ov.add_patch(FancyBboxPatch((x, y), 0.74*s, 0.68*s,
                 boxstyle=f'round,pad=0,rounding_size={0.07*s}', fill=False,
                 ec=pal['accent'], lw=L, alpha=alpha))
    sx, sy = p(0.30, 0.62)              # sun
    ov.add_patch(Circle((sx, sy), 0.07*s, facecolor=pal['accent'], ec='none', alpha=alpha))
    pts = [p(0.17, 0.24), p(0.40, 0.52), p(0.55, 0.40), p(0.83, 0.58)]   # landscape
    ov.add_line(Line2D([q[0] for q in pts], [q[1] for q in pts], color=pal['accent'],
                       lw=L, alpha=alpha, solid_joinstyle='round'))
    ex, ey = p(0.71, 0.73)                 # eye badge, top-right
    _eye(ov, ex, ey, 0.28*s, pal, alpha, L*0.9, pupil=0.16)


VISIONS = {'grid': seg_grid, 'eye': seg_eye, 'image': seg_image, 'imageye': seg_imageye,
           'aperture': seg_aperture, 'bbox': seg_bbox, 'field': seg_field}


def motif(ov, x0, x1, yc, h, pal, parts, alpha=1.0, lw=1.0):
    n = len(parts); gap = (x1-x0)*0.05
    seg = ((x1-x0) - gap*(n-1)) / n
    for i, fn in enumerate(parts):
        sx = x0 + i*(seg+gap)
        fn(ov, sx, sx+seg, yc, h, pal, alpha, lw)


AVATAR_PARTS = [seg_wave, seg_bars, seg_nn, seg_imageye]
BANNER_PARTS = [seg_wave, seg_bars, seg_nn, seg_imageye]


def build_avatar(pal, out):
    S = 1000
    fig = plt.figure(figsize=(S/100, S/100), dpi=100); fig.patch.set_facecolor(pal['bg'])
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, S); ov.set_ylim(0, S); ov.axis('off')
    ov.add_patch(Rectangle((0, 0), S, S, facecolor=pal['bg'], zorder=0))
    ov.add_patch(Circle((S/2, S/2), 470, fill=False, ec=pal['ring'], lw=3, zorder=1))
    ov.text(S/2, S*0.52, 'TDT', ha='center', va='center', color=pal['ink'],
            fontsize=250, fontweight='bold', zorder=3)
    ew, gap = 150, 30
    tw = len(AVATAR_PARTS)*ew + (len(AVATAR_PARTS)-1)*gap
    motif(ov, S/2 - tw/2, S/2 + tw/2, S*0.30, 150, pal, AVATAR_PARTS, lw=1.6)
    fig.savefig(out, dpi=100, facecolor=pal['bg']); plt.close(fig)
    print('saved', out)


def build_banner(pal, out):
    W, H = 1200, 300
    fig = plt.figure(figsize=(W/100, H/100), dpi=100); fig.patch.set_facecolor(pal['bg'])
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(0, H); ov.axis('off')
    ov.add_patch(Rectangle((0, 0), W, H, facecolor=pal['bg'], zorder=0))
    ov.text(64, 182, 'Thomas Durand-Texte', ha='left', va='center',
            color=pal['ink'], fontsize=40, fontweight='bold')
    ov.add_line(Line2D([66, 216], [150, 150], color=pal['accent'], lw=3, solid_capstyle='round'))
    ov.text(66, 116, 'Senior Applied AI Scientist', ha='left', va='center',
            color=pal['ink'], fontsize=18, alpha=0.9)
    ov.text(66, 84, 'Acoustics  ·  Signal Processing  ·  Computer Vision',
            ha='left', va='center', color=pal['muted'], fontsize=16)
    mx0, mx1 = 752, 1160                 # square icon boxes: h == segment width
    segw = (mx1-mx0) * 0.85 / len(BANNER_PARTS)
    motif(ov, mx0, mx1, H/2, segw, pal, BANNER_PARTS, alpha=0.95, lw=1.1)
    ov.text((mx0+mx1)/2, 232, 'signal  →  vision', ha='center', va='center',
            color=pal['muted'], fontsize=11, style='italic', alpha=0.8)
    fig.savefig(out, dpi=100, facecolor=pal['bg']); plt.close(fig)
    print('saved', out)


def build_social(pal, out):
    """1280x640 (GitHub 2:1 social-preview) poster: name + tagline + motif."""
    W, H = 1280, 640
    fig = plt.figure(figsize=(W/100, H/100), dpi=100); fig.patch.set_facecolor(pal['bg'])
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, W); ov.set_ylim(0, H); ov.axis('off')
    ov.add_patch(Rectangle((0, 0), W, H, facecolor=pal['bg'], zorder=0))
    ov.text(W/2, H*0.605, 'Thomas Durand-Texte', ha='center', va='center',
            color=pal['ink'], fontsize=58, fontweight='bold')
    ov.add_line(Line2D([W/2-150, W/2+150], [H*0.525, H*0.525], color=pal['accent'],
                       lw=4, solid_capstyle='round'))
    ov.text(W/2, H*0.475, 'Senior Applied AI Scientist', ha='center', va='center',
            color=pal['ink'], fontsize=26, alpha=0.9)
    ov.text(W/2, H*0.415, 'Acoustics   ·   Signal Processing   ·   Computer Vision',
            ha='center', va='center', color=pal['muted'], fontsize=22)
    Wm = 620; x0, x1 = W/2 - Wm/2, W/2 + Wm/2
    segw = Wm * 0.85 / len(BANNER_PARTS)          # square icon boxes
    motif(ov, x0, x1, H*0.235, segw, pal, BANNER_PARTS, lw=1.5)
    fig.savefig(out, dpi=100, facecolor=pal['bg']); plt.close(fig)
    print('saved', out)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--outdir', default=SP)
    a = ap.parse_args()
    for name, pal in (('dark', DARK), ('light', LIGHT)):
        build_avatar(pal, os.path.join(a.outdir, f'avatar_{name}.png'))
        build_banner(pal, os.path.join(a.outdir, f'banner_{name}.png'))
        build_social(pal, os.path.join(a.outdir, f'social_{name}.png'))
