#!/usr/bin/env python3
"""Reproduce the introduction's figures and tables from the native laboratory runner.

    cargo build --release --manifest-path core/Cargo.toml --example lab_dump
    python3 scripts/render_intro_figures.py --runs /path/to/cache

Every request is listed in REQUESTS and executed with core/target/release/examples/lab_dump
when its JSON is missing from the cache directory (the 64^3 runs take minutes each). The
script then writes chapters/figures/intro-runs.json (compact summaries and fine-binned spectra,
without the large fields), the figures intro-bao.svg/png and intro-estimators.svg/png, and the
two Markdown table includes chapters/_intro-order-table.qmd and chapters/_intro-bao-table.qmd.
Without --runs it re-renders the tables and the BAO figure from the committed summary file.
"""
import argparse, json, math, subprocess, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / 'chapters' / 'figures'
SUMMARY = FIG / 'intro-runs.json'
BIN = ROOT / 'core' / 'target' / 'release' / 'examples' / 'lab_dump'
BASE = dict(omegaM=.315, omegaB=.049, h=.674, ns=.965, sigma8=.811, boxSize=300, initialSmoothing=14, smoothing=.03,
            epoch='scaleFactor', scaleFactor=1.0, fraction=.5, order=4, seed=7, ic='gaussian', initialCutoff=7, transfer='smooth')
REQUESTS = {}
for order in (1, 2, 3, 4):
    REQUESTS[f'lcdm3d_o{order}'] = dict(BASE, dim=3, n=64, forceMesh=64, mesh=64, steps=128, order=order)
for t in ('smooth', 'bao'):
    REQUESTS[f'bao2d_{t}'] = dict(BASE, dim=2, n=256, forceMesh=256, mesh=256, steps=64, boxSize=1000, initialCutoff=31, transfer=t, smoothing=0.0)
    REQUESTS[f'bao2d_{t}_o2'] = dict(BASE, dim=2, n=256, forceMesh=256, mesh=256, steps=64, boxSize=1000, initialCutoff=31, transfer=t, smoothing=0.0, order=2)
    REQUESTS[f'bao3d_{t}'] = dict(BASE, dim=3, n=64, forceMesh=64, mesh=64, steps=64, boxSize=600, initialCutoff=7, transfer=t, smoothing=0.0)
REQUESTS['estimators2d'] = dict(BASE, dim=2, n=64, forceMesh=64, mesh=256, steps=64, smoothing=0.0, initialCutoff=7)

# ---- Eisenstein & Hu (1998) transfer functions, duplicated here only to evaluate the linear input on fine
# ---- wavenumber bins; the script asserts agreement with the core's own values at the exported bins.
TCMB = 2.7255 / 2.7
def transfer_smooth(k, om, ob, h):
    k = np.asarray(k, float); omhh, obhh, fb = om * h * h, ob * h * h, ob / om
    sound = 44.5 * math.log(9.83 / omhh) / math.sqrt(1 + 10 * obhh ** .75)
    alpha = 1 - .328 * math.log(431 * omhh) * fb + .38 * math.log(22.3 * omhh) * fb * fb
    gamma = om * h * (alpha + (1 - alpha) / (1 + (.43 * k * h * sound) ** 4))
    q = k * TCMB ** 2 / gamma; l = np.log(2 * math.e + 1.8 * q)
    return np.where(k == 0, 1., l / (l + (14.2 + 731 / (1 + 62.5 * q)) * q * q))
def transfer_bao(k, om, ob, h):
    k = np.asarray(k, float); omhh, obhh, fb = om * h * h, ob * h * h, ob / om; fc = 1 - fb; kk = k * h
    z_eq = 2.5e4 * omhh / TCMB ** 4; k_eq = 7.46e-2 * omhh / TCMB ** 2
    b1 = .313 * omhh ** -.419 * (1 + .607 * omhh ** .674); b2 = .238 * omhh ** .223
    z_d = 1291 * omhh ** .251 / (1 + .659 * omhh ** .828) * (1 + b1 * obhh ** b2)
    r = lambda z: 31.5 * obhh / TCMB ** 4 / (z / 1e3); r_d, r_eq = r(z_d), r(z_eq)
    s = 2 / (3 * k_eq) * math.sqrt(6 / r_eq) * math.log((math.sqrt(1 + r_d) + math.sqrt(r_d + r_eq)) / (1 + math.sqrt(r_eq)))
    k_silk = 1.6 * obhh ** .52 * omhh ** .73 * (1 + (10.4 * omhh) ** -.95)
    q = kk / (13.41 * k_eq)
    a1 = (46.9 * omhh) ** .67 * (1 + (32.1 * omhh) ** -.532); a2 = (12 * omhh) ** .424 * (1 + (45 * omhh) ** -.582)
    alpha_c = a1 ** -fb * a2 ** (-fb ** 3)
    bb1 = .944 / (1 + (458 * omhh) ** -.708); bb2 = (.395 * omhh) ** -.0266; beta_c = 1 / (1 + bb1 * (fc ** bb2 - 1))
    def t0(alpha, beta):
        c = 14.2 / alpha + 386 / (1 + 69.9 * q ** 1.08); l = np.log(math.e + 1.8 * beta * q); return l / (l + c * q * q)
    f = 1 / (1 + (kk * s / 5.4) ** 4); t_c = f * t0(1, beta_c) + (1 - f) * t0(alpha_c, beta_c)
    y = (1 + z_eq) / (1 + z_d); sq = math.sqrt(1 + y); g = y * (-6 * sq + (2 + 3 * y) * math.log((sq + 1) / (sq - 1)))
    alpha_b = 2.07 * k_eq * s * (1 + r_d) ** -.75 * g; beta_b = .5 + fb + (3 - 2 * fb) * math.sqrt((17.2 * omhh) ** 2 + 1)
    beta_node = 8.41 * omhh ** .435
    with np.errstate(divide='ignore', invalid='ignore'):
        s_tilde = s / np.cbrt(1 + (beta_node / (kk * s)) ** 3); x = kk * s_tilde; j0 = np.where(np.abs(x) < 1e-4, 1 - x * x / 6, np.sin(x) / x)
        t_b = (t0(1, 1) / (1 + (kk * s / 5.2) ** 2) + alpha_b / (1 + (beta_b / (kk * s)) ** 3) * np.exp(-(kk / k_silk) ** 1.4)) * j0
    return np.where(k == 0, 1., fb * t_b + fc * t_c)
def top_hat(x):
    x = np.asarray(x, float); return np.where(np.abs(x) < .01, 1 - x * x / 10 + x ** 4 / 280 - x ** 6 / 15120, 3 * (np.sin(x) - x * np.cos(x)) / x ** 3)
def amplitude(req):
    bao = req['transfer'] == 'bao'; lnk = np.linspace(math.log(1e-5), math.log(100), 4097); k = np.exp(lnk)
    T = (transfer_bao if bao else transfer_smooth)(k, req['omegaM'], req['omegaB'], req['h'])
    f = k ** (3 + req['ns']) * T ** 2 * top_hat(8 * k) ** 2
    w = np.ones_like(k); w[1:-1:2] = 4; w[2:-1:2] = 2
    return req['sigma8'] ** 2 / (np.sum(w * f) * (lnk[1] - lnk[0]) / (6 * math.pi ** 2))
def linear_input(req, k, growth_ratio):
    k = np.asarray(k, float); bao = req['transfer'] == 'bao'
    T = (transfer_bao if bao else transfer_smooth)(k, req['omegaM'], req['omegaB'], req['h'])
    p = amplitude(req) * k ** req['ns'] * T ** 2 * np.exp(-(k * req['initialSmoothing']) ** 2) * growth_ratio ** 2 / req['boxSize'] ** (3 - req['dim'])
    return np.where(k * req['boxSize'] <= 2 * math.pi * req['initialCutoff'] + 1e-9, p, 0.)

def run_request(name, req, cache):
    f = cache / f'{name}.json'
    if f.exists():
        return json.loads(f.read_text())
    if not BIN.exists():
        sys.exit(f'{BIN} is missing; build it with cargo first')
    t0 = time.time(); res = subprocess.run([str(BIN)], input=json.dumps(req), capture_output=True, text=True, check=True)
    d = json.loads(res.stdout); d['_seconds'] = time.time() - t0
    if 'error' in d: sys.exit(f'{name}: {d["error"]}')
    f.write_text(json.dumps(d)); print(f'{name}: {d["_seconds"]:.0f} s', flush=True)
    return d

def fine_spectra(d, req):
    """Per-mode power of the exported cell-average fields, binned in annuli of one fundamental mode,
    with the top-hat cell window deconvolved per mode. P is |delta_k|^2 L^d in (Mpc/h)^d."""
    dim, mesh, L = req['dim'], req['mesh'], req['boxSize']
    kf = 2 * math.pi / L; freqs = np.fft.fftfreq(mesh, d=1 / mesh)
    grids = np.meshgrid(*([freqs] * dim), indexing='ij'); modes = np.sqrt(sum(g * g for g in grids))
    window = np.prod([np.sinc(g / mesh) for g in grids], axis=0)  # sinc(k dx / 2 / pi) with dx = L / mesh
    growth_ratio = d['D'] / d['growthAtOne']
    plin = linear_input(req, modes * kf, growth_ratio)
    m = np.rint(modes).astype(int); nb = mesh // 2; sel = (m >= 1) & (m <= nb)
    counts = np.bincount(m[sel], minlength=nb + 1)[1:]
    out = {'k': (np.arange(1, nb + 1) * kf).tolist(), 'counts': counts.tolist()}
    for key in ('rho', 'pmRho', 'cicRho', 'tscRho'):
        field = np.asarray(d[key], float).reshape([mesh] * dim)
        delta = np.fft.fftn(field / field.mean() - 1) / field.size
        power = np.abs(delta) ** 2 * L ** dim
        out[key] = (np.bincount(m[sel], weights=power[sel], minlength=nb + 1)[1:] / counts).tolist()
        out[key + 'Deconvolved'] = (np.bincount(m[sel], weights=(power / window ** 2)[sel], minlength=nb + 1)[1:] / counts).tolist()
    out['linear'] = (np.bincount(m[sel], weights=plin[sel], minlength=nb + 1)[1:] / counts).tolist()
    # Check the duplicated transfer function against the core's exported values at its own bins.
    core_k, core_p = np.asarray(d['spectra']['linearK']), np.asarray(d['spectra']['linear'])
    mine = linear_input(req, core_k, growth_ratio)
    assert np.allclose(mine, core_p, rtol=1e-6), f'linear input mismatch: {mine} vs {core_p}'
    return out

def summarize(name, d, req):
    g, diag = d['guard'], d['diagnostics']
    s = {'request': req, 'seconds': d.get('_seconds'), 'engineVersion': d.get('engineVersion'), 'a': d['a'], 'D': d['D'], 'growthAtOne': d['growthAtOne'],
         'minimumVertexJacobian': g['minimumJacobian'], 'minimumSimplexJacobian': g['minimumSimplexJacobian'], 'pmMinimumSimplexJacobian': g['pmMinimumSimplexJacobian'],
         'lptInvertedMass': g['lptFinalInvertedMass'], 'pmInvertedMass': g['pmFinalInvertedMass'], 'crossed': g['crossed'],
         'positionRelativeRms': diag['positionRelativeRms'], 'powerRelativeError': diag['powerRelativeError'],
         'variance': d['pdf']['variance'], 'pmVariance': d['pdf']['pmVariance'], 'mass': diag['mass'], 'pmMass': diag['pmMass'], 'cicMass': diag['cicMass'], 'tscMass': diag['tscMass']}
    s['spectra'] = fine_spectra(d, req)
    if name == 'estimators2d':
        s['cells'] = {}
        for key in ('rho', 'pmRho', 'cicRho', 'tscRho'):
            f = np.asarray(d[key], float); s['cells'][key] = {'empty': int(np.sum(f <= 1e-6)), 'min': float(f.min()), 'max': float(f.max())}
    return s

def band_mean_abs(k, a, b, lo, hi):
    k, a, b = map(np.asarray, (k, a, b)); sel = (k >= lo) & (k <= hi) & (b > 0)
    return float(np.mean(np.abs(a[sel] / b[sel] - 1)))

def write_tables(summary):
    rows = ['| LPT order | Min. vertex $J$ (LPT) | Min. simplex $J$ (PM drift) | Inverted mass LPT / PM | Position RMS / displacement RMS | $\\langle\\lvert P_{\\rm LPT}/P_{\\rm PM}-1\\rvert\\rangle$, $k\\le0.1$ | $\\sigma^2_{\\rm LPT}/\\sigma^2_{\\rm PM}$ at $R_s=9\\,h^{-1}$Mpc |',
            '|---|---|---|---|---|---|---|']
    for order in (1, 2, 3, 4):
        s = summary[f'lcdm3d_o{order}']; sp = s['spectra']
        rows.append(f"| {order} | {s['minimumVertexJacobian']:.3f} | {s['pmMinimumSimplexJacobian']:.4f} | {100*s['lptInvertedMass']:.2f}% / {100*s['pmInvertedMass']:.2f}% | {100*s['positionRelativeRms']:.1f}% | {100*band_mean_abs(sp['k'], sp['rho'], sp['pmRho'], 0, .1):.1f}% | {s['variance']/s['pmVariance']:.3f} |")
    s = summary['lcdm3d_o4']
    caption = (f": Fourth-order LPT against the particle-mesh reference for the fiducial $64^3$ run at $a=1$ ($z=0$). The displacement residual is the RMS of $\\mathbf x_{\\rm LPT}-\\mathbf x_{\\rm PM}$ over the LPT displacement RMS; the power column averages the absolute ratio residual over fine bins up to $0.1\\,h\\,\\mathrm{Mpc}^{-1}$; the variance ratio uses the Gaussian-smoothed cell fields. The PM trajectory is the same at every order, so its minimum drift Jacobian repeats. Engine {s['engineVersion']}; requests in `scripts/render_intro_figures.py`. {{#tbl-intro-orders}}")
    (ROOT / 'chapters' / '_intro-order-table.qmd').write_text('\n'.join(rows) + '\n\n' + caption + '\n')
    rows = ['| Run | Transfer | Min. vertex $J$ | Inverted mass LPT / PM | Position RMS / displacement RMS | $\\langle\\lvert P_{\\rm 4LPT}/P_{\\rm PM}-1\\rvert\\rangle$ in the BAO band | $\\langle\\lvert P_{\\rm 2LPT}/P_{\\rm PM}-1\\rvert\\rangle$ in the BAO band |',
            '|---|---|---|---|---|---|---|']
    for name, label, lo, hi in (('bao2d', '2D, $1000\\,h^{-1}$Mpc, $256^2$', .04, .19), ('bao3d', '3D, $600\\,h^{-1}$Mpc, $64^3$', .04, .073)):
        for t in ('bao', 'smooth'):
            s = summary[f'{name}_{t}']; sp = s['spectra']
            second = summary.get(f'{name}_{t}_o2')
            o2 = f"{100*band_mean_abs(second['spectra']['k'], second['spectra']['rho'], second['spectra']['pmRho'], lo, hi):.1f}%" if second else '—'
            rows.append(f"| {label} | {'EH98 with BAO' if t == 'bao' else 'EH98 smooth'} | {s['minimumVertexJacobian']:.3f} | {100*s['lptInvertedMass']:.2f}% / {100*s['pmInvertedMass']:.2f}% | {100*s['positionRelativeRms']:.1f}% | {100*band_mean_abs(sp['k'], sp['rho'], sp['pmRho'], lo, hi):.1f}% | {o2} |")
    caption = (": The BAO runs at $z=0$: the same seed with the oscillating and the smooth transfer. The band averages run over fine bins between $0.04\\,h\\,\\mathrm{Mpc}^{-1}$ and the radial cutoff of each box. The second-order column uses the same realization evolved with 2LPT where it was run. {#tbl-intro-bao}")
    (ROOT / 'chapters' / '_intro-bao-table.qmd').write_text('\n'.join(rows) + '\n\n' + caption + '\n')

def plot_bao(summary):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.hashsalt': 'precrossedsheet-intro'})
    BLUE, ORANGE, GREEN, GREY, PURPLE = '#0072B2', '#D55E00', '#009E73', '#777777', '#CC79A7'
    b, s2, b2 = summary['bao2d_bao'], summary['bao2d_smooth'], summary['bao2d_bao_o2']
    k = np.asarray(b['spectra']['k']); kc = 2 * math.pi * b['request']['initialCutoff'] / b['request']['boxSize']
    sel = (k <= kc); kk = k[sel]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axs[0]
    ax.loglog(kk, np.asarray(b['spectra']['linear'])[sel], color=GREY, lw=2.2, label='linear input, BAO transfer')
    ax.loglog(kk, np.asarray(s2['spectra']['linear'])[sel], color=GREY, lw=1.2, ls=':', label='linear input, smooth transfer')
    ax.loglog(kk, np.asarray(b['spectra']['rhoDeconvolved'])[sel], color=BLUE, lw=1.6, label='4LPT sheet, BAO')
    ax.loglog(kk, np.asarray(b['spectra']['pmRhoDeconvolved'])[sel], color=ORANGE, lw=1.2, ls='--', label='PM sheet, BAO')
    ax.loglog(kk, np.asarray(s2['spectra']['rhoDeconvolved'])[sel], color=GREEN, lw=1.0, ls='-.', label='4LPT sheet, smooth')
    ax.set(xlabel='k [h/Mpc]', ylabel='P(k) [(Mpc/h)²]', title='Power spectra at z = 0, same phases'); ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    lin_ratio = np.asarray(b['spectra']['linear'])[sel] / np.asarray(s2['spectra']['linear'])[sel]
    ax.plot(kk, lin_ratio, color=GREY, lw=2.2, label='linear: BAO / smooth input')
    ax.plot(kk, np.asarray(b['spectra']['rho'])[sel] / np.asarray(s2['spectra']['rho'])[sel], color=BLUE, lw=1.6, label='4LPT sheet: BAO run / smooth run')
    ax.plot(kk, np.asarray(b['spectra']['pmRho'])[sel] / np.asarray(s2['spectra']['pmRho'])[sel], color=ORANGE, lw=1.2, ls='--', label='PM sheet: BAO run / smooth run')
    ax.plot(kk, np.asarray(b2['spectra']['rho'])[sel] / np.asarray(summary['bao2d_smooth_o2']['spectra']['rho'])[sel], color=PURPLE, lw=1.0, ls='-.', label='2LPT sheet: BAO run / smooth run')
    ax.axhline(1, color='#bbbbbb', lw=.8); ax.set_xscale('log')
    ax.set(xlabel='k [h/Mpc]', ylabel='power ratio', title='The acoustic modulation through the nonlinear sheet'); ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / 'intro-bao.svg', bbox_inches='tight', metadata={'Date': None}); fig.savefig(FIG / 'intro-bao.png', dpi=170, bbox_inches='tight'); plt.close(fig)

def plot_estimators(d, req, summary):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'svg.hashsalt': 'precrossedsheet-intro'})
    mesh = req['mesh']; fields = [('rho', 'Exact sheet, 4LPT vertices'), ('cicRho', 'CIC, the same vertices as points'), ('tscRho', 'TSC, the same vertices as points')]
    vmax = max(np.asarray(d[key], float).max() for key, _ in fields)
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw={'width_ratios': [1, 1, 1, 1.25]})
    for ax, (key, title) in zip(axs[:3], fields):
        f = np.asarray(d[key], float).reshape(mesh, mesh)
        im = ax.imshow(np.log10(np.maximum(f, 1e-3)).T, origin='lower', extent=[0, req['boxSize'], 0, req['boxSize']], cmap='cividis', vmin=-2, vmax=math.log10(vmax))
        cells = summary['estimators2d']['cells'][key]
        ax.set(title=title, xlabel='x₀ [Mpc/h]'); ax.text(.02, .97, f"{cells['empty']} empty of {mesh**2} cells", transform=ax.transAxes, va='top', fontsize=8, color='white')
    axs[0].set_ylabel('x₁ [Mpc/h]')
    cbar = fig.colorbar(im, ax=axs[:3], fraction=.02, pad=.01); cbar.set_label('log₁₀ ρ/mean')
    sp = summary['estimators2d']['spectra']; k = np.asarray(sp['k'])
    ax = axs[3]
    ax.loglog(k, sp['rho'], color='#0072B2', lw=1.6, label='exact sheet')
    ax.loglog(k, sp['cicRho'], color='#D55E00', lw=1.1, label='CIC')
    ax.loglog(k, sp['tscRho'], color='#009E73', lw=1.1, label='TSC')
    shot = req['boxSize'] ** 2 / req['n'] ** 2
    ax.axhline(shot, color='#CC79A7', ls='--', lw=1, label='Poisson L²/N')
    ax.axvline(math.pi * req['n'] / req['boxSize'], color='#999999', ls=':', lw=1, label='particle Nyquist')
    ax.set(xlabel='k [h/Mpc]', ylabel='P(k) [(Mpc/h)²]', title='Spectra of the three maps'); ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIG / 'intro-estimators.svg', bbox_inches='tight', metadata={'Date': None}); fig.savefig(FIG / 'intro-estimators.png', dpi=170, bbox_inches='tight'); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', type=Path, help='cache directory of native run JSONs (created as needed)')
    args = ap.parse_args()
    if args.runs:
        args.runs.mkdir(parents=True, exist_ok=True); summary = {}
        for name, req in REQUESTS.items():
            d = run_request(name, req, args.runs); summary[name] = summarize(name, d, req)
            if name == 'estimators2d': estimators = d
        SUMMARY.write_text(json.dumps(summary, indent=0))
        plot_estimators(estimators, REQUESTS['estimators2d'], summary)
    else:
        summary = json.loads(SUMMARY.read_text())
    write_tables(summary); plot_bao(summary)
    print('wrote', SUMMARY.relative_to(ROOT), 'and the introduction figures and tables')

if __name__ == '__main__':
    main()
