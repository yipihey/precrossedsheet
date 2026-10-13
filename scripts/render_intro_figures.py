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
    s['spectra']['coreRealizedK'] = d['spectra'].get('realizedK', [])
    s['spectra']['coreRealized'] = d['spectra'].get('realized', [])
    if name == 'estimators2d':
        s['cells'] = {}
        for key in ('rho', 'pmRho', 'cicRho', 'tscRho'):
            f = np.asarray(d[key], float); s['cells'][key] = {'empty': int(np.sum(f <= 1e-6)), 'min': float(f.min()), 'max': float(f.max())}
    return s

def band_mean_abs(k, a, b, lo, hi):
    k, a, b = map(np.asarray, (k, a, b)); sel = (k >= lo) & (k <= hi) & (b > 0)
    return float(np.mean(np.abs(a[sel] / b[sel] - 1)))

def transmission_error(summary, name, lo, hi, key):
    """Mean |(P_bao/P_smooth)_measured / (P_bao/P_smooth)_linear - 1| over the band: how well the
    acoustic modulation of the input is carried through the nonlinear map and the estimator."""
    b, sm = summary[f'{name}_bao']['spectra'], summary[f'{name}_smooth']['spectra']
    k = np.asarray(b['k']); sel = (k >= lo) & (k <= hi)
    linear = np.asarray(b['linear'])[sel] / np.asarray(sm['linear'])[sel]
    measured = np.asarray(b[key])[sel] / np.asarray(sm[key])[sel]
    return float(np.mean(np.abs(measured / linear - 1)))

def write_tables(summary):
    reference = summary['lcdm3d_o4']['spectra']
    rows = ['| LPT order $n$ | Min. vertex $J$ | Inverted mass LPT / PM | $\\langle\\lvert P_n/P_4-1\\rvert\\rangle$, $k\\le0.1$ | $\\sigma^2_n/\\sigma^2_4$ | Position RMS vs PM / displacement RMS | $\\langle\\lvert P_n/P_{\\rm PM}-1\\rvert\\rangle$, $k\\le0.1$ | $\\sigma^2_n/\\sigma^2_{\\rm PM}$ |',
            '|---|---|---|---|---|---|---|---|']
    for order in (1, 2, 3, 4):
        s = summary[f'lcdm3d_o{order}']; sp = s['spectra']
        rows.append(f"| {order} | {s['minimumVertexJacobian']:.3f} | {100*s['lptInvertedMass']:.2f}% / {100*s['pmInvertedMass']:.2f}% | {100*band_mean_abs(sp['k'], sp['rho'], reference['rho'], 0, .1):.2f}% | {s['variance']/summary['lcdm3d_o4']['variance']:.4f} | {100*s['positionRelativeRms']:.1f}% | {100*band_mean_abs(sp['k'], sp['rho'], sp['pmRho'], 0, .1):.1f}% | {s['variance']/s['pmVariance']:.3f} |")
    s = summary['lcdm3d_o4']
    caption = (": Orders one to four for the fiducial $64^3$ run at $a=1$ ($z=0$), with the same PM trajectory as reference (its minimum drift Jacobian is "
               + f"{s['pmMinimumSimplexJacobian']:.4f}" + " and it inverts no element). The middle columns measure the convergence of the expansion itself: the exact-sheet power of order $n$ against order four, averaged over fine bins up to $0.1\\,h\\,\\mathrm{Mpc}^{-1}$, and the variance of the Gaussian-smoothed field ($R_s=9\\,h^{-1}$Mpc). The right columns compare with the $64^3$, 128-step PM run: the RMS of $\\mathbf x_n-\\mathbf x_{\\rm PM}$ over the LPT displacement RMS, and the same power and variance ratios. Engine "
               + str(s['engineVersion']) + "; requests in `scripts/render_intro_figures.py`. {#tbl-intro-orders}")
    (ROOT / 'chapters' / '_intro-order-table.qmd').write_text('\n'.join(rows) + '\n\n' + caption + '\n')
    rows = ['| Box | Band | Min. vertex $J$ (BAO / smooth) | Inverted mass | Position RMS vs PM / displacement RMS | Modulation error, 4LPT sheet | Modulation error, 2LPT sheet | Modulation error, PM sheet | $\\langle\\lvert P_{\\rm 4LPT}/P_{\\rm PM}-1\\rvert\\rangle$ |',
            '|---|---|---|---|---|---|---|---|---|']
    for name, label, lo, hi, pm_hi in (('bao2d', '2D, $1000\\,h^{-1}$Mpc, $256^2$', .03, .19, .1), ('bao3d', '3D, $600\\,h^{-1}$Mpc, $64^3$', .03, .073, .073)):
        b, sm = summary[f'{name}_bao'], summary[f'{name}_smooth']
        if f'{name}_bao_o2' in summary:
            o2 = {'x_bao': summary[f'{name}_bao_o2'], 'x_smooth': summary[f'{name}_smooth_o2']}
            second = f"{100*transmission_error(o2, 'x', lo, hi, 'rho'):.2f}%"
        else:
            second = '—'
        rows.append(f"| {label} | ${lo}$–${hi}\\,h\\,\\mathrm{{Mpc}}^{{-1}}$ | {b['minimumVertexJacobian']:.3f} / {sm['minimumVertexJacobian']:.3f} | {100*max(b['lptInvertedMass'], b['pmInvertedMass'], sm['lptInvertedMass'], sm['pmInvertedMass']):.2f}% | {100*b['positionRelativeRms']:.1f}% | {100*transmission_error(summary, name, lo, hi, 'rho'):.2f}% | {second} | {100*transmission_error(summary, name, lo, hi, 'pmRho'):.2f}% | {100*band_mean_abs(b['spectra']['k'], b['spectra']['rho'], b['spectra']['pmRho'], .03, pm_hi):.1f}% |")
    caption = (": The BAO runs at $z=0$: one seed evolved with the oscillating and with the smooth transfer. The modulation error is the band average of $\\lvert(P_{\\rm BAO}/P_{\\rm smooth})_{\\rm measured}/(P_{\\rm BAO}/P_{\\rm smooth})_{\\rm linear}-1\\rvert$, the departure of the measured acoustic modulation from the one put in; because both runs share their phases it has no sample variance. The last column compares the 4LPT and PM sheets where the filtered input still carries most of the power (up to $0.1\\,h\\,\\mathrm{Mpc}^{-1}$ in 2D); beyond that the PM reference at this resolution adds discreteness power where the input is damped by the $14\\,h^{-1}$Mpc filter. {#tbl-intro-bao}")
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
    if b['spectra'].get('coreRealizedK'):
        ax.loglog(b['spectra']['coreRealizedK'], b['spectra']['coreRealized'], 'o', color='#444444', ms=3.5, mfc='none', label='linear input, this realization (lab bins)')
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
    mesh = req['mesh']; fields = [('rho', 'Exact sheet, 4LPT vertices'), ('cicRho', 'CIC, same vertices as points'), ('tscRho', 'TSC, same vertices as points')]
    fig = plt.figure(figsize=(15.5, 4.3), constrained_layout=True)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1, .05, 1.35])
    axs = [fig.add_subplot(gs[0, i]) for i in range(3)]; cax = fig.add_subplot(gs[0, 3]); axp = fig.add_subplot(gs[0, 4])
    for ax, (key, title) in zip(axs, fields):
        f = np.asarray(d[key], float).reshape(mesh, mesh)
        im = ax.imshow(f.T, origin='lower', extent=[0, req['boxSize'], 0, req['boxSize']], cmap='cividis', vmin=0, vmax=2)
        cells = summary['estimators2d']['cells'][key]
        ax.set(title=title, xlabel='x₀ [Mpc/h]'); ax.text(.03, .97, f"{cells['empty']} empty of {mesh**2} cells", transform=ax.transAxes, va='top', fontsize=8.5, color='white', bbox={'facecolor': 'black', 'alpha': .45, 'pad': 2, 'lw': 0})
    axs[0].set_ylabel('x₁ [Mpc/h]'); axs[1].set_yticklabels([]); axs[2].set_yticklabels([])
    cbar = fig.colorbar(im, cax=cax); cbar.set_label('ρ / mean (clipped at 2)')
    sp = summary['estimators2d']['spectra']; k = np.asarray(sp['k'])
    axp.loglog(k, sp['rho'], color='#0072B2', lw=1.6, label='exact sheet')
    axp.loglog(k, sp['cicRho'], color='#D55E00', lw=1.1, label='CIC')
    axp.loglog(k, sp['tscRho'], color='#009E73', lw=1.1, label='TSC')
    shot = req['boxSize'] ** 2 / req['n'] ** 2
    axp.axhline(shot, color='#CC79A7', ls='--', lw=1, label='Poisson L²/N')
    axp.axvline(math.pi * req['n'] / req['boxSize'], color='#999999', ls=':', lw=1, label='particle Nyquist')
    axp.set(xlabel='k [h/Mpc]', ylabel='P(k) [(Mpc/h)²]', title='Spectra of the three maps'); axp.legend(frameon=False, fontsize=8)
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
