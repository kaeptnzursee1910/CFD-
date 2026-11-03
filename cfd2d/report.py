"""Auswertung nach der Simulation: Diagramme, Kennzahlen, Datenexport."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import validation
from .visualization import field_cmap, field_data, field_label


def _plt():
    import matplotlib.pyplot as plt

    return plt


def _field_axes(ax, sim, field, title=None, streamlines=False, contour=False):
    plt = _plt()
    g = sim.grid
    data = field_data(sim, field)
    cmap = plt.get_cmap(field_cmap(field)).with_extremes(bad="0.35")
    finite = data[np.isfinite(data)]
    vmin = vmax = None
    if finite.size and field in ("vorticity", "v"):
        a = np.percentile(np.abs(finite), 99) or 1e-12
        vmin, vmax = -a, a
    im = ax.imshow(np.ma.masked_invalid(data.T), origin="lower", extent=(0, g.lx, 0, g.ly),
                   cmap=cmap, interpolation="bilinear", aspect="equal", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02).set_label(field_label(sim, field), fontsize=8)
    if streamlines:
        uc, vc = sim.cell_velocity()
        uc, vc = np.nan_to_num(uc), np.nan_to_num(vc)
        if np.any(uc) or np.any(vc):
            dens = (min(3.0, 1.2 * g.lx / g.ly), 1.2) if g.lx > g.ly else (1.4, 1.4)
            ax.streamplot(g.xc, g.yc, uc.T, vc.T, color="k", linewidth=0.5, density=dens,
                          arrowsize=0.6)
    if contour:
        psi = sim.streamfunction()
        s = ~g.fluid.copy()
        s[0, :] = s[-1, :] = s[:, 0] = s[:, -1] = False
        inside = s[:-1, :-1] & s[1:, :-1] & s[:-1, 1:] & s[1:, 1:]
        ax.contour(g.xn, g.yn, np.ma.masked_array(psi, inside).T, levels=30, colors="k",
                   linewidths=0.5)
    ax.set_title(title or field_label(sim, field), fontsize=10)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0, g.lx)
    ax.set_ylim(0, g.ly)


def plot_fields(case, path):
    plt = _plt()
    sim = case.sim
    g = sim.grid
    wide = g.lx / g.ly > 1.6
    if wide:
        fig, axes = plt.subplots(4, 1, figsize=(11, min(4 * 11 / (g.lx / g.ly) * 0.9 + 2, 16)))
    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = np.ravel(axes)
    _field_axes(axes[0], sim, "speed", "Geschwindigkeitsbetrag mit Stromlinien", streamlines=True)
    _field_axes(axes[1], sim, "vorticity")
    _field_axes(axes[2], sim, "pressure")
    _field_axes(axes[3], sim, "streamfunction", "Stromfunktion ψ (Isolinien = Stromlinien)",
                contour=True)
    fig.suptitle(f"{case.title}: Re = {case.re:g}, t = {sim.t:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_model_fields(case, path) -> bool:
    """Felder der Zusatzmodelle (Temperatur, Skalare, Turbulenzgrößen) mit Isolinien."""
    from .visualization import FIELDS, available_fields

    plt = _plt()
    sim = case.sim
    g = sim.grid
    keys = [k for k in available_fields(sim) if k not in FIELDS]
    if not keys:
        return False
    wide = g.lx / g.ly > 1.6
    n = len(keys)
    if wide:
        fig, axes = plt.subplots(n, 1, figsize=(11, min(n * 11 / (g.lx / g.ly) * 0.9 + 1.5, 18)),
                                 squeeze=False)
    else:
        cols = min(n, 2)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(6.2 * cols, 5 * rows), squeeze=False)
    axes = np.ravel(axes)
    for ax, k in zip(axes, keys):
        _field_axes(ax, sim, k)
        data = field_data(sim, k)
        if k in ("T",) or k in sim.scalars:
            if np.isfinite(data).any() and np.nanmax(data) - np.nanmin(data) > 1e-12:
                ax.contour(g.xc, g.yc, np.ma.masked_invalid(data).T, levels=12, colors="w",
                           linewidths=0.5, alpha=0.8)
    for ax in axes[len(keys):]:
        ax.set_visible(False)
    fig.suptitle(f"{case.title}: Modellfelder, t = {sim.t:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def plot_convergence(case, path):
    plt = _plt()
    h = case.sim.history.as_arrays()
    if len(h["t"]) == 0:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].semilogy(h["t"], np.maximum(h["residual"], 1e-300))
    axes[0].set_title("Residuum max|Δu|/Δt")
    axes[1].semilogy(h["t"], np.maximum(h["divergence"], 1e-300), color="C3")
    axes[1].set_title("max |div u| (Massenerhaltung)")
    axes[2].plot(h["t"], h["kinetic_energy"], color="C2")
    axes[2].set_title("Kinetische Energie")
    for ax in axes:
        ax.set_xlabel("Zeit t")
        ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------- #
# Fallspezifische Auswertungen
# ---------------------------------------------------------------------- #
def report_cavity(case, outdir: Path) -> dict:
    plt = _plt()
    sim = case.sim
    res = {}
    y = np.linspace(0, 1, 201)
    u = sim.sample_u(np.full_like(y, 0.5), y)
    v = sim.sample_v(y, np.full_like(y, 0.5))
    ref = validation.ghia_reference(case.re)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(u, y, label="Simulation")
    axes[0].set_xlabel("u(x = 0.5, y)")
    axes[0].set_ylabel("y")
    axes[0].set_title("u entlang der vertikalen Mittellinie")
    axes[1].plot(y, v, label="Simulation")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("v(x, y = 0.5)")
    axes[1].set_title("v entlang der horizontalen Mittellinie")
    if ref:
        r, d = ref
        axes[0].plot(d["u"], d["y"], "o", mfc="none", color="C3", label=f"Ghia et al. (Re={r})")
        axes[1].plot(d["x"], d["v"], "o", mfc="none", color="C3", label=f"Ghia et al. (Re={r})")
        err = validation.cavity_centerline_errors(sim, case.re)
        res["ghia_vergleich"] = err
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Lid-Driven Cavity, Re = {case.re:g}, Gitter {sim.grid.nx}x{sim.grid.ny}")
    fig.tight_layout()
    fig.savefig(outdir / "validierung_mittellinien.png", dpi=130)
    plt.close(fig)

    # Hauptwirbel: Minimum der Stromfunktion
    psi = sim.streamfunction()
    i, j = np.unravel_index(np.argmin(psi), psi.shape)
    res["hauptwirbel"] = {"x": float(sim.grid.xn[i]), "y": float(sim.grid.yn[j]),
                          "psi_min": float(psi[i, j])}
    return res


def report_cylinder(case, outdir: Path) -> dict:
    plt = _plt()
    sim = case.sim
    h = sim.history.as_arrays()
    res = {}
    if len(h["fx"]) == 0:
        return res
    t = h["t"]
    cd, cl = validation.force_coefficients(h["fx"], h["fy"], case.U_ref, case.L_ref)
    t_start = t[0] + 0.5 * (t[-1] - t[0])
    m = t >= t_start
    res["cd_mittel"] = float(cd[m].mean())
    res["cd_max"] = float(cd[m].max())
    res["cl_max"] = float(cl[m].max())
    res["cl_min"] = float(cl[m].min())
    res["cd_ende"] = float(cd[-1])
    res["cl_ende"] = float(cl[-1])
    st = validation.strouhal_number(t, cl, case.L_ref, case.U_ref, t_start)
    res["strouhal"] = st
    if "referenz" in case.info:
        res["referenz_schaefer_turek"] = case.info["referenz"]

    fig, axes = plt.subplots(1, 3 if st else 2, figsize=(15 if st else 11, 4))
    axes[0].plot(t, cd, color="C0")
    axes[0].set_title("Widerstandsbeiwert $c_D$")
    axes[1].plot(t, cl, color="C1")
    axes[1].set_title("Auftriebsbeiwert $c_L$")
    ref = case.info.get("referenz", {})
    if "cd_max" in ref:
        axes[0].axhspan(*ref["cd_max"], color="C3", alpha=0.25, label="Referenz $c_{D,max}$")
        axes[1].axhspan(*ref["cl_max"], color="C3", alpha=0.25, label="Referenz $c_{L,max}$")
    if "cd" in ref:
        axes[0].axhspan(*ref["cd"], color="C3", alpha=0.25, label="Referenz $c_D$")
    for ax in axes[:2]:
        ax.set_xlabel("Zeit t")
        ax.grid(True, alpha=0.3)
        if ref:
            ax.legend(fontsize=8)
    # nach dem Einschwingen sinnvoll skalieren
    for ax, y in ((axes[0], cd), (axes[1], cl)):
        tail = y[len(y) // 5:]
        lo, hi = tail.min(), tail.max()
        pad = 0.15 * max(hi - lo, 1e-3)
        ax.set_ylim(lo - pad, hi + pad)
    if st:
        f, amp, freqs, spec = validation.dominant_frequency(t, cl, t_start)
        stf = freqs * case.L_ref / case.U_ref
        axes[2].plot(stf, spec / spec.max(), color="C2")
        axes[2].axvline(st, color="k", ls="--", lw=0.8, label=f"St = {st:.4f}")
        if "st" in ref:
            axes[2].axvspan(*ref["st"], color="C3", alpha=0.25, label="Referenz")
        axes[2].set_xlim(0, min(4 * st, stf.max()))
        axes[2].set_xlabel("Strouhal-Zahl St = f D / U")
        axes[2].set_title("Spektrum des Auftriebs")
        axes[2].legend(fontsize=8)
        axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"{case.title}, Re = {case.re:g}")
    fig.tight_layout()
    fig.savefig(outdir / "kraftbeiwerte.png", dpi=130)
    plt.close(fig)
    return res


def report_channel(case, outdir: Path) -> dict:
    plt = _plt()
    sim = case.sim
    g = sim.grid
    H = case.info.get("H", 1.0)
    y = np.linspace(0, H, 101)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    xs = [x for x in (0.1, 0.25, 0.5, 1.0, 2.0, 0.9 * g.lx / H) if x * H < g.lx * 0.95]
    for k, xr in enumerate(xs):
        u = sim.sample_u(np.full_like(y, xr * H), y)
        axes[0].plot(u, y, color=f"C{k}", label=f"x = {xr * H:.2f}")
    ua = validation.poiseuille_profile(y, H, case.U_ref)
    axes[0].plot(ua, y, "k--", lw=1.5, label="Poiseuille (analytisch)")
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("y")
    axes[0].set_title("Geschwindigkeitsprofile")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    pc = sim.cell_pressure()[:, g.ny // 2]
    axes[1].plot(g.xc, pc, label="Simulation (Kanalmitte)")
    dpdx = validation.poiseuille_pressure_gradient(sim.nu, H, case.U_ref)
    x_out = g.xc[-1]
    axes[1].plot(g.xc, dpdx * (g.xc - x_out) + pc[-1], "k--", lw=1,
                 label=f"ausgebildet: dp/dx = {dpdx:.4f}")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("p")
    axes[1].set_title("Druckverlauf")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Kanalströmung, Re = {case.re:g}")
    fig.tight_layout()
    fig.savefig(outdir / "profile.png", dpi=130)
    plt.close(fig)

    x_end = 0.9 * g.lx
    yy = g.yc
    u_end = sim.sample_u(np.full_like(yy, x_end), yy)
    ua_end = validation.poiseuille_profile(yy, H, case.U_ref)
    i1, i2 = int(0.7 * g.nx), int(0.9 * g.nx)
    dpdx_num = float((pc[i2] - pc[i1]) / (g.xc[i2] - g.xc[i1]))
    return {
        "profil_x": float(x_end),
        "profil_max_fehler": float(np.abs(u_end - ua_end).max()),
        "profil_rel_fehler_L2": float(np.linalg.norm(u_end - ua_end) / np.linalg.norm(ua_end)),
        "dpdx_numerisch": dpdx_num,
        "dpdx_analytisch": float(dpdx),
        "einlauflaenge_abschaetzung": case.info.get("einlauflaenge"),
    }


def report_step(case, outdir: Path) -> dict:
    plt = _plt()
    sim = case.sim
    g = sim.grid
    xs = case.info.get("x_stufe", 0.0)
    H = case.info.get("H", 1.0)
    xr = validation.reattachment_length(sim, xs, "bottom")
    upper = validation.separation_bubbles(sim, "top", x_min=xs)
    res = {"wiederanlegelaenge_unten": xr,
           "wiederanlegelaenge_unten_pro_H": (xr / H) if xr else None,
           "abloesegebiete_oben": upper}

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    x = np.arange(g.nx + 1) * g.dx
    tau_b = sim.nu * sim.u[:, 1] / (0.5 * g.dy)
    tau_t = -sim.nu * sim.u[:, g.ny] / (0.5 * g.dy)
    m = x > xs
    axes[0].plot(x[m], tau_b[m], label="untere Wand")
    axes[0].plot(x, -tau_t, label="obere Wand (Vorzeichen gedreht)")
    axes[0].axhline(0, color="k", lw=0.6)
    if xr:
        axes[0].axvline(xs + xr, color="C3", ls="--", label=f"Wiederanlegung x_r = {xr:.3f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("Wandschubspannung τ_w")
    axes[0].set_title("Wandschubspannung (τ_w < 0 heißt Rückströmung)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    y = np.linspace(0, H, 101)
    for k, xp in enumerate([xs + d for d in (0.5, 1, 2, 4, 6, 8, 12) if xs + d < g.lx]):
        u = sim.sample_u(np.full_like(y, xp), y)
        axes[1].plot(xp + 0.5 * u, y, color=f"C{k % 10}")
        axes[1].axvline(xp, color=f"C{k % 10}", lw=0.4, ls=":")
    axes[1].fill_between([0, xs], 0, case.info.get("h", 0.5), color="0.5")
    axes[1].set_xlim(0, g.lx)
    axes[1].set_ylim(0, H)
    axes[1].set_xlabel("x  (Profile: x + u/2)")
    axes[1].set_ylabel("y")
    axes[1].set_title("Geschwindigkeitsprofile hinter der Stufe")
    axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Rückwärtsgewandte Stufe, Re = {case.re:g}")
    fig.tight_layout()
    fig.savefig(outdir / "stufe_auswertung.png", dpi=130)
    plt.close(fig)
    return res


def report_custom(case, outdir: Path) -> dict:
    if case.sim.compute_forces:
        return report_cylinder(case, outdir)
    return {}


def plot_monitors(case, path) -> bool:
    """Zeitverläufe der Modell-Monitore (Nusselt-Zahlen, Skalarintegrale, ...)."""
    plt = _plt()
    h = case.sim.history
    if not h.extra:
        return False
    t = np.asarray(h.t)
    keys = list(h.extra)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.6 * len(keys), 3.6), squeeze=False)
    for ax, k in zip(axes[0], keys):
        y = np.asarray(h.extra[k])
        ax.plot(t[-len(y):], y)
        ax.set_title(k.replace("_", " "))
        ax.set_xlabel("Zeit t")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def report_heated_cavity(case, outdir: Path) -> dict:
    plt = _plt()
    sim = case.sim
    th = sim.model("thermal")
    sc = case.info["geschw_skala_alpha"]
    y = np.linspace(0, 1, 201)
    u = sim.sample_u(np.full_like(y, 0.5), y) * sc
    v = sim.sample_v(y, np.full_like(y, 0.5)) * sc
    res = {
        "Nu_heiss": th.nusselt("left"), "Nu_kalt": th.nusselt("right"),
        "u_max": float(u.max()), "y_umax": float(y[u.argmax()]),
        "v_max": float(v.max()), "x_vmax": float(y[v.argmax()]),
        "hinweis": "Geschwindigkeiten in Einheiten alpha/L (wie de Vahl Davis)",
    }
    ref = validation.de_vahl_davis_reference(case.info["ra"])
    if ref:
        res["referenz_de_vahl_davis"] = ref
        res["Nu_rel_fehler"] = abs(res["Nu_heiss"] - ref["nu"]) / ref["nu"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    g = sim.grid
    T = th.field.cell()
    im = axes[0].contourf(g.xc, g.yc, T.T, levels=21, cmap="inferno")
    axes[0].contour(g.xc, g.yc, T.T, levels=11, colors="w", linewidths=0.6)
    plt.colorbar(im, ax=axes[0]).set_label("T")
    axes[0].set_aspect("equal")
    axes[0].set_title("Isothermen")
    axes[1].plot(u, y, label="u(0.5, y)")
    axes[1].plot(v, y, label="v(x, 0.5) über x")
    if ref:
        axes[1].plot(ref["u_max"], ref["y_umax"], "o", color="C0", mfc="none",
                     label="Referenz u_max")
        axes[1].plot(ref["v_max"], ref["x_vmax"], "s", color="C1", mfc="none",
                     label="Referenz v_max")
    axes[1].set_xlabel("Geschwindigkeit [alpha/L]")
    axes[1].set_ylabel("y bzw. x")
    axes[1].set_title("Mittellinienprofile")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(th.local_nusselt("left"), g.yc, label="heiße Wand")
    axes[2].plot(th.local_nusselt("right"), g.yc, label="kalte Wand")
    axes[2].set_xlabel("lokale Nusselt-Zahl")
    axes[2].set_ylabel("y")
    axes[2].set_title(f"Nu = {res['Nu_heiss']:.4f}" + (f" (Referenz {ref['nu']})" if ref else ""))
    axes[2].legend(fontsize=8)
    axes[2].grid(True, alpha=0.3)
    fig.suptitle(f"Beheizte Cavity, Ra = {case.info['ra']:g}, Pr = {case.info['pr']:g}, "
                 f"Gitter {g.nx}x{g.ny}")
    fig.tight_layout()
    fig.savefig(outdir / "waermeuebergang.png", dpi=130)
    plt.close(fig)
    return res


def report_rayleigh_benard(case, outdir: Path) -> dict:
    plt = _plt()
    sim = case.sim
    th = sim.model("thermal")
    g = sim.grid
    res = {"Nu_unten": th.nusselt("bottom"), "Nu_oben": th.nusselt("top"),
           "ra": case.info["ra"], "ra_krit": case.info["ra_krit"],
           "konvektion": bool(th.nusselt("bottom") > 1.05)}
    h = sim.history.extra
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    T = th.field.cell()
    im = axes[0].imshow(T.T, origin="lower", extent=(0, g.lx, 0, g.ly), cmap="inferno",
                        aspect="equal")
    uc, vc = sim.cell_velocity()
    axes[0].streamplot(g.xc, g.yc, np.nan_to_num(uc).T, np.nan_to_num(vc).T, color="w",
                       linewidth=0.5, density=1.2, arrowsize=0.6)
    plt.colorbar(im, ax=axes[0], fraction=0.02).set_label("T")
    axes[0].set_title("Temperatur und Stromlinien (Konvektionsrollen)")
    if "Nu_bottom" in h:
        t = np.asarray(sim.history.t)
        axes[1].plot(t, h["Nu_bottom"], label="Nu unten")
        axes[1].plot(t, h["Nu_top"], label="Nu oben")
        axes[1].axhline(1.0, color="k", ls="--", lw=0.8, label="reine Wärmeleitung")
        axes[1].set_xlabel("Zeit t")
        axes[1].set_ylabel("Nu")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    fig.suptitle(f"Rayleigh-Bénard, Ra = {case.info['ra']:g} (Ra_krit = 1708), "
                 f"Nu = {res['Nu_unten']:.4f}")
    fig.tight_layout()
    fig.savefig(outdir / "rayleigh_benard.png", dpi=130)
    plt.close(fig)
    return res


REPORTS = {"cavity": report_cavity, "cylinder": report_cylinder, "channel": report_channel,
           "step": report_step, "custom": report_custom,
           "heated_cavity": report_heated_cavity, "rayleigh_benard": report_rayleigh_benard}


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def create_report(case, outdir, reason: str = "") -> dict:
    """Erstellt alle Diagramme und Dateien im Ordner ``outdir``.

    Dateien:
      * ``felder.png``: Geschwindigkeit, Wirbelstärke, Druck, Stromfunktion
      * ``konvergenz.png``: Residuum, Divergenz, kinetische Energie
      * fallspezifische Diagramme (Validierung, Kraftbeiwerte, Profile …)
      * ``ergebnis.npz``: Rohdaten (u, v, p, Zeitverläufe)
      * ``zusammenfassung.json``: Kennzahlen und Parameter
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sim = case.sim
    plot_fields(case, outdir / "felder.png")
    plot_convergence(case, outdir / "konvergenz.png")
    plot_model_fields(case, outdir / "felder_modelle.png")
    plot_monitors(case, outdir / "monitore.png")
    specific = REPORTS.get(case.key, lambda c, o: {})(case, outdir)

    hist = sim.history.as_arrays()
    np.savez_compressed(outdir / "ergebnis.npz", u=sim.u, v=sim.v, p=sim.p,
                        fluid=sim.grid.fluid, dx=sim.grid.dx, dy=sim.grid.dy,
                        **{f"hist_{k}": v for k, v in hist.items()})

    summary = {
        "fall": case.title,
        "reynolds": case.re,
        "gitter": [sim.grid.nx, sim.grid.ny],
        "gebiet": [sim.grid.lx, sim.grid.ly],
        "nu": sim.nu,
        "zeitschema": sim.scheme,
        "endzeit": sim.t,
        "schritte": sim.n,
        "abbruchgrund": reason,
        "rechenzeit_s": sim.wall_time,
        "residuum_ende": sim.residual,
        "max_divergenz": sim.max_divergence(),
        "auswertung": specific,
        "modelle": [m.describe() for m in sim.models],
    }
    if case.key == "custom":
        summary["hindernisse"] = case.info.get("hindernisse")
    with open(outdir / "zusammenfassung.json", "w", encoding="utf-8") as f:
        json.dump(_jsonable(summary), f, indent=2, ensure_ascii=False)
    return summary


def print_summary(summary: dict) -> None:
    print("\n" + "=" * 70)
    print(f" Ergebnis: {summary['fall']}  (Re = {summary['reynolds']:g})")
    print("=" * 70)
    print(f" Schritte: {summary['schritte']},  Endzeit t = {summary['endzeit']:.4f},  "
          f"Abbruch: {summary['abbruchgrund']}")
    print(f" Residuum: {summary['residuum_ende']:.3e},  max|div u|: {summary['max_divergenz']:.2e}")
    a = summary.get("auswertung") or {}

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.5g}"
        if isinstance(v, (tuple, list)) and all(isinstance(x, (int, float)) for x in v):
            return " – ".join(f"{x:g}" for x in v)
        if isinstance(v, dict):
            return ", ".join(f"{k}: {fmt(x)}" for k, x in v.items())
        return str(v)

    for k, v in a.items():
        print(f" {k}: {fmt(v)}")
    print("=" * 70)
