"""Live-Visualisierung während der Simulation und Aufnahme als Animation."""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np

FIELDS = {
    "speed": "Geschwindigkeitsbetrag |u|",
    "vorticity": "Wirbelstärke ω",
    "pressure": "Druck p",
    "u": "Geschwindigkeit u (x-Richtung)",
    "v": "Geschwindigkeit v (y-Richtung)",
    "streamfunction": "Stromfunktion ψ",
}

CMAPS = {"speed": "viridis", "vorticity": "RdBu_r", "pressure": "coolwarm",
         "u": "RdBu_r", "v": "RdBu_r", "streamfunction": "PuOr"}


def display_available() -> bool:
    """Prüft, ob ein interaktives Matplotlib-Fenster geöffnet werden kann."""
    import matplotlib

    backend = matplotlib.get_backend().lower()
    if backend in ("agg", "pdf", "svg", "ps", "cairo", "template") or "inline" in backend:
        return False
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY")
                                                  or os.environ.get("WAYLAND_DISPLAY")):
        return False
    return True


def ensure_backend(live: bool) -> bool:
    """Wählt ein passendes Backend. Gibt zurück, ob Live-Anzeige möglich ist."""
    import matplotlib

    if not live or not display_available():
        try:
            matplotlib.use("Agg", force=True)
        except Exception:  # pragma: no cover
            pass
        return False
    return True


def field_data(sim, field: str) -> np.ndarray:
    """Feld in den Zellmittelpunkten, Form (nx, ny), NaN im Festkörper."""
    if field == "speed":
        uc, vc = sim.cell_velocity()
        return np.sqrt(uc**2 + vc**2)
    if field == "vorticity":
        return sim.cell_vorticity()
    if field == "pressure":
        return sim.cell_pressure()
    if field == "u":
        return sim.cell_velocity()[0]
    if field == "v":
        return sim.cell_velocity()[1]
    if field == "streamfunction":
        psi = sim.streamfunction()
        pc = 0.25 * (psi[:-1, :-1] + psi[1:, :-1] + psi[:-1, 1:] + psi[1:, 1:])
        return np.where(sim.grid.solid, np.nan, pc)
    raise ValueError(f"Unbekanntes Feld {field!r}, erlaubt: {list(FIELDS)}")


def color_limits(data: np.ndarray, field: str) -> tuple[float, float]:
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return -1.0, 1.0
    if field in ("vorticity", "u", "v", "pressure"):
        lo, hi = np.percentile(finite, [1, 99])
        if field == "pressure":
            if hi - lo < 1e-12:
                hi = lo + 1e-12
            return float(lo), float(hi)
        a = max(abs(lo), abs(hi), 1e-12)
        if field == "u":
            return float(min(lo, -1e-12)), float(max(hi, 1e-12))
        return -float(a), float(a)
    lo, hi = float(finite.min()), float(finite.max())
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    return lo, hi


class LivePlotter:
    """Live-Ansicht: Strömungsfeld, optional mit Stromlinien, dazu Monitor-Diagramme.

    Parameters
    ----------
    case : Case
    field : str
        Angezeigtes Feld (siehe ``FIELDS``).
    streamlines : bool
        Stromlinien über das Feld zeichnen.
    show : bool
        Fenster öffnen (Live-Anzeige). Bei ``False`` wird nur im Speicher
        gezeichnet, etwa für eine GIF-Aufnahme ohne Bildschirm.
    record : bool
        Jedes Update als Einzelbild für eine Animation speichern.
    max_frames : int
        Höchstzahl gespeicherter Einzelbilder.
    """

    def __init__(self, case, field: str = "speed", streamlines: bool = False,
                 show: bool = True, record: bool = False, max_frames: int = 400):
        import matplotlib.pyplot as plt

        self.plt = plt
        self.case = case
        self.sim = case.sim
        self.field = field
        self.streamlines = streamlines
        self.show = show
        self.record = record
        self.max_frames = max_frames
        self.frames: list[np.ndarray] = []
        self._frame_warned = False
        self._stream_artists = None
        self.closed = False

        g = self.sim.grid
        aspect = g.lx / g.ly
        self.has_forces = self.sim.compute_forces
        fig_w = 11.0
        field_h = min(max(fig_w / aspect * 0.8, 2.6), 6.5)
        fig_h = field_h + 3.0

        if show:
            plt.ion()
        self.fig = plt.figure(figsize=(fig_w, fig_h))
        try:
            self.fig.canvas.manager.set_window_title(f"2D-CFD: {case.title}")
        except Exception:
            pass
        gs = self.fig.add_gridspec(2, 2 if self.has_forces else 1,
                                   height_ratios=[field_h, 2.4], hspace=0.45, wspace=0.28)
        self.ax_field = self.fig.add_subplot(gs[0, :])
        self.ax_res = self.fig.add_subplot(gs[1, 0])
        self.ax_force = self.fig.add_subplot(gs[1, 1]) if self.has_forces else None

        data = field_data(self.sim, field)
        cmap = plt.get_cmap(CMAPS[field]).with_extremes(bad="0.35")
        self.im = self.ax_field.imshow(np.ma.masked_invalid(data.T), origin="lower",
                                       extent=(0, g.lx, 0, g.ly), cmap=cmap,
                                       interpolation="bilinear", aspect="equal")
        self.cbar = self.fig.colorbar(self.im, ax=self.ax_field, fraction=0.035, pad=0.02)
        self.cbar.set_label(FIELDS[field])
        self.ax_field.set_xlabel("x")
        self.ax_field.set_ylabel("y")

        (self.l_res,) = self.ax_res.semilogy([], [], color="C0", label="Residuum |Δu|/Δt")
        (self.l_div,) = self.ax_res.semilogy([], [], color="C3", lw=0.8, label="max |div u|")
        self.ax_res.set_xlabel("Zeit t")
        self.ax_res.set_title("Konvergenz", fontsize=10)
        self.ax_res.grid(True, which="both", alpha=0.3)
        self.ax_res.legend(fontsize=8, loc="upper right")

        if self.has_forces:
            (self.l_cd,) = self.ax_force.plot([], [], color="C0", label="$c_D$")
            self.ax_force2 = self.ax_force.twinx()
            (self.l_cl,) = self.ax_force2.plot([], [], color="C1", label="$c_L$")
            self.ax_force.set_xlabel("Zeit t")
            self.ax_force.set_ylabel("$c_D$", color="C0")
            self.ax_force2.set_ylabel("$c_L$", color="C1")
            self.ax_force.set_title("Kraftbeiwerte", fontsize=10)
            self.ax_force.grid(True, alpha=0.3)

        self.fig.subplots_adjust(left=0.07, right=0.93, top=0.93, bottom=0.08)
        if show:
            plt.show(block=False)
        self.update(self.sim)

    # ------------------------------------------------------------------ #
    def __call__(self, sim):
        return self.update(sim)

    def update(self, sim) -> bool:
        """Aktualisiert alle Diagramme. Gibt ``False`` zurück, wenn das Fenster geschlossen wurde."""
        plt = self.plt
        if self.show and not plt.fignum_exists(self.fig.number):
            self.closed = True
            return False

        g = sim.grid
        data = field_data(sim, self.field)
        self.im.set_data(np.ma.masked_invalid(data.T))
        self.im.set_clim(*color_limits(data, self.field))

        if self.streamlines:
            self._draw_streamlines(sim)

        self.ax_field.set_title(
            f"{self.case.title}  |  Re = {self.case.re:g}  |  t = {sim.t:.3f}  |  "
            f"Schritt {sim.n}", fontsize=10)

        h = sim.history
        if h.t:
            t = np.asarray(h.t)
            res = np.maximum(np.asarray(h.residual), 1e-300)
            div = np.maximum(np.asarray(h.divergence), 1e-300)
            self.l_res.set_data(t, res)
            self.l_div.set_data(t, div)
            self.ax_res.relim()
            self.ax_res.autoscale_view()
            if self.has_forces and h.fx:
                cd, cl = self._coefficients(h)
                self.l_cd.set_data(t, cd)
                self.l_cl.set_data(t, cl)
                for ax, y in ((self.ax_force, cd), (self.ax_force2, cl)):
                    ax.set_xlim(t[0], max(t[-1], t[0] + 1e-9))
                    tail = y[len(y) // 5:] if len(y) > 50 else y
                    lo, hi = float(np.min(tail)), float(np.max(tail))
                    pad = 0.1 * max(hi - lo, 1e-3)
                    ax.set_ylim(lo - pad, hi + pad)

        if self.show:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            plt.pause(0.001)
        if self.record:
            self._grab_frame()
        return True

    def _coefficients(self, h):
        q = 0.5 * self.case.U_ref**2 * self.case.L_ref
        return np.asarray(h.fx) / q, np.asarray(h.fy) / q

    def _draw_streamlines(self, sim):
        g = sim.grid
        if self._stream_artists is not None:
            for art in self._stream_artists:
                try:
                    art.remove()
                except Exception:
                    pass
        uc, vc = sim.cell_velocity()
        uc = np.nan_to_num(uc)
        vc = np.nan_to_num(vc)
        if not (np.any(uc) or np.any(vc)):
            return
        density = (min(3.0, 1.2 * g.lx / g.ly), 1.2) if g.lx > g.ly else (1.2, 1.2)
        ax = self.ax_field
        before = set(map(id, list(ax.collections) + list(ax.patches)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.streamplot(g.xc, g.yc, uc.T, vc.T, color="k", linewidth=0.5,
                          density=density, arrowsize=0.6)
        # alle neu erzeugten Linien und Pfeile merken, damit sie beim nächsten Update entfernt werden
        self._stream_artists = [a for a in list(ax.collections) + list(ax.patches)
                                if id(a) not in before]

    def _grab_frame(self):
        if len(self.frames) >= self.max_frames:
            if not self._frame_warned:
                print(f"Hinweis: Höchstzahl von {self.max_frames} Animationsbildern erreicht.")
                self._frame_warned = True
            return
        self.fig.canvas.draw()
        buf = np.asarray(self.fig.canvas.buffer_rgba())[..., :3].copy()
        self.frames.append(buf)

    def save_animation(self, path, fps: int = 15) -> str | None:
        """Speichert die aufgenommenen Bilder als GIF."""
        if not self.frames:
            return None
        from PIL import Image

        imgs = [Image.fromarray(f) for f in self.frames]
        # Speicher sparen: Bilder auf maximal 900 px Breite verkleinern
        w, h = imgs[0].size
        if w > 900:
            s = 900 / w
            imgs = [im.resize((900, int(h * s)), Image.LANCZOS) for im in imgs]
        imgs = [im.convert("P", palette=Image.ADAPTIVE, colors=255) for im in imgs]
        imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=int(1000 / fps),
                     loop=0, optimize=False)
        return str(path)

    def close(self):
        try:
            self.plt.close(self.fig)
        except Exception:
            pass

    def keep_open(self):
        """Hält das Fenster nach Ende der Simulation offen, bis der Nutzer es schließt."""
        if self.show and self.plt.fignum_exists(self.fig.number):
            self.plt.ioff()
            self.plt.show()
