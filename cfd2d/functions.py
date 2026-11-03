"""Function Objects, Export und Speicherstände (OpenFOAM-Analogie: ``functions`` in controlDict).

* :class:`Probes`: Zeitreihen an Punkten (``probes``)
* :class:`LineSample`: Profile entlang einer Linie (``sample``/``sets``)
* :class:`FieldAverage`: zeitlich gemittelte Felder (``fieldAverage``)
* :class:`VTKWriter`: Felder als VTK-Dateien für ParaView (``foamToVTK``)
* :class:`Checkpoint`: Speicherstände für den Neustart (Zeitverzeichnisse)

Function Objects werden mit ``sim.function_objects.append(fo)`` registriert;
der Löser ruft nach jedem Zeitschritt ``fo.execute(sim)`` auf, der Report am
Ende ``fo.write(outdir)``.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


def sample_field(sim, field: str, x, y) -> np.ndarray:
    """Interpoliert ein Feld (Grundfeld oder Modellfeld) an beliebigen Punkten."""
    from .visualization import field_data

    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    if field == "u":
        return sim.sample_u(x, y)
    if field == "v":
        return sim.sample_v(x, y)
    g = sim.grid
    data = field_data(sim, field)
    f = RegularGridInterpolator((g.xc, g.yc), data, bounds_error=False, fill_value=None)
    xs = np.clip(x, g.xc[0], g.xc[-1])
    ys = np.clip(y, g.yc[0], g.yc[-1])
    return f(np.column_stack([xs, ys]))


class FunctionObject:
    name = "function"

    def execute(self, sim) -> None:
        pass

    def write(self, outdir: Path) -> list[Path]:
        return []

    def describe(self) -> str:
        return self.name

    def spec(self) -> dict:
        return {}


# ---------------------------------------------------------------------- #
class Probes(FunctionObject):
    """Zeitreihen ausgewählter Felder an festen Punkten."""

    def __init__(self, points, fields=("u", "v", "pressure"), every: int = 1):
        self.points = [tuple(map(float, p)) for p in points]
        self.fields = list(fields)
        self.every = max(1, int(every))
        self.t: list = []
        self.data: dict = {f: [] for f in self.fields}
        self.name = "probes"

    def execute(self, sim) -> None:
        if sim.n % self.every:
            return
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        self.t.append(sim.t)
        for f in self.fields:
            self.data[f].append(sample_field(sim, f, xs, ys))

    def write(self, outdir: Path) -> list[Path]:
        if not self.t:
            return []
        outdir = Path(outdir)
        path = outdir / "sonden.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(["t"] + [f"{f}@({x:g},{y:g})" for f in self.fields for (x, y) in self.points])
            arrs = {f: np.asarray(self.data[f]) for f in self.fields}
            for k, t in enumerate(self.t):
                w.writerow([f"{t:.6g}"] + [f"{arrs[f][k, i]:.6g}" for f in self.fields
                                           for i in range(len(self.points))])
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(len(self.fields), 1, figsize=(10, 2.6 * len(self.fields)),
                                 squeeze=False)
        for ax, f in zip(axes[:, 0], self.fields):
            arr = np.asarray(self.data[f])
            for i, (x, y) in enumerate(self.points):
                ax.plot(self.t, arr[:, i], label=f"({x:g}, {y:g})")
            ax.set_ylabel(f)
            ax.grid(True, alpha=0.3)
        axes[0, 0].legend(fontsize=8, ncol=min(4, len(self.points)))
        axes[-1, 0].set_xlabel("Zeit t")
        fig.suptitle("Sonden")
        fig.tight_layout()
        fig.savefig(outdir / "sonden.png", dpi=120)
        plt.close(fig)
        return [path, outdir / "sonden.png"]

    def describe(self) -> str:
        return f"Sonden an {len(self.points)} Punkten: {', '.join(self.fields)}"

    def spec(self) -> dict:
        return {"typ": "sonden", "punkte": self.points, "felder": self.fields, "alle": self.every}


# ---------------------------------------------------------------------- #
class LineSample(FunctionObject):
    """Profil entlang der Strecke p0 -> p1 am Ende der Rechnung."""

    def __init__(self, p0, p1, n: int = 100, fields=("u", "v", "pressure"), label: str = "linie"):
        self.p0 = tuple(map(float, p0))
        self.p1 = tuple(map(float, p1))
        self.n = int(n)
        self.fields = list(fields)
        self.label = label
        self.name = "linie"
        self.result = None
        self.sim = None

    def execute(self, sim) -> None:
        self.sim = sim

    def sample(self, sim) -> dict:
        s = np.linspace(0, 1, self.n)
        x = self.p0[0] + s * (self.p1[0] - self.p0[0])
        y = self.p0[1] + s * (self.p1[1] - self.p0[1])
        dist = s * np.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1])
        out = {"s": dist, "x": x, "y": y}
        for f in self.fields:
            out[f] = sample_field(sim, f, x, y)
        return out

    def write(self, outdir: Path) -> list[Path]:
        if self.sim is None:
            return []
        import matplotlib.pyplot as plt

        outdir = Path(outdir)
        d = self.sample(self.sim)
        self.result = d
        path = outdir / f"linie_{self.label}.csv"
        keys = ["s", "x", "y"] + self.fields
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, delimiter=";")
            w.writerow(keys)
            for k in range(self.n):
                w.writerow([f"{d[c][k]:.6g}" for c in keys])
        fig, axes = plt.subplots(1, len(self.fields), figsize=(4.5 * len(self.fields), 3.8),
                                 squeeze=False)
        for ax, f in zip(axes[0], self.fields):
            ax.plot(d["s"], d[f])
            ax.set_xlabel("Bogenlänge s")
            ax.set_title(f)
            ax.grid(True, alpha=0.3)
        fig.suptitle(f"Linienprobe {self.label}: {self.p0} → {self.p1}")
        fig.tight_layout()
        fig.savefig(outdir / f"linie_{self.label}.png", dpi=120)
        plt.close(fig)
        return [path]

    def describe(self) -> str:
        return f"Linienprobe {self.p0} → {self.p1} ({self.n} Punkte): {', '.join(self.fields)}"

    def spec(self) -> dict:
        return {"typ": "linie", "p0": self.p0, "p1": self.p1, "n": self.n, "felder": self.fields,
                "name": self.label}


# ---------------------------------------------------------------------- #
class FieldAverage(FunctionObject):
    """Zeitliche Mittelwerte und Schwankungen (RMS) von u, v, p ab einer Startzeit."""

    def __init__(self, t_start: float = 0.0):
        self.t_start = float(t_start)
        self.name = "mittelwerte"
        self.T = 0.0
        self.sum = None

    def execute(self, sim) -> None:
        if sim.t < self.t_start:
            return
        uc, vc = sim.cell_velocity()
        p = sim.cell_pressure()
        dt = sim.dt
        vals = {"u": uc, "v": vc, "p": p}
        if self.sum is None:
            self.sum = {k: np.zeros_like(a) for k, a in vals.items()}
            self.sum2 = {k: np.zeros_like(a) for k, a in vals.items()}
        for k, a in vals.items():
            self.sum[k] += a * dt
            self.sum2[k] += a * a * dt
        self.T += dt
        self.sim = sim

    def mean(self, k: str) -> np.ndarray:
        return self.sum[k] / max(self.T, 1e-300)

    def rms(self, k: str) -> np.ndarray:
        m = self.mean(k)
        return np.sqrt(np.maximum(self.sum2[k] / max(self.T, 1e-300) - m * m, 0.0))

    def write(self, outdir: Path) -> list[Path]:
        if self.sum is None or self.T <= 0:
            return []
        import matplotlib.pyplot as plt

        outdir = Path(outdir)
        g = self.sim.grid
        np.savez_compressed(outdir / "mittelwerte.npz", T=self.T,
                            **{f"{k}_mittel": self.mean(k) for k in self.sum},
                            **{f"{k}_rms": self.rms(k) for k in self.sum})
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        for ax, (k, kind) in zip(axes.ravel(), [("u", "mittel"), ("v", "mittel"), ("u", "rms"),
                                                ("v", "rms")]):
            d = self.mean(k) if kind == "mittel" else self.rms(k)
            im = ax.imshow(np.ma.masked_invalid(d).T, origin="lower", extent=(0, g.lx, 0, g.ly),
                           cmap="viridis" if kind == "rms" else "RdBu_r", aspect="equal")
            plt.colorbar(im, ax=ax, fraction=0.03)
            ax.set_title(f"{k} {'Mittelwert' if kind == 'mittel' else 'RMS-Schwankung'}")
        fig.suptitle(f"Zeitmittel über T = {self.T:.3g} (ab t = {self.t_start:g})")
        fig.tight_layout()
        fig.savefig(outdir / "mittelwerte.png", dpi=120)
        plt.close(fig)
        return [outdir / "mittelwerte.npz"]

    def describe(self) -> str:
        return f"Zeitmittelung ab t = {self.t_start:g}"

    def spec(self) -> dict:
        return {"typ": "mittelwerte", "t_start": self.t_start}


# ---------------------------------------------------------------------- #
def write_vtk(sim, path) -> Path:
    """Schreibt alle Felder als VTK-Datei (Legacy, binär, RECTILINEAR_GRID) für ParaView."""
    from .visualization import available_fields, field_data

    g = sim.grid
    path = Path(path)
    nx, ny = g.nx, g.ny
    with open(path, "wb") as fh:
        def line(s):
            fh.write((s + "\n").encode("ascii"))

        line("# vtk DataFile Version 3.0")
        line(f"cfd2d t={sim.t:.8g}")
        line("BINARY")
        line("DATASET RECTILINEAR_GRID")
        line(f"DIMENSIONS {nx + 1} {ny + 1} 1")
        line(f"X_COORDINATES {nx + 1} float")
        fh.write(g.xn.astype(">f4").tobytes())
        line("")
        line(f"Y_COORDINATES {ny + 1} float")
        fh.write(g.yn.astype(">f4").tobytes())
        line("")
        line("Z_COORDINATES 1 float")
        fh.write(np.zeros(1, ">f4").tobytes())
        line("")
        line(f"CELL_DATA {nx * ny}")
        uc, vc = sim.cell_velocity()
        vec = np.stack([np.nan_to_num(uc).T.ravel(), np.nan_to_num(vc).T.ravel(),
                        np.zeros(nx * ny)], axis=1)
        line("VECTORS U float")
        fh.write(vec.astype(">f4").tobytes())
        line("")
        names = [k for k in available_fields(sim) if k not in ("u", "v")]
        for k in names + ["fluid"]:
            if k == "fluid":
                d = g.fluid_interior.astype(float)
            else:
                d = field_data(sim, k)
            safe = k.replace(" ", "_")
            line(f"SCALARS {safe} float 1")
            line("LOOKUP_TABLE default")
            fh.write(np.nan_to_num(d, nan=0.0).T.astype(">f4").ravel().tobytes())
            line("")
    return path


class VTKWriter(FunctionObject):
    """Schreibt in festen Zeitabständen VTK-Dateien und eine ParaView-Serie (.vtk.series)."""

    def __init__(self, interval: float, outdir):
        self.interval = float(interval)
        self.outdir = Path(outdir) / "vtk"
        self.next_t = 0.0
        self.files: list = []
        self.name = "vtk"

    def execute(self, sim) -> None:
        if sim.t + 1e-12 >= self.next_t:
            self.outdir.mkdir(parents=True, exist_ok=True)
            f = write_vtk(sim, self.outdir / f"feld_{len(self.files):05d}.vtk")
            self.files.append((sim.t, f.name))
            self.next_t = sim.t + self.interval

    def write(self, outdir: Path) -> list[Path]:
        if not self.files:
            return []
        series = {"file-series-version": "1.0",
                  "files": [{"name": n, "time": t} for t, n in self.files]}
        p = self.outdir / "feld.vtk.series"
        p.write_text(json.dumps(series, indent=1), encoding="utf-8")
        return [p]

    def describe(self) -> str:
        return f"VTK-Export alle Δt = {self.interval:g} nach {self.outdir}"

    def spec(self) -> dict:
        return {"typ": "vtk", "intervall": self.interval}


# ---------------------------------------------------------------------- #
# Speicherstände
# ---------------------------------------------------------------------- #
def save_checkpoint(sim, path) -> Path:
    """Speichert den vollständigen Zustand (Felder, Zeit, Modelle, Zeitverläufe)."""
    path = Path(path)
    data = {"u": sim.u, "v": sim.v, "p": sim.p, "t": sim.t, "n": sim.n, "dt": sim.dt,
            "flow_force": sim.flow_force, "dt_old": sim.dt_old if sim.dt_old else 0.0,
            "fluid": sim.grid.fluid}
    if sim.hu_old is not None:
        data["hu_old"] = sim.hu_old
        data["hv_old"] = sim.hv_old
    for name, f in sim.scalars.items():
        data[f"skalar_{name}"] = f.phi
    for k, m in enumerate(sim.models):
        for key, arr in (m.state() if hasattr(m, "state") else {}).items():
            data[f"modell{k}_{key}"] = arr
    hist = sim.history
    data["history_json"] = json.dumps({k: list(map(float, v)) for k, v in hist.__dict__.items()
                                       if k != "extra"} | {"extra": {k: list(map(float, v))
                                                                    for k, v in hist.extra.items()}})
    np.savez_compressed(path, **data)
    return path


def load_checkpoint(sim, path) -> None:
    """Lädt einen Speicherstand in eine gleich aufgebaute Simulation."""
    d = np.load(Path(path), allow_pickle=False)
    if d["u"].shape != sim.u.shape or not np.array_equal(d["fluid"], sim.grid.fluid):
        raise ValueError("Der Speicherstand passt nicht zum Gitter bzw. zur Geometrie dieses Falls.")
    sim.u[:] = d["u"]
    sim.v[:] = d["v"]
    sim.p[:] = d["p"]
    sim.t = float(d["t"])
    sim.n = int(d["n"])
    sim.dt = float(d["dt"])
    sim.flow_force = float(d["flow_force"])
    sim.dt_old = float(d["dt_old"]) or None
    if "hu_old" in d and sim.scheme == "ab2":
        sim.hu_old = d["hu_old"].copy()
        sim.hv_old = d["hv_old"].copy()
    for name, f in sim.scalars.items():
        key = f"skalar_{name}"
        if key in d:
            f.phi[:] = d[key]
            f.fill_ghosts()
    for k, m in enumerate(sim.models):
        if hasattr(m, "set_state"):
            st = {key.split("_", 1)[1]: d[key] for key in d.files if key.startswith(f"modell{k}_")}
            if st:
                m.set_state(sim, st)
    h = json.loads(str(d["history_json"]))
    extra = h.pop("extra", {})
    for k, v in h.items():
        setattr(sim.history, k, v)
    sim.history.extra = extra
    sim._apply_bc()


class Checkpoint(FunctionObject):
    """Schreibt in festen Zeitabständen einen Speicherstand (``checkpoint.npz``, überschrieben)."""

    def __init__(self, interval: float, outdir):
        self.interval = float(interval)
        self.path = Path(outdir) / "checkpoint.npz"
        self.next_t = self.interval
        self.name = "checkpoint"
        self.sim = None

    def execute(self, sim) -> None:
        self.sim = sim
        if sim.t + 1e-12 >= self.next_t:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(sim, self.path)
            self.next_t = sim.t + self.interval

    def write(self, outdir: Path) -> list[Path]:
        if self.sim is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            save_checkpoint(self.sim, self.path)
            return [self.path]
        return []

    def describe(self) -> str:
        return f"Speicherstand alle Δt = {self.interval:g}"

    def spec(self) -> dict:
        return {"typ": "checkpoint", "intervall": self.interval}


def build_function_objects(specs: list[dict], outdir) -> list[FunctionObject]:
    out = []
    for s in specs or []:
        t = s["typ"]
        if t == "sonden":
            out.append(Probes(s["punkte"], s.get("felder", ("u", "v", "pressure")), s.get("alle", 1)))
        elif t == "linie":
            out.append(LineSample(s["p0"], s["p1"], s.get("n", 100), s.get("felder", ("u", "v")),
                                  s.get("name", "linie")))
        elif t == "mittelwerte":
            out.append(FieldAverage(s.get("t_start", 0.0)))
        elif t == "vtk":
            out.append(VTKWriter(s["intervall"], outdir))
        elif t == "checkpoint":
            out.append(Checkpoint(s["intervall"], outdir))
        else:
            raise ValueError(f"Unbekanntes Function Object {t!r}")
    return out
