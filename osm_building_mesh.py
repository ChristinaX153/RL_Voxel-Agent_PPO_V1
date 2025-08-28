"""
osm_building_mesh.py

Fetch OSM building outlines around a coordinate, rasterize into a 2D grid, and
estimate wind speeds over the grid via a lightweight 2D LBM CFD simulation.

Outputs: PNG heatmap, NPY array, and CSV of wind speed magnitude (m/s).

Dependencies:
- osmnx
- shapely
- numpy
- matplotlib (optional, for visualization)

Example:
    # 6 m/s wind towards +x (0°)
    python osm_building_mesh.py --lat -37.81361 --lon 144.96332 --radius 150 --wind 6.0 \
        --dir-deg 0 --grid 200 --steps 3000 --out-prefix melbourne_cbd
    # 6 m/s wind towards +y (90°)
    python osm_building_mesh.py --lat -37.81361 --lon 144.96332 --radius 150 --wind 6.0 \
        --dir-deg 90 --grid 200 --steps 3000 --out-prefix melbourne_cbd
"""
import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Tuple, cast

import numpy as np
import osmnx as ox
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import urllib3


# OSMnx tweaks (avoid SSL verification hiccups on some systems)
ox.settings.use_cache = False
ox.settings.requests_kwargs = {"verify": False}
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ------------------------------ Data Types ------------------------------


@dataclass
class Domain:
    """Continuous domain in projected meters and its rasterization parameters."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float
    nx: int
    ny: int

    @property
    def dx(self) -> float:
        return (self.xmax - self.xmin) / self.nx

    @property
    def dy(self) -> float:
        return (self.ymax - self.ymin) / self.ny


# ------------------------------ OSM Fetch ------------------------------


def fetch_buildings_projected(lat: float, lon: float, radius_m: float):
    """
    Fetch building polygons from OSM around a point and project to a local meter CRS.

    Returns a tuple (polygons, bounds), where polygons is a list[Polygon] in meters,
    and bounds is (xmin, ymin, xmax, ymax) in meters of a square box with side = 2*radius_m.
    """
    # Fetch raw geometries (likely EPSG:4326)
    gdf = ox.features_from_point((lat, lon), tags={"building": True}, dist=radius_m)
    if gdf.empty:
        return [], None

    # Keep only polygonal geometries
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        return [], None

    # Project to a suitable metric CRS for the given location
    gdf_proj = ox.projection.project_gdf(gdf)

    # Project the user-specified center point to the same CRS using OSMnx
    pt = Point(lon, lat)
    center_proj, _ = ox.projection.project_geometry(pt, to_crs=gdf_proj.crs)
    center_pt = cast(Point, center_proj)
    cx, cy = float(center_pt.x), float(center_pt.y)

    # Include building footprints that intersect the radius circle (not just fully within)
    circle = Point(cx, cy).buffer(radius_m)
    polygons_all: list[Polygon] = []
    for geom in gdf_proj.geometry:
        if geom.is_empty:
            continue
        # Include geometries that intersect the circle
        if isinstance(geom, Polygon):
            if geom.intersects(circle):
                polygons_all.append(geom)
        elif isinstance(geom, MultiPolygon):
            for sub in geom.geoms:
                if not sub.is_empty and sub.intersects(circle):
                    polygons_all.append(sub)

    # Domain bounds: add additional empty space outside the radius by 2*radius (total half-extent = 3*radius)
    half_extent = 3.0 * radius_m
    xmin = cx - half_extent
    xmax = cx + half_extent
    ymin = cy - half_extent
    ymax = cy + half_extent

    return polygons_all, (xmin, ymin, xmax, ymax)


# ------------------------------ Rasterization ------------------------------


def rasterize_buildings(polygons: Iterable[Polygon], dom: Domain) -> np.ndarray:
    """
    Rasterize building polygons into an obstacle mask (True=obstacle) of shape (ny, nx).
    Uses cell-center point-in-polygon tests. Adequate for moderate grid sizes.
    """
    if not polygons:
        return np.zeros((dom.ny, dom.nx), dtype=bool)

    # Merge for faster prepared contains
    merged = unary_union([p.buffer(0) for p in polygons])  # buffer(0) fixes minor invalidities
    prep_geom = prep(merged)

    # Build grid of cell centers
    xs = np.linspace(dom.xmin + 0.5 * dom.dx, dom.xmax - 0.5 * dom.dx, dom.nx)
    ys = np.linspace(dom.ymin + 0.5 * dom.dy, dom.ymax - 0.5 * dom.dy, dom.ny)
    XX, YY = np.meshgrid(xs, ys)

    # Vectorized contains via flatten loop (keeps memory manageable)
    mask = np.zeros_like(XX, dtype=bool)
    flat_pts = [Point(x, y) for x, y in zip(XX.ravel(), YY.ravel())]
    # Batch in chunks to reduce Python overhead
    chunk = 20000
    for i in range(0, len(flat_pts), chunk):
        batch = flat_pts[i : i + chunk]
        mask.ravel()[i : i + len(batch)] = np.fromiter((prep_geom.contains(pt) for pt in batch), dtype=bool)
    return mask


# ------------------------------ 2D LBM CFD ------------------------------


class LBM2D:
    """Minimal D2Q9 LBM solver with bounce-back obstacles and vector wind inlet.

    Lattice units. Keep |u_in| ≲ 0.1 for stability.
    """

    # D2Q9 parameters
    cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    weights = np.array([4 / 9] + [1 / 9] * 4 + [1 / 36] * 4)

    def __init__(self, nx: int, ny: int, omega: float, obstacle: np.ndarray, u_in_x: float, u_in_y: float):
        self.nx, self.ny = nx, ny
        self.omega = omega
        self.obstacle = obstacle.astype(bool)
        self.uin_x = float(u_in_x)
        self.uin_y = float(u_in_y)

        # Fields
        self.f = np.zeros((9, ny, nx), dtype=np.float32)
        self.rho = np.ones((ny, nx), dtype=np.float32)
        self.ux = np.zeros((ny, nx), dtype=np.float32)
        self.uy = np.zeros((ny, nx), dtype=np.float32)

        # Initialize with uniform flow
        self.ux[:, :] = self.uin_x
        self.uy[:, :] = self.uin_y
        self.f[:] = self.equilibrium(self.rho, self.ux, self.uy)

        # Opposite directions for bounce-back
        self.opposite = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])

        # Stability helpers
        self.umax = 0.12  # max lattice speed magnitude for equilibrium
        self.rho_min = 1e-6

    def _clip_velocity(self, ux, uy):
        speed2 = ux**2 + uy**2
        mask = speed2 > (self.umax**2)
        if np.any(mask):
            scale = self.umax / np.sqrt(speed2[mask])
            ux = ux.copy(); uy = uy.copy()
            ux[mask] *= scale
            uy[mask] *= scale
        return ux, uy

    @classmethod
    def equilibrium(cls, rho, ux, uy):
        cu = (
            np.einsum("i,yx->iyx", cls.cxs, ux)
            + np.einsum("i,yx->iyx", cls.cys, uy)
        )
        usq = 1.5 * (ux**2 + uy**2)
        feq = np.einsum("i,yx->iyx", cls.weights, rho) * (1 + 3 * cu + 4.5 * cu**2 - usq)
        return feq.astype(np.float32)

    def collide(self):
        # Clip velocities before equilibrium to control Mach number
        ux_c, uy_c = self._clip_velocity(self.ux, self.uy)
        feq = self.equilibrium(self.rho, ux_c, uy_c)
        self.f += self.omega * (feq - self.f)
        # Sanitize distributions
        np.nan_to_num(self.f, copy=False, nan=0.0, posinf=1e6, neginf=0.0)
        self.f = np.clip(self.f, 0.0, 1e6)

    def stream(self):
        for i, (cx, cy) in enumerate(zip(self.cxs, self.cys)):
            self.f[i] = np.roll(self.f[i], shift=cx, axis=1)
            self.f[i] = np.roll(self.f[i], shift=cy, axis=0)

    def apply_boundaries(self):
        # X-major vs Y-major flow
        if abs(self.uin_x) >= abs(self.uin_y):
            if self.uin_x >= 0:
                # Inlet at left (x=0)
                rho_in = self.rho[:, [1]]
                uxin, uyin = self._clip_velocity(np.full_like(rho_in, self.uin_x), np.full_like(rho_in, self.uin_y))
                feq_in = self.equilibrium(
                    rho_in,
                    uxin,
                    uyin,
                )
                self.f[:, :, 0] = feq_in[:, :, 0]
                # Outlet at right (copy)
                self.f[:, :, -1] = self.f[:, :, -2]
            else:
                # Inlet at right (x=-1)
                rho_in = self.rho[:, [-2]]
                uxin, uyin = self._clip_velocity(np.full_like(rho_in, self.uin_x), np.full_like(rho_in, self.uin_y))
                feq_in = self.equilibrium(
                    rho_in,
                    uxin,
                    uyin,
                )
                self.f[:, :, -1] = feq_in[:, :, 0]
                # Outlet at left (copy)
                self.f[:, :, 0] = self.f[:, :, 1]

            # Periodic top/bottom
            self.f[:, 0, :] = self.f[:, -2, :]
            self.f[:, -1, :] = self.f[:, 1, :]
        else:
            if self.uin_y >= 0:
                # Inlet at bottom (y=0)
                rho_in = self.rho[[1], :]
                uxin, uyin = self._clip_velocity(np.full_like(rho_in, self.uin_x), np.full_like(rho_in, self.uin_y))
                feq_in = self.equilibrium(
                    rho_in,
                    uxin,
                    uyin,
                )
                self.f[:, 0, :] = feq_in[:, 0, :]
                # Outlet at top (copy)
                self.f[:, -1, :] = self.f[:, -2, :]
            else:
                # Inlet at top (y=-1)
                rho_in = self.rho[[-2], :]
                uxin, uyin = self._clip_velocity(np.full_like(rho_in, self.uin_x), np.full_like(rho_in, self.uin_y))
                feq_in = self.equilibrium(
                    rho_in,
                    uxin,
                    uyin,
                )
                self.f[:, -1, :] = feq_in[:, 0, :]
                # Outlet at bottom (copy)
                self.f[:, 0, :] = self.f[:, 1, :]

            # Periodic left/right
            self.f[:, :, 0] = self.f[:, :, -2]
            self.f[:, :, -1] = self.f[:, :, 1]

        # Bounce-back for obstacles
        obs = self.obstacle
        if obs.any():
            for i in range(9):
                inv = self.opposite[i]
                f_i = self.f[i]
                f_inv = self.f[inv]
                tmp = f_i[obs].copy()
                f_i[obs] = f_inv[obs]
                f_inv[obs] = tmp

    def macroscopic(self):
        rho = np.sum(self.f, axis=0)
        rho_safe = np.maximum(rho, self.rho_min)
        ux = (self.f[1] - self.f[3] + self.f[5] - self.f[6] - self.f[7] + self.f[8]) / rho_safe
        uy = (self.f[2] - self.f[4] + self.f[5] + self.f[6] - self.f[7] - self.f[8]) / rho_safe
        # Clip for reporting stability
        ux, uy = self._clip_velocity(ux, uy)
        ux[self.obstacle] = 0.0
        uy[self.obstacle] = 0.0
        self.rho, self.ux, self.uy = rho, ux, uy
        return rho, ux, uy

    def step(self):
        self.collide()
        self.stream()
        self.apply_boundaries()
        return self.macroscopic()


# ------------------------------ Visualization ------------------------------

def _upsample_bilinear_masked(a: np.ndarray, obstacle: np.ndarray, scale: int) -> np.ndarray:
    """Upsample array a by integer factor using bilinear interpolation while avoiding
    smoothing across building cells. Obstacle (True) cells do not contribute to
    interpolation; if all neighbors are obstacles, fallback to nearest.

    This is used for prettier plots only; it does not change saved NPY/CSV data.
    """
    if scale <= 1:
        return a
    ny, nx = a.shape
    ys = np.linspace(0, ny - 1, ny * scale)
    xs = np.linspace(0, nx - 1, nx * scale)
    YY, XX = np.meshgrid(ys, xs, indexing="ij")
    y0 = np.floor(YY).astype(int)
    x0 = np.floor(XX).astype(int)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    dy = (YY - y0).astype(a.dtype)
    dx = (XX - x0).astype(a.dtype)
    wy0 = 1.0 - dy; wy1 = dy
    wx0 = 1.0 - dx; wx1 = dx
    # Gather neighbor values and masks (non-obstacle contributes)
    free = ~obstacle
    v00 = a[y0, x0]; m00 = free[y0, x0]
    v10 = a[y1, x0]; m10 = free[y1, x0]
    v01 = a[y0, x1]; m01 = free[y0, x1]
    v11 = a[y1, x1]; m11 = free[y1, x1]
    w00 = (wy0 * wx0) * m00
    w10 = (wy1 * wx0) * m10
    w01 = (wy0 * wx1) * m01
    w11 = (wy1 * wx1) * m11
    wsum = w00 + w10 + w01 + w11
    out = np.empty_like(YY, dtype=a.dtype)
    valid = wsum > 0
    out[valid] = (w00[valid] * v00[valid] + w10[valid] * v10[valid] + w01[valid] * v01[valid] + w11[valid] * v11[valid]) / wsum[valid]
    out[~valid] = a[y0[~valid], x0[~valid]]
    return out


def save_wind_outputs(u_mps: np.ndarray, ux_mps: np.ndarray, uy_mps: np.ndarray, dom: Domain, out_prefix: str, radius_m: float, polygons: list[Polygon], obstacle: np.ndarray, upsample: int):
    os.makedirs(os.path.dirname(out_prefix) or ".", exist_ok=True)

    # Save NPY
    np.save(f"{out_prefix}_wind_speed.npy", u_mps)
    np.save(f"{out_prefix}_ux.npy", ux_mps)
    np.save(f"{out_prefix}_uy.npy", uy_mps)

    # Save CSV (y,x,value) for easy consumption
    ny, nx = u_mps.shape
    ys, xs = np.indices((ny, nx))
    flat = np.column_stack([ys.ravel(), xs.ravel(), u_mps.ravel(), ux_mps.ravel(), uy_mps.ravel()])
    np.savetxt(
        f"{out_prefix}_wind.csv",
        flat,
        delimiter=",",
        header="y,x,speed_mps,ux_mps,uy_mps",
        comments="",
    )

    # Prepare physical coordinates
    ny, nx = u_mps.shape
    xs = np.linspace(dom.xmin + 0.5 * dom.dx, dom.xmax - 0.5 * dom.dx, nx)
    ys = np.linspace(dom.ymin + 0.5 * dom.dy, dom.ymax - 0.5 * dom.dy, ny)
    XX, YY = np.meshgrid(xs, ys)
    cx = 0.5 * (dom.xmin + dom.xmax)
    cy = 0.5 * (dom.ymin + dom.ymax)

    # Compute colormap range from free-space wind speeds (ignore buildings)
    try:
        free_mask = ~obstacle if obstacle is not None else np.ones_like(u_mps, dtype=bool)
    except Exception:
        free_mask = np.ones_like(u_mps, dtype=bool)
    vals = u_mps[free_mask]
    if vals.size == 0:
        vals = u_mps.ravel()
    vmin = float(np.nanmin(vals)) if vals.size else 0.0
    vmax = float(np.nanmax(vals)) if vals.size else 1.0
    # Avoid degenerate range
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax):
        vmax = 1.0
    if vmax - vmin < 1e-9:
        vmax = vmin + 1e-6

    # Prettier plots: optional masked upsampling (keeps NPY/CSV unchanged)
    plot_u = _upsample_bilinear_masked(u_mps, obstacle, max(1, int(upsample)))
    plot_ux = _upsample_bilinear_masked(ux_mps, obstacle, max(1, int(upsample)))
    plot_uy = _upsample_bilinear_masked(uy_mps, obstacle, max(1, int(upsample)))
    if upsample and upsample > 1:
        xs_hi = np.linspace(dom.xmin + 0.5 * dom.dx, dom.xmax - 0.5 * dom.dx, plot_u.shape[1])
        ys_hi = np.linspace(dom.ymin + 0.5 * dom.dy, dom.ymax - 0.5 * dom.dy, plot_u.shape[0])
        XX_hi, YY_hi = np.meshgrid(xs_hi, ys_hi)
    else:
        XX_hi, YY_hi = XX, YY

    # Save PNG heatmap (no mask), with physical coords and zoomed to ROI
    plt.figure(figsize=(6, 6))
    x0, x1 = dom.xmin + 0.5 * dom.dx, dom.xmax - 0.5 * dom.dx
    y0, y1 = dom.ymin + 0.5 * dom.dy, dom.ymax - 0.5 * dom.dy
    im = plt.imshow(
        plot_u,
        origin="lower",
        cmap="viridis",
        extent=(x0, x1, y0, y1),
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    # Overlay building footprints (outlines)
    ax = plt.gca()
    for poly in polygons:
        try:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="black", linewidth=0.8, alpha=0.9)
            # Optionally draw holes (interiors)
            for interior in poly.interiors:
                xi, yi = interior.xy
                ax.plot(xi, yi, color="black", linewidth=0.5, alpha=0.6)
        except Exception:
            # Robust against any invalid geometry edge-cases
            continue
    plt.colorbar(im, label="Wind speed (m/s)")
    plt.title("Estimated near-ground wind speed")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim(cx - radius_m, cx + radius_m)
    plt.ylim(cy - radius_m, cy + radius_m)
    # Cleaner tick formatting (whole meters, fewer decimals)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_wind_speed.png", dpi=200)
    plt.close()

    # Optional: quiver plot of downsampled vectors
    step = max(1, min(plot_u.shape) // 25)
    # Downsampled physical coordinates
    X_phys = XX_hi[::step, ::step]
    Y_phys = YY_hi[::step, ::step]
    plt.figure(figsize=(6, 6))
    plt.imshow(
        plot_u,
        origin="lower",
        cmap="Greys",
        alpha=0.4,
        extent=(x0, x1, y0, y1),
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )
    # Overlay building footprints (outlines)
    ax = plt.gca()
    for poly in polygons:
        try:
            x, y = poly.exterior.xy
            ax.plot(x, y, color="black", linewidth=0.8, alpha=0.9)
            for interior in poly.interiors:
                xi, yi = interior.xy
                ax.plot(xi, yi, color="black", linewidth=0.5, alpha=0.6)
        except Exception:
            continue
    plt.quiver(
        X_phys,
        Y_phys,
        plot_ux[::step, ::step],
        plot_uy[::step, ::step],
        color="tab:blue",
        angles="xy",
        scale_units="xy",
        scale=50,
    )
    plt.title("Wind vectors (downsampled)")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.xlim(cx - radius_m, cx + radius_m)
    plt.ylim(cy - radius_m, cy + radius_m)
    # Cleaner tick formatting
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_wind_vectors.png", dpi=200)
    plt.close()


# ------------------------------ Main ------------------------------


def main():
    parser = argparse.ArgumentParser(description="OSM 2D wind estimation (LBM) around a location")
    parser.add_argument("--lat", type=float, required=True, help="Latitude")
    parser.add_argument("--lon", type=float, required=True, help="Longitude")
    parser.add_argument("--radius", type=float, default=100.0, help="Radius in meters (default: 100)")
    parser.add_argument("--wind", type=float, required=True, help="Inlet wind speed in m/s (e.g., 5.0)")
    parser.add_argument("--grid", type=int, default=200, help="Grid resolution (nx==ny). Default 200")
    parser.add_argument(
        "--dir-deg",
        type=float,
        default=0.0,
        help="Wind direction in degrees towards which it blows (0=+x, 90=+y, CCW)",
    )
    parser.add_argument("--steps", type=int, default=3000, help="LBM steps to run. Default 3000")
    parser.add_argument(
        "--upsample",
        type=int,
        default=1,
        help="Integer upsampling factor for plots only (does not affect NPY/CSV). Use 1 to disable.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="osm_wind",
        help="Output prefix for files (PNG/NPY/CSV)",
    )
    args = parser.parse_args()

    print(f"Fetching OSM buildings around ({args.lat}, {args.lon}) within {args.radius} m…")
    polygons, bounds = fetch_buildings_projected(args.lat, args.lon, args.radius)
    if bounds is None:
        print("No buildings found or fetch failed. Proceeding with empty domain.")
        # Build a synthetic domain around the point
        xmin = -args.radius
        ymin = -args.radius
        xmax = args.radius
        ymax = args.radius
    else:
        xmin, ymin, xmax, ymax = bounds

    dom = Domain(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, nx=args.grid, ny=args.grid)

    print(f"Rasterizing buildings onto a {dom.nx}x{dom.ny} grid…")
    obstacle = rasterize_buildings(polygons, dom)
    building_cells = int(obstacle.sum())
    print(f"Obstacle cells: {building_cells} ({building_cells / (dom.nx*dom.ny):.1%} coverage)")

    # LBM setup
    U_in_lbm = 0.08  # lattice units (keep low for stability)
    tau = 0.8  # relaxation time (higher tau -> lower omega, more stable)
    omega = 1.0 / tau
    # Inlet vector components per direction
    theta = np.deg2rad(args.dir_deg)
    uin_x = U_in_lbm * np.cos(theta)
    uin_y = U_in_lbm * np.sin(theta)
    lbm = LBM2D(nx=dom.nx, ny=dom.ny, omega=omega, obstacle=obstacle, u_in_x=uin_x, u_in_y=uin_y)

    print(
        f"Running 2D LBM for {args.steps} steps… (U_in={args.wind} m/s @ {args.dir_deg}° → u=({np.cos(theta):.2f},{np.sin(theta):.2f}))"
    )
    for it in range(args.steps):
        lbm.step()
        if (it + 1) % max(1, args.steps // 10) == 0:
            umax = float(np.sqrt(lbm.ux**2 + lbm.uy**2).max())
            print(f"  step {it+1}/{args.steps} | max|u| (lattice) ~ {umax:.4f}")

    # Convert lattice speeds to m/s by linear scaling with inlet speed
    u_lattice = np.sqrt(lbm.ux**2 + lbm.uy**2)
    scale = args.wind / U_in_lbm if U_in_lbm != 0 else 0.0
    u_mps = u_lattice * scale
    ux_mps = lbm.ux * scale
    uy_mps = lbm.uy * scale

    print(f"Saving outputs with prefix '{args.out_prefix}'…")
    save_wind_outputs(
        u_mps,
        ux_mps,
        uy_mps,
        dom,
        args.out_prefix,
        radius_m=args.radius,
        polygons=polygons,
        obstacle=obstacle,
        upsample=args.upsample,
    )
    print("Done.")


if __name__ == "__main__":
    main()
