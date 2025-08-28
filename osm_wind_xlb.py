"""
osm_wind_xlb.py

GPU-accelerated wind estimation using xlb with the Warp (CUDA) backend.
Fetch OSM building footprints, extrude to a thin 3D slab, run 3D LBM (D3Q27),
and export a near-ground 2D wind field similar to the CPU script.

Requirements: xlb, warp, jax, numpy, shapely, trimesh, matplotlib

Example:
  uv run osm_wind_xlb.py --lat -37.81361 --lon 144.96332 --radius 200 --wind 8 \
    --dir-deg 90 --grid 200 --nz 12 --steps 2000 --upsample 2 --out-prefix outputs/melb_xlb
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple, Optional

import numpy as np
import shapely.affinity as sa
from shapely.geometry import Polygon

# Reuse fetching/rasterization and plotting from the CPU script
from osm_building_mesh import fetch_buildings_projected, Domain, rasterize_buildings, save_wind_outputs


def _pick_precision_policy():
    """Prefer full FP32 if available in this xlb version; else fall back to FP32FP16."""
    from xlb.precision_policy import PrecisionPolicy as PP
    for name in ("FP32", "FP32FP32", "FULL_FP32", "F32"):
        if hasattr(PP, name):
            return getattr(PP, name)
    return PP.FP32FP16


def _require_packages():
    try:
        import xlb  # noqa: F401
        import warp as wp  # noqa: F401
        import trimesh  # noqa: F401
        from xlb.compute_backend import ComputeBackend  # noqa: F401
        from xlb.precision_policy import PrecisionPolicy  # noqa: F401
        from xlb.grid import grid_factory  # noqa: F401
        from xlb.operator.stepper import IncompressibleNavierStokesStepper  # noqa: F401
        from xlb.operator.boundary_condition import (  # noqa: F401
            HalfwayBounceBackBC,
            FullwayBounceBackBC,
            RegularizedBC,
            ExtrapolationOutflowBC,
        )
        from xlb.operator.macroscopic import Macroscopic  # noqa: F401
        from xlb.velocity_set import D3Q27  # noqa: F401
    except Exception as e:
        raise RuntimeError(
            "xlb/warp/trimesh are required. Install packages: xlb, warp-lang, jax, trimesh"
        ) from e


def _parse_height_m(h_raw: Optional[object], levels_raw: Optional[object]) -> Optional[float]:
    def to_float(x: object) -> Optional[float]:
        if x is None:
            return None
        try:
            s = str(x).strip().lower().replace("m", "")
            return float(s)
        except Exception:
            return None

    h = to_float(h_raw)
    if h is not None:
        return max(0.0, h)
    lv = to_float(levels_raw)
    if lv is not None:
        return max(0.0, lv * 3.2)  # assume ~3.2 m per level
    return None


def _fetch_buildings_with_heights(lat: float, lon: float, radius_m: float):
    import osmnx as ox
    from shapely.geometry import Point, Polygon, MultiPolygon
    from typing import cast

    gdf = ox.features_from_point((lat, lon), tags={"building": True}, dist=radius_m)
    if gdf.empty:
        return [], [], None
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()
    if gdf.empty:
        return [], [], None

    gdf_proj = ox.projection.project_gdf(gdf)
    pt = Point(lon, lat)
    center_proj, _ = ox.projection.project_geometry(pt, to_crs=gdf_proj.crs)
    cx, cy = float(cast(Point, center_proj).x), float(cast(Point, center_proj).y)

    circle = Point(cx, cy).buffer(radius_m)
    polys: List[Polygon] = []
    heights: List[float] = []
    for _, row in gdf_proj.iterrows():
        geom = row.geometry
        if geom.is_empty:
            continue
        h_m = _parse_height_m(row.get("height"), row.get("building:levels"))
        if isinstance(geom, Polygon):
            if geom.intersects(circle):
                polys.append(geom)
                heights.append(h_m if h_m is not None else 12.0)
        elif isinstance(geom, MultiPolygon):
            for sub in geom.geoms:
                if not sub.is_empty and sub.intersects(circle):
                    polys.append(sub)
                    heights.append(h_m if h_m is not None else 12.0)

    half_extent = 3.0 * radius_m
    bounds = (cx - half_extent, cy - half_extent, cx + half_extent, cy + half_extent)
    return polys, heights, bounds


def _extrude_buildings_to_vertices(polys: List[Polygon], dom: Domain, nx: int, ny: int, nz: int, heights_m: Optional[List[float]] = None) -> np.ndarray:
    import trimesh

    if not polys:
        return np.empty((0, 3), dtype=np.float32)

    width = dom.xmax - dom.xmin
    height = dom.ymax - dom.ymin
    # Uniform spacing in LBM units
    dx = max(width / nx, height / ny)

    meshes = []
    for idx, p in enumerate(polys):
        p = p.buffer(0)
        if p.is_empty:
            continue
        # Transform to LBM grid coordinates: x'=(x-xmin)/dx, y'=(y-ymin)/dx
        p0 = sa.translate(p, xoff=-dom.xmin, yoff=-dom.ymin)
        p1 = sa.scale(p0, xfact=1.0 / dx, yfact=1.0 / dx, origin=(0, 0))
        # Determine extrusion height in lattice units
        if heights_m is not None and idx < len(heights_m):
            h_cells = int(max(1, min(nz - 2, round(heights_m[idx] / dx))))
        else:
            h_cells = max(1, nz - 2)
        try:
            m = trimesh.creation.extrude_polygon(p1, height=h_cells)
        except Exception:
            # Skip invalid geometries that fail to extrude
            continue
        # Lift to z in [1, 1+h]
        v = m.vertices.copy()
        v[:, 2] += 1.0
        m.vertices = v
        meshes.append(m)

    if not meshes:
        return np.empty((0, 3), dtype=np.float32)

    # Concatenate vertices of all meshes (xlb HalfwayBounceBackBC accepts vertices)
    verts = np.vstack([m.vertices for m in meshes]).astype(np.float32)
    return verts


def run_xlb_gpu(
    polygons: List[Polygon],
    dom: Domain,
    wind_mps: float,
    dir_deg: float,
    nx: int,
    ny: int,
    nz: int,
    steps: int,
    heights_m: Optional[List[float]] = None,
    compute_volume: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    import xlb
    import warp as wp
    from xlb.compute_backend import ComputeBackend
    from xlb.precision_policy import PrecisionPolicy
    from xlb.grid import grid_factory
    from xlb.operator.stepper import IncompressibleNavierStokesStepper
    from xlb.operator.boundary_condition import (
        HalfwayBounceBackBC,
        FullwayBounceBackBC,
        RegularizedBC,
        ExtrapolationOutflowBC,
    )
    from xlb.operator.macroscopic import Macroscopic
    from xlb.velocity_set import D3Q27

    compute_backend = ComputeBackend.WARP
    # Try to use full FP32; fall back to FP32FP16 if not available
    precision_policy = _pick_precision_policy()

    vel = D3Q27(precision_policy=precision_policy, compute_backend=compute_backend)

    # Initialize xlb
    xlb.init(velocity_set=vel, default_backend=compute_backend, default_precision_policy=precision_policy)

    grid = grid_factory((nx, ny, nz), compute_backend=compute_backend)

    # Determine inlet/outlet faces based on major component of wind direction (XY plane)
    theta = np.deg2rad(dir_deg)
    u_dir = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=np.float32)
    major_axis = int(np.argmax(np.abs(u_dir[:2])))  # 0=x, 1=y

    box = grid.bounding_box_indices()
    box_no_edge = grid.bounding_box_indices(remove_edges=True)

    # Helpers to work with index sets as [xs, ys, zs] Python lists (what xlb expects)
    def _ensure_triplet(idxs):
        # Normalize to 3 Python lists of ints
        xs = np.asarray(idxs[0]).ravel().astype(int).tolist()
        ys = np.asarray(idxs[1]).ravel().astype(int).tolist()
        zs = np.asarray(idxs[2]).ravel().astype(int).tolist()
        return [xs, ys, zs]

    def _concat_triplets(list_of_triplets):
        if not list_of_triplets:
            return [[], [], []]
        xs_all: list[int] = []
        ys_all: list[int] = []
        zs_all: list[int] = []
        for t in list_of_triplets:
            xs, ys, zs = _ensure_triplet(t)
            xs_all += xs
            ys_all += ys
            zs_all += zs
        return [xs_all, ys_all, zs_all]

    def _unique_triplet(idxs):
        xs, ys, zs = _ensure_triplet(idxs)
        if len(xs) == 0:
            return [xs, ys, zs]
        stacked = np.array(list(zip(xs, ys, zs)), dtype=np.int64)
        uniq = np.unique(stacked, axis=0)
        return [uniq[:, 0].astype(int).tolist(), uniq[:, 1].astype(int).tolist(), uniq[:, 2].astype(int).tolist()]

    if major_axis == 0:
        inlet = _ensure_triplet(box_no_edge["left"] if u_dir[0] >= 0 else box_no_edge["right"])
        outlet = _ensure_triplet(box_no_edge["right"] if u_dir[0] >= 0 else box_no_edge["left"])
        wall_faces = ["top", "bottom", "front", "back"]
    else:
        inlet = _ensure_triplet(box_no_edge["bottom"] if u_dir[1] >= 0 else box_no_edge["top"])
        outlet = _ensure_triplet(box_no_edge["top"] if u_dir[1] >= 0 else box_no_edge["bottom"])
        wall_faces = ["left", "right", "front", "back"]

    # Build walls as union of the remaining faces, excluding inlet/outlet
    walls = _unique_triplet(_concat_triplets([box[name] for name in wall_faces]))

    # Obstacles: extrude buildings across z
    obstacle_vertices = _extrude_buildings_to_vertices(polygons, dom, nx, ny, nz, heights_m=heights_m)

    # LBM relaxation (stable-ish)
    tau = 0.8
    omega = 1.0 / tau

    # Map physical inlet speed to lattice units by choosing a modest lattice inlet (similar to CPU path)
    # IMPORTANT: Some xlb/Warp versions expect a velocity vector with only the normal component non-zero.
    # Use a Python float tuple to let Warp infer consistent scalar types.
    U_in_lbm = 0.08
    scale = wind_mps / U_in_lbm if U_in_lbm != 0 else 0.0
    if major_axis == 0:
        bc_value = (float(U_in_lbm), 0.0, 0.0)
    else:
        bc_value = (0.0, float(U_in_lbm), 0.0)

    bcs = [
        FullwayBounceBackBC(indices=walls),
        RegularizedBC("velocity", prescribed_value=bc_value, indices=inlet),
        ExtrapolationOutflowBC(indices=outlet),
    ]
    if obstacle_vertices.size > 0:
        bcs.append(HalfwayBounceBackBC(mesh_vertices=obstacle_vertices))

    stepper = IncompressibleNavierStokesStepper(
        grid=grid,
        boundary_conditions=bcs,
        collision_type="KBC",
    )

    f_0, f_1, bc_mask, missing_mask = stepper.prepare_fields()

    for it in range(steps):
        f_0, f_1 = stepper(f_0, f_1, bc_mask, missing_mask, omega, it)
        f_0, f_1 = f_1, f_0
    # Optional: synchronize between chunks if desired (removed to avoid linter/runtime mismatch)

    # Macroscopic fields computed on CPU from distribution function
    # 1) Copy distribution to NumPy via Torch (warp -> torch -> cpu -> numpy)
    import torch  # type: ignore
    f_torch = wp.to_torch(f_0)  # type: ignore[attr-defined]
    f_np = f_torch.detach().cpu().numpy()

    # 2) Identify Q axis (should be 27 for D3Q27)
    shape = f_np.shape
    try:
        qax = list(shape).index(27)
    except ValueError as e:
        raise RuntimeError(f"Could not find Q=27 axis in distribution shape {shape}") from e
    fQ = np.moveaxis(f_np, qax, 0)  # (27, A, B, C)

    # 3) Build D3Q27 lattice velocities c_i = (cx, cy, cz) in lexicographic order
    c = np.array([(cx, cy, cz) for cz in (-1, 0, 1) for cy in (-1, 0, 1) for cx in (-1, 0, 1)], dtype=np.int8)
    if c.shape[0] != 27:
        raise RuntimeError("D3Q27 velocity set construction failed")

    # 4) Compute density and momentum
    rho = np.sum(fQ, axis=0)  # (A, B, C)
    eps = 1e-20
    mx = np.tensordot(c[:, 0], fQ, axes=(0, 0))  # (A, B, C)
    my = np.tensordot(c[:, 1], fQ, axes=(0, 0))
    mz = np.tensordot(c[:, 2], fQ, axes=(0, 0))
    ux_all = mx / (rho + eps)
    uy_all = my / (rho + eps)
    uz_all = mz / (rho + eps)

    # 5) Extract mid-Z slice; determine which axis corresponds to nz
    spatial_shape = ux_all.shape  # (A, B, C)
    try:
        z_axis = list(spatial_shape).index(nz)
    except ValueError:
        z_axis = int(np.argmin(spatial_shape))  # fallback to smallest dim
    k = nz // 2
    ux = np.take(ux_all, indices=k, axis=z_axis)
    uy = np.take(uy_all, indices=k, axis=z_axis)
    u_mag = np.sqrt(ux**2 + uy**2)

    # Convert to m/s by the linear scale
    ux_mps = ux * scale
    uy_mps = uy * scale
    u_mps = u_mag * scale
    if compute_volume:
        try:
            # Reorder arrays so last axis is vertical (z), first two are horizontal (x,y)
            other_axes = [ax for ax in (0, 1, 2) if ax != z_axis]
            perm = [other_axes[0], other_axes[1], z_axis]
            ux_vol = np.transpose(ux_all, perm) * scale
            uy_vol = np.transpose(uy_all, perm) * scale
            uz_vol = np.transpose(uz_all, perm) * scale
            return u_mps.T, ux_mps.T, uy_mps.T, ux_vol, uy_vol, uz_vol
        except Exception as e:
            print(f"Warning: 3D volume export disabled due to: {e}")
            return u_mps.T, ux_mps.T, uy_mps.T, None, None, None
    else:
        return u_mps.T, ux_mps.T, uy_mps.T, None, None, None  # transpose to (ny, nx) for plotting


def main():
    _require_packages()

    parser = argparse.ArgumentParser(description="OSM wind (xlb + Warp CUDA)")
    parser.add_argument("--lat", type=float, required=True)
    parser.add_argument("--lon", type=float, required=True)
    parser.add_argument("--radius", type=float, default=100.0)
    parser.add_argument("--wind", type=float, required=True, help="Inlet wind speed (m/s)")
    parser.add_argument("--dir-deg", type=float, default=0.0)
    parser.add_argument("--grid", type=int, default=200, help="Grid nx==ny")
    parser.add_argument("--nz", type=int, default=12, help="Grid thickness (z) in cells")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--upsample", type=int, default=1)
    parser.add_argument("--use-osm-heights", action="store_true", help="Use OSM height/levels to extrude 3D buildings")
    parser.add_argument("--out-prefix", type=str, default="osm_wind_xlb")
    parser.add_argument("--plotly", action="store_true", help="Write an interactive 3D Plotly HTML (buildings + wind)")
    parser.add_argument("--domain-scale", type=float, default=1.0, help="Scale the XY extents of the simulation domain around its center (e.g., 1.5 to expand by 50%).")
    args = parser.parse_args()

    print(f"Fetching OSM buildings around ({args.lat}, {args.lon}) within {args.radius} m…")
    heights_m: Optional[List[float]] = None
    if args.use_osm_heights:
        polygons, heights_m, bounds = _fetch_buildings_with_heights(args.lat, args.lon, args.radius)
    else:
        polygons, bounds = fetch_buildings_projected(args.lat, args.lon, args.radius)
    if bounds is None:
        print("No buildings found; aborting.")
        return
    xmin, ymin, xmax, ymax = bounds
    # Optionally scale the domain extents around center to increase the wind tunnel size
    if args.domain_scale and args.domain_scale != 1.0:
        cx = 0.5 * (xmin + xmax)
        cy = 0.5 * (ymin + ymax)
        hx = 0.5 * (xmax - xmin) * args.domain_scale
        hy = 0.5 * (ymax - ymin) * args.domain_scale
        xmin, xmax = cx - hx, cx + hx
        ymin, ymax = cy - hy, cy + hy
    dom = Domain(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, nx=args.grid, ny=args.grid)

    print(f"Rasterizing buildings to a {dom.nx}x{dom.ny} grid for overlays…")
    obstacle = rasterize_buildings(polygons, dom)

    print(f"Running xlb (Warp CUDA) on a {args.grid}x{args.grid}x{args.nz} lattice for {args.steps} steps…")
    u_mps, ux_mps, uy_mps, ux_vol, uy_vol, uz_vol = run_xlb_gpu(
        polygons,
        dom,
        args.wind,
        args.dir_deg,
        dom.nx,
        dom.ny,
        args.nz,
        args.steps,
        heights_m=heights_m,
        compute_volume=args.plotly,
    )

    # Align orientation with rasterized obstacle for correct overlay
    def _align_orientation(u: np.ndarray, ux: np.ndarray, uy: np.ndarray, obstacle_mask: np.ndarray):
        # Ensure candidate A has same shape as obstacle; if not, try transpose
        def ensure_shape(arr: np.ndarray):
            if arr.shape == obstacle_mask.shape:
                return arr
            if arr.T.shape == obstacle_mask.shape:
                return arr.T
            return arr

        uA = ensure_shape(u)
        uxA = ensure_shape(ux)
        uyA = ensure_shape(uy)
        # Candidate B = transpose with component swap
        uB = uA.T
        uxB = uyA.T
        uyB = uxA.T

        def score(u_field: np.ndarray):
            try:
                return float(np.nanmean(u_field[obstacle_mask]))
            except Exception:
                return float('inf')

        sA = score(uA)
        sB = score(uB)
        return (uA, uxA, uyA) if sA <= sB else (uB, uxB, uyB)

    u_mps, ux_mps, uy_mps = _align_orientation(u_mps, ux_mps, uy_mps, obstacle)

    # Quick diagnostics: report velocity stats
    try:
        import numpy as _np
        print(
            "u_mps stats: min={:.3f} max={:.3f} mean={:.3f}".format(
                float(_np.nanmin(u_mps)), float(_np.nanmax(u_mps)), float(_np.nanmean(u_mps))
            )
        )
        if ux_vol is not None and uy_vol is not None and uz_vol is not None:
            u3 = _np.sqrt(ux_vol**2 + uy_vol**2 + uz_vol**2)
            print(
                "u3D stats: min={:.3f} max={:.3f} mean={:.3f}".format(
                    float(_np.nanmin(u3)), float(_np.nanmax(u3)), float(_np.nanmean(u3))
                )
            )
    except Exception:
        pass

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

    # Optional: Plotly interactive 3D visualization
    if args.plotly:
        from plotly import graph_objects as go
        import plotly.io as pio
        import numpy as np
        import trimesh

        # Recreate extruded meshes with faces for plotting
        def _extrude_meshes(polys: List[Polygon], dom: Domain, nx: int, ny: int, nz: int, heights_m: Optional[List[float]]):
            import shapely.affinity as sa
            meshes = []
            width = dom.xmax - dom.xmin
            height = dom.ymax - dom.ymin
            dx = max(width / nx, height / ny)
            for idx, p in enumerate(polys):
                p = p.buffer(0)
                if p.is_empty:
                    continue
                p0 = sa.translate(p, xoff=-dom.xmin, yoff=-dom.ymin)
                p1 = sa.scale(p0, xfact=1.0 / dx, yfact=1.0 / dx, origin=(0, 0))
                if heights_m is not None and idx < len(heights_m):
                    h_cells = int(max(1, min(nz - 2, round(heights_m[idx] / dx))))
                else:
                    h_cells = max(1, nz - 2)
                try:
                    m = trimesh.creation.extrude_polygon(p1, height=h_cells)
                except Exception:
                    continue
                v = m.vertices.copy()
                v[:, 2] += 1.0
                # Convert back to meters
                v_m = v.copy()
                v_m[:, 0] = v[:, 0] * dx + dom.xmin
                v_m[:, 1] = v[:, 1] * dx + dom.ymin
                v_m[:, 2] = v[:, 2] * dx
                meshes.append((v_m, m.faces.copy()))
            return meshes

        meshes = _extrude_meshes(polygons, dom, dom.nx, dom.ny, args.nz, heights_m)

        # Merge meshes into a single Mesh3d trace
        Xv = []
        Yv = []
        Zv = []
        I = []
        J = []
        K = []
        vert_offset = 0
        for v, faces in meshes:
            Xv.append(v[:, 0])
            Yv.append(v[:, 1])
            Zv.append(v[:, 2])
            I.append(faces[:, 0] + vert_offset)
            J.append(faces[:, 1] + vert_offset)
            K.append(faces[:, 2] + vert_offset)
            vert_offset += v.shape[0]
        if vert_offset > 0:
            mesh_trace = go.Mesh3d(
                x=np.concatenate(Xv),
                y=np.concatenate(Yv),
                z=np.concatenate(Zv),
                i=np.concatenate(I),
                j=np.concatenate(J),
                k=np.concatenate(K),
                color="white",
                opacity=1.0,
                name="Buildings",
            )
        else:
            mesh_trace = None

        # Build 3D wind visualization
        traces = []
        if mesh_trace is not None:
            traces.append(mesh_trace)
        if ux_vol is not None and uy_vol is not None and uz_vol is not None:
            # Coordinates in meters
            xs = np.linspace(dom.xmin + 0.5 * dom.dx, dom.xmax - 0.5 * dom.dx, dom.nx)
            ys = np.linspace(dom.ymin + 0.5 * dom.dy, dom.ymax - 0.5 * dom.dy, dom.ny)
            # Align wind layer centers with building base (buildings start at z = 1*dx)
            zs = np.linspace(1.0 * dom.dx, args.nz * dom.dx, args.nz)

            # Volume magnitude (downsample for performance)
            ds = max(1, int(max(dom.nx, dom.ny, args.nz) // 64))
            u_mag_vol = np.sqrt(ux_vol**2 + uy_vol**2 + uz_vol**2)
            v_val = u_mag_vol[::ds, ::ds, ::ds]
            xs_ds = xs[::ds]
            ys_ds = ys[::ds]
            zs_ds = zs[::ds]
            nx_s, ny_s, nz_s = len(xs_ds), len(ys_ds), len(zs_ds)

            # Build a single full-height volume trace
            Xv3 = np.repeat(xs_ds, ny_s * nz_s)
            Yv3 = np.tile(np.repeat(ys_ds, nz_s), nx_s)
            Zv3 = np.tile(zs_ds, nx_s * ny_s)
            vmax = float(np.nanmax(v_val)) if np.size(v_val) > 0 else 0.0
            vmin = float(np.nanmin(v_val)) if np.size(v_val) > 0 else 0.0
            # Avoid degenerate color range
            if not np.isfinite(vmin):
                vmin = 0.0
            if not np.isfinite(vmax):
                vmax = 1e-6
            if vmax - vmin < 1e-12:
                vmax = vmin + 1e-6
            # Choose a small threshold so faint flow is still visible
            isomin = 0.05 * vmax if vmax > 0 else 0.0
            isomax = vmax if vmax > 0 else 1e-6
            traces.append(
                go.Volume(
                    x=Xv3,
                    y=Yv3,
                    z=Zv3,
                    value=v_val.ravel(),
                    opacity=0.5,
                    surface_count=8,
                    colorscale="Turbo",
                    name="|u|",
                    showscale=True,
                    isomin=isomin,
                    isomax=isomax,
                    cmin=vmin,
                    cmax=vmax,
                )
            )

        fig = go.Figure(data=traces)
        fig.update_layout(
            scene=dict(
                xaxis_title="X (m)",
                yaxis_title="Y (m)",
                zaxis_title="Z (m)",
                aspectmode="data",
            ),
            title="3D Wind and Buildings",
        )
        html_path = f"{args.out_prefix}_3d.html"
        pio.write_html(fig, file=html_path, auto_open=False, include_plotlyjs=True)
        print(f"Plotly 3D written to {html_path}")


if __name__ == "__main__":
    main()
