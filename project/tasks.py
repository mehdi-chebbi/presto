"""
WorldCereal job submission, status management, result download, and tile serving.

Handles openEO backend communication, process graph construction
(using worldcereal's create_inference_process_graph for real classification,
or a basic S2 NDVI fallback), batch job lifecycle, local job state tracking,
GeoTIFF download, and on-demand XYZ tile rendering.
"""

import json
import logging
import math
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

from config import Config

logger = logging.getLogger("worldcereal")

# ── Check if worldcereal package is available ────────────────────────────
# Verified real API from worldcereal v2.6.1:
#   worldcereal.job exports: create_inference_process_graph, DEFAULT_INFERENCE_JOB_OPTIONS,
#     WorldCerealProductType, WorldCerealTask, BoundingBoxExtent, TemporalContext,
#     BackendContext, Backend, load_model_artifact, ...
#   worldcereal.parameters exports: WorldCerealProductType, BaseParameters, FeaturesParameters,
#     EmbeddingsParameters
#   NO CropLandParameters, CropTypeParameters, or PostprocessParameters exist.

WORLDCEREAL_AVAILABLE = False
WC_JOB_OPTIONS = None
WC_WORKFLOW_CONFIG_CLS = None
WC_SEASON_SECTION_CLS = None
try:
    from worldcereal.job import (
        create_inference_process_graph,
        DEFAULT_INFERENCE_JOB_OPTIONS,
        WorldCerealProductType,
        BoundingBoxExtent,
        TemporalContext,
        BackendContext,
        Backend,
    )
    from worldcereal.openeo.workflow_config import (
        WorldCerealWorkflowConfig,
        SeasonSection,
    )
    WC_JOB_OPTIONS = DEFAULT_INFERENCE_JOB_OPTIONS
    WC_WORKFLOW_CONFIG_CLS = WorldCerealWorkflowConfig
    WC_SEASON_SECTION_CLS = SeasonSection
    WORLDCEREAL_AVAILABLE = True
    logger.info("worldcereal v2.6+ loaded — real classification enabled")
    logger.info("  job_options keys: %s", list(WC_JOB_OPTIONS.keys()))
except ImportError as exc:
    logger.warning(
        "worldcereal / openeo-gfmap not fully available (%s). "
        "Falling back to NDVI demo. Install from GitHub:\n"
        '  pip install "worldcereal @ git+https://github.com/WorldCereal/worldcereal-classification.git"',
        exc,
    )

# ── Classification colormaps ─────────────────────────────────────────────

# CROPLAND: 0 = non-cropland, 1 = cropland (binary)
CROPLAND_COLORMAP = {
    0: (200, 200, 200, 0),      # non-cropland → transparent
    1: (34, 139, 34, 220),      # cropland → forest green
}

# CROPTYPE: integer class labels → crop names + colors
CROPTYPE_COLORMAP = {
    0:  (200, 200, 200, 0),     # no data / other → transparent
    1:  (255, 200, 0, 220),     # maize → gold
    2:  (180, 130, 70, 220),    # wheat → tan/brown
    3:  (220, 180, 130, 220),   # barley → light brown
    4:  (180, 220, 50, 220),    # rapeseed / canola → lime
    5:  (255, 165, 0, 220),     # sunflower → orange
    6:  (0, 150, 0, 220),       # soybean → dark green
    7:  (160, 100, 50, 220),    # potato → brown
    8:  (100, 180, 255, 220),   # sugar beet → light blue
    9:  (80, 200, 120, 220),    # temporary grassland → mint
    10: (50, 120, 50, 220),     # orchard / permanent crops → forest green
    11: (0, 100, 200, 220),     # rice → blue
    12: (200, 100, 200, 220),   # other crop → purple
    13: (200, 200, 200, 0),     # no crop / non-cropland → transparent
}

CROPTYPE_LABELS = {
    0: "No data",
    1: "Maize",
    2: "Wheat",
    3: "Barley",
    4: "Rapeseed / Canola",
    5: "Sunflower",
    6: "Soybean",
    7: "Potato",
    8: "Sugar beet",
    9: "Temporary grassland",
    10: "Orchard / Permanent crops",
    11: "Rice",
    12: "Other crop",
    13: "Non-cropland",
}

# LANDCOVER: 10 classes from the Presto LANDCOVER head
# Class indices match the model's class_names order:
#   ['temporary_crops', 'temporary_grasses', 'bare_sparsely_vegetated',
#    'permanent_crops', 'grasslands', 'wetlands', 'shrubland', 'trees',
#    'built_up', 'water']
LANDCOVER_COLORMAP = {
    0: (255, 255, 0, 220),       # temporary_crops → yellow
    1: (173, 255, 47, 220),      # temporary_grasses → green yellow
    2: (210, 180, 140, 220),     # bare_sparsely_vegetated → tan
    3: (255, 165, 0, 220),       # permanent_crops → orange
    4: (144, 238, 144, 220),     # grasslands → light green
    5: (0, 170, 170, 220),       # wetlands → teal
    6: (189, 184, 120, 220),     # shrubland → olive
    7: (0, 100, 0, 220),         # trees → dark green
    8: (200, 50, 50, 220),       # built_up → red
    9: (0, 100, 200, 220),       # water → blue
}

LANDCOVER_LABELS = {
    0: "Temporary Crops",
    1: "Temporary Grasses",
    2: "Bare / Sparse Vegetation",
    3: "Permanent Crops / Orchards",
    4: "Grasslands",
    5: "Wetlands",
    6: "Shrubland",
    7: "Trees / Forest",
    8: "Built-up",
    9: "Water",
}

# ── Check for tile rendering deps ────────────────────────────────────────

RASTERIO_AVAILABLE = False
try:
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import transform_bounds, reproject, Resampling
    from rasterio.transform import from_bounds
    RASTERIO_AVAILABLE = True
    logger.info("rasterio available for tile rendering")
except ImportError:
    logger.warning("rasterio not installed — tile serving will not work. pip install rasterio")

RIO_TILER_AVAILABLE = False
try:
    from rio_tiler.io import Reader
    from rio_tiler.colormap import cmap as rio_cmap
    RIO_TILER_AVAILABLE = True
    logger.info("rio-tiler available for fast tile rendering")
except ImportError:
    logger.warning("rio-tiler not installed — will use rasterio fallback. pip install rio-tiler")

try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("Pillow not installed — pip install Pillow")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    logger.warning("numpy not installed — pip install numpy")


# ── Geometry Helpers ─────────────────────────────────────────────────────

def geojson_to_bbox(geometry: dict) -> dict:
    """
    Extract bounding box from a GeoJSON geometry (Polygon or MultiPolygon).
    Returns dict like {"west": ..., "south": ..., "east": ..., "north": ...}
    """
    geom_type = geometry.get("type", "")
    coords = geometry.get("coordinates", [])

    if geom_type == "Polygon":
        rings = coords
    elif geom_type == "MultiPolygon":
        rings = []
        for polygon in coords:
            rings.extend(polygon)
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")

    all_lng = []
    all_lat = []
    for ring in rings:
        for point in ring:
            all_lng.append(point[0])
            all_lat.append(point[1])

    return {
        "west": min(all_lng),
        "south": min(all_lat),
        "east": max(all_lng),
        "north": max(all_lat),
    }


# ── Local Job Store (simple JSON file) ──────────────────────────────────

def _job_store_path() -> Path:
    return Path(Config.JOBS_DIR) / "jobs.json"


def _load_job_store() -> dict:
    path = _job_store_path()
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as exc:
            logger.warning("Could not load job store: %s", exc)
    return {}


def _save_job_store(jobs: dict) -> None:
    path = _job_store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(jobs, f, indent=2, default=str)
    except IOError as exc:
        logger.error("Could not save job store: %s", exc)


# ── openEO Connection ───────────────────────────────────────────────────

def connect_openeo(access_token: str):
    """
    Connect to the CDSE openEO backend using an existing OIDC access token.
    """
    logger.info("Connecting to openEO backend: %s", Config.OPENEO_BACKEND)
    logger.debug("Access token (first 20 chars): %s...", access_token[:20] if access_token else "EMPTY")

    if not access_token:
        raise ValueError("No access token provided — session may have expired. Please log in again.")

    import openeo

    conn = openeo.connect(f"https://{Config.OPENEO_BACKEND}")

    logger.info("Authenticating with OIDC access token...")
    try:
        conn.authenticate_oidc_access_token(
            access_token=access_token,
            provider_id="CDSE",
        )
    except (AttributeError, TypeError):
        logger.info("authenticate_oidc_access_token not available, using manual header injection")
        conn.session.auth = None
        conn.session.headers["Authorization"] = f"Bearer {access_token}"

    logger.info("Verifying authentication...")
    try:
        conn.list_jobs(limit=1)
        logger.info("openEO authentication successful!")
    except Exception as exc:
        logger.warning("Auth verification returned an error (may still work): %s", exc)

    try:
        caps = conn.capabilities()
        logger.info(
            "Backend: %s | API v%s",
            caps.get("production", "unknown"),
            caps.get("api_version", "unknown"),
        )
    except Exception as exc:
        logger.warning("Could not fetch backend capabilities (non-fatal): %s", exc)

    return conn


# ── Process Graph Building ──────────────────────────────────────────────

def _build_process_graph(conn, geometry: dict, start_date: str, end_date: str, product_type: str):
    """
    Build the openEO DataCube for crop classification.

    Returns tuple: (datacube_or_list, job_options_or_None)
    """
    if WORLDCEREAL_AVAILABLE:
        logger.info("Using worldcereal create_inference_process_graph")
        return _build_worldcereal_pg(conn, geometry, start_date, end_date, product_type)
    else:
        logger.info("worldcereal package not available — building basic S2 NDVI demo process")
        return _build_basic_s2_pg(conn, geometry, start_date, end_date), None


def _build_worldcereal_pg(conn, geometry, start_date, end_date, product_type):
    """
    Build process graph using worldcereal's create_inference_process_graph.
    Returns tuple: (datacube, job_options)
    """
    logger.info("Building WorldCereal classification process graph")
    logger.info("  product_type = %s", product_type)
    logger.info("  temporal     = [%s, %s]", start_date, end_date)

    bbox = geojson_to_bbox(geometry)
    logger.info("  bbox = %s", bbox)

    try:
        spatial_extent = BoundingBoxExtent(
            west=bbox["west"],
            south=bbox["south"],
            east=bbox["east"],
            north=bbox["north"],
            epsg=4326,
        )
        temporal_extent = TemporalContext(start_date, end_date)

        # Determine product type enum
        # Note: "landcover" uses CROPLAND enum internally — it triggers the same
        # LANDCOVER head but our modified UDF also outputs landcover_classification
        if product_type == "croptype":
            ptype = WorldCerealProductType.CROPTYPE
            logger.info("  Classification: CROPTYPE (multi-class)")
        else:
            ptype = WorldCerealProductType.CROPLAND
            logger.info("  Classification: CROPLAND/LANDCOVER (product_type=%s)", product_type)

        # Build workflow config with proper season section.
        # The default 'phase_ii_multitask' preset sets season=None, which
        # crashes in _finalize_season_requirements. We override it here.
        season_section = WC_SEASON_SECTION_CLS(
            season_ids=["annual"],
        )
        workflow_config = WC_WORKFLOW_CONFIG_CLS(
            season=season_section,
        )
        logger.info("  workflow_config season: %s", season_section)

        # create_inference_process_graph accepts connection= directly
        logger.info("  Calling create_inference_process_graph(connection=<user_conn>)...")
        datacubes = create_inference_process_graph(
            spatial_extent=spatial_extent,
            temporal_extent=temporal_extent,
            product_type=ptype,
            connection=conn,
            backend_context=BackendContext(Backend.CDSE),
            out_format="GTiff",
            workflow_config=workflow_config,
        )

        logger.info("  Process graph built! Got %d datacube(s)", len(datacubes) if isinstance(datacubes, list) else 1)

        # Return first datacube (small AOI = 1 tile)
        if isinstance(datacubes, list):
            if len(datacubes) > 1:
                logger.warning("  AOI was split into %d tiles — using first tile only. "
                               "Large AOIs may need job-manager approach.", len(datacubes))
            cube = datacubes[0]
        else:
            cube = datacubes

        return cube, WC_JOB_OPTIONS

    except Exception as exc:
        logger.error("Failed to build worldcereal process graph: %s", exc)
        logger.debug("Full traceback:\n%s", traceback.format_exc())
        raise RuntimeError(
            f"worldcereal process graph failed: {exc}\n"
            "Check the logs above for details."
        ) from exc


def _build_basic_s2_pg(conn, geometry, start_date, end_date):
    """
    Build a basic Sentinel-2 NDVI process graph (demo/fallback when
    worldcereal is not installed). Computes max NDVI from S2 L2A.
    """
    logger.info("Building basic S2 NDVI process graph (DEMO — not real classification)...")
    logger.info("  temporal = [%s, %s]", start_date, end_date)

    bbox = geojson_to_bbox(geometry)
    logger.info("  bbox = %s", bbox)

    logger.info("Loading SENTINEL2_L2A collection...")
    cube = conn.load_collection(
        "SENTINEL2_L2A",
        spatial_extent=bbox,
        temporal_extent=[start_date, end_date],
        bands=["B04", "B08"],
        max_cloud_cover=80,
    )

    logger.info("Computing NDVI...")

    def compute_ndvi(data):
        red = data.array_element(index=0)   # B04 Red
        nir = data.array_element(index=1)   # B08 NIR
        return (nir - red) / (nir + red)

    ndvi = cube.reduce_dimension(dimension="bands", reducer=compute_ndvi)

    logger.info("Taking temporal maximum NDVI...")
    ndvi_max = ndvi.reduce_dimension(dimension="t", reducer="max")

    logger.info("Basic NDVI process graph built successfully")
    return ndvi_max


# ── Job Submission ──────────────────────────────────────────────────────

def submit_classification_job(
    access_token: str,
    geometry: dict,
    start_date: str,
    end_date: str,
    product_type: str,
) -> dict:
    """
    Submit a crop classification job to the CDSE openEO backend.
    Returns dict: {"job_id": str, "status": str, "message": str}
    """
    logger.info("=" * 60)
    logger.info("NEW CLASSIFICATION JOB SUBMISSION")
    logger.info("=" * 60)
    logger.info("Product type  : %s", product_type)
    logger.info("Date range    : %s to %s", start_date, end_date)
    logger.info("Geometry type : %s", geometry.get("type", "unknown"))
    logger.info("WorldCereal   : %s", "YES (real classification)" if WORLDCEREAL_AVAILABLE else "NO (NDVI demo)")

    geom_str = json.dumps(geometry)
    if len(geom_str) > 500:
        logger.info("Geometry: %s...[truncated]...%s", geom_str[:200], geom_str[-200:])
    else:
        logger.info("Geometry: %s", geom_str)

    logger.info("[Step 1/4] Connecting to openEO backend...")
    conn = connect_openeo(access_token)

    logger.info("[Step 2/4] Building process graph...")
    result = _build_process_graph(conn, geometry, start_date, end_date, product_type)

    # result can be (datacube, job_options) from worldcereal or (datacube, None) from fallback
    if isinstance(result, tuple):
        datacube, job_options = result
    else:
        datacube = result
        job_options = None

    job_title = f"WorldCereal {product_type} | {start_date} to {end_date}"
    logger.info("[Step 3/4] Creating batch job '%s'...", job_title)

    # Create job — WorldCereal handles save_result format internally
    create_kwargs = {
        "title": job_title,
        "description": "Submitted via WorldCereal Explorer",
    }

    if job_options:
        logger.info("  Applying WorldCereal UDF job_options (ONNX + Torch deps)")
        create_kwargs["job_options"] = job_options
    else:
        # NDVI fallback needs explicit out_format
        create_kwargs["out_format"] = "GTiff"

    job = datacube.create_job(**create_kwargs)
    job_id = job.job_id
    logger.info("Batch job created successfully! ID: %s", job_id)

    logger.info("[Step 4/4] Starting job %s...", job_id)
    job.start()
    logger.info("Job %s started!", job_id)

    # Save to local store
    jobs = _load_job_store()
    jobs[job_id] = {
        "status": "queued",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "product_type": product_type,
        "start_date": start_date,
        "end_date": end_date,
        "geometry_type": geometry.get("type", "unknown"),
        "title": job_title,
        "worldcereal": WORLDCEREAL_AVAILABLE,
    }
    _save_job_store(jobs)
    logger.info("Job %s saved to local store", job_id)

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Job submitted. Monitor progress via /api/status/{job_id}",
    }


# ── Job Status ──────────────────────────────────────────────────────────

def get_job_status(access_token: str, job_id: str) -> dict:
    """
    Check the status of a submitted batch job.
    Returns dict: {"job_id": str, "status": str, "updated": str, ...}
    """
    logger.info("Checking status for job: %s", job_id)

    local_jobs = _load_job_store()
    if job_id in local_jobs:
        local_info = local_jobs[job_id]
        logger.info("Local store has job %s: status=%s", job_id, local_info.get("status"))
        if local_info.get("result_path") and Path(local_info["result_path"]).exists():
            logger.info("Results already downloaded for job %s: %s", job_id, local_info["result_path"])
    else:
        logger.info("Job %s not found in local store", job_id)

    try:
        conn = connect_openeo(access_token)
    except Exception as exc:
        logger.error("Failed to connect to openEO for status check: %s", exc)
        return {
            "job_id": job_id,
            "status": "error",
            "error": f"Connection failed: {exc}",
        }

    try:
        job = conn.job(job_id)
        status = job.status()
        logger.info("Remote job %s status: %s", job_id, status)

        try:
            metadata = job.describe()
            logger.debug("Job metadata: %s", json.dumps(metadata, indent=2, default=str))
            title = metadata.get("title", "")
            progress = metadata.get("progress", None)
            created = metadata.get("created", "")
            updated = metadata.get("updated", "")
            error_info = metadata.get("error", None)
        except Exception as exc:
            logger.warning("Could not fetch job metadata (non-fatal): %s", exc)
            title = ""
            progress = None
            created = ""
            updated = ""
            error_info = None

        # Update local store
        if job_id in local_jobs:
            local_jobs[job_id]["status"] = status
            local_jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            if error_info:
                local_jobs[job_id]["error"] = error_info
            _save_job_store(local_jobs)

        result = {
            "job_id": job_id,
            "status": status,
            "title": title,
            "created": created,
            "updated": updated,
        }

        if progress is not None:
            result["progress"] = progress

        if status == "finished":
            logger.info("Job %s is FINISHED!", job_id)
            result["message"] = "Job completed successfully."

            # Check if results already downloaded
            result_path = _get_result_path(job_id)
            if result_path and Path(result_path).exists():
                logger.info("Results already at %s", result_path)
                result["results_downloaded"] = True
                result["result_path"] = result_path
            else:
                result["results_downloaded"] = False

        elif status == "error":
            logger.error("Job %s FAILED!", job_id)
            result["error"] = error_info or "Job failed on the backend."
            try:
                logs = job.logs()
                if logs:
                    log_lines = []
                    for entry in logs[-10:]:
                        log_lines.append(f"[{entry.get('level', '?')}] {entry.get('message', '')}")
                    result["logs"] = log_lines
                    for line in log_lines:
                        logger.error("  Job log: %s", line)
            except Exception as log_exc:
                logger.warning("Could not fetch job logs: %s", log_exc)

        return result

    except Exception as exc:
        logger.error("Failed to get status for job %s: %s", job_id, exc)
        logger.debug("Traceback:\n%s", traceback.format_exc())
        return {
            "job_id": job_id,
            "status": "error",
            "error": str(exc),
        }


# ── Result Download ─────────────────────────────────────────────────────

def _get_result_path(job_id: str) -> str | None:
    """Return the expected local path for a job's GeoTIFF result."""
    result_dir = os.path.join(Config.RESULTS_DIR, job_id)
    if os.path.isdir(result_dir):
        for f in os.listdir(result_dir):
            if f.lower().endswith((".tif", ".tiff")):
                return os.path.join(result_dir, f)
    return None


def download_job_result(access_token: str, job_id: str) -> dict:
    """
    Download the GeoTIFF result from a finished batch job.
    Returns dict: {"job_id": str, "result_path": str, "bounds": {...}}
    """
    logger.info("=== Downloading results for job %s ===", job_id)

    # Check if already downloaded
    existing = _get_result_path(job_id)
    if existing:
        logger.info("Results already downloaded: %s", existing)

        # Read bounds if not stored
        local_jobs = _load_job_store()
        job_info = local_jobs.get(job_id, {})
        bounds = job_info.get("bounds")

        if not bounds and RASTERIO_AVAILABLE:
            bounds = _read_bounds(existing)
            if bounds and job_id in local_jobs:
                local_jobs[job_id]["bounds"] = bounds
                _save_job_store(local_jobs)

        return {
            "job_id": job_id,
            "result_path": existing,
            "bounds": bounds,
            "message": "Results already downloaded.",
        }

    # Connect and get the job
    conn = connect_openeo(access_token)
    job = conn.job(job_id)

    status = job.status()
    if status != "finished":
        logger.error("Job %s is not finished (status=%s), cannot download", job_id, status)
        return {
            "job_id": job_id,
            "error": f"Job is not finished (status={status})",
        }

    # Create output directory
    output_dir = os.path.join(Config.RESULTS_DIR, job_id)
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Downloading to: %s", output_dir)

    try:
        results = job.get_results()
        paths = results.download_files(output_dir)
        logger.info("Downloaded %d file(s):", len(paths))
        for p in paths:
            size_kb = os.path.getsize(p) / 1024
            logger.info("  %s (%.1f KB)", p, size_kb)

        # Find the GeoTIFF
        tif_path = None
        for p in paths:
            if str(p).lower().endswith((".tif", ".tiff")):
                tif_path = str(p)
                break

        if not tif_path:
            for f in os.listdir(output_dir):
                if f.lower().endswith((".tif", ".tiff")):
                    tif_path = os.path.join(output_dir, f)
                    break

        if not tif_path:
            logger.error("No GeoTIFF found in downloaded results!")
            return {
                "job_id": job_id,
                "error": "No GeoTIFF found in downloaded results",
                "downloaded_files": [str(p) for p in paths],
            }

        logger.info("GeoTIFF result: %s", tif_path)

        # Read metadata
        bounds = None
        if RASTERIO_AVAILABLE:
            bounds = _read_bounds(tif_path)

        # Update local store
        local_jobs = _load_job_store()
        if job_id in local_jobs:
            local_jobs[job_id]["result_path"] = tif_path
            if bounds:
                local_jobs[job_id]["bounds"] = bounds
            local_jobs[job_id]["downloaded_at"] = datetime.now(timezone.utc).isoformat()
            _save_job_store(local_jobs)

        return {
            "job_id": job_id,
            "result_path": tif_path,
            "bounds": bounds,
            "message": "Results downloaded successfully.",
        }

    except Exception as exc:
        logger.error("Failed to download results for job %s:\n%s", job_id, traceback.format_exc())
        return {
            "job_id": job_id,
            "error": str(exc),
        }


def _read_bounds(tiff_path: str) -> dict | None:
    """Read WGS84 bounds from a GeoTIFF using rasterio."""
    try:
        with rasterio.open(tiff_path) as ds:
            wgs84 = CRS.from_epsg(4326)
            west, south, east, north = transform_bounds(
                ds.crs, wgs84,
                ds.bounds.left, ds.bounds.bottom,
                ds.bounds.right, ds.bounds.top,
            )
            return {"west": west, "south": south, "east": east, "north": north}
    except Exception as exc:
        logger.warning("Could not read GeoTIFF bounds: %s", exc)
        return None


# ── Tile Rendering (On-Demand) ──────────────────────────────────────────

def render_tile(job_id: str, z: int, x: int, y: int) -> bytes | None:
    """
    Render a single 256x256 PNG tile for a job's downloaded GeoTIFF result.
    Returns PNG bytes, or None if no result is available / tile is empty.

    Automatically detects product type (cropland/croptype/ndvi) and applies
    the appropriate colormap.
    """
    tiff_path = _get_result_path(job_id)
    if not tiff_path or not os.path.exists(tiff_path):
        logger.debug("No GeoTIFF for job %s", job_id)
        return None

    if not RASTERIO_AVAILABLE or not HAS_NUMPY or not HAS_PIL:
        logger.error("Missing dependencies for tile rendering (rasterio/numpy/Pillow)")
        return None

    # Look up product type from local store
    local_jobs = _load_job_store()
    job_info = local_jobs.get(job_id, {})
    product_type = job_info.get("product_type", "cropland")
    is_worldcereal = job_info.get("worldcereal", False)

    logger.debug("Rendering tile %s/%d/%d/%d (product=%s, wc=%s)",
                 job_id, z, x, y, product_type, is_worldcereal)

    # Try rio-tiler first (faster)
    if RIO_TILER_AVAILABLE:
        try:
            return _render_tile_rio_tiler(tiff_path, z, x, y, product_type, is_worldcereal)
        except Exception as exc:
            logger.debug("rio-tiler failed, falling back to rasterio: %s", exc)

    # Fallback: rasterio direct
    return _render_tile_rasterio(tiff_path, z, x, y, product_type, is_worldcereal)


def _find_band_index(tiff_path: str, band_name: str, fallback: int = 1) -> int:
    """Find a band index by its description/name in a GeoTIFF."""
    try:
        import rasterio as _rio
        with _rio.open(tiff_path) as ds:
            if ds.descriptions:
                for b in range(1, ds.count + 1):
                    desc = ds.descriptions[b - 1]
                    if desc and band_name in str(desc):
                        return b
            # Fallback to hardcoded index based on known band order
            if band_name == "landcover_classification" and ds.count >= 4:
                return 4
    except Exception:
        pass
    return fallback


def _render_tile_rio_tiler(tiff_path: str, z: int, x: int, y: int,
                           product_type: str, is_worldcereal: bool) -> bytes | None:
    """Render a tile using rio-tiler."""
    # For landcover, we need to read the landcover_classification band
    band_idx = None
    if product_type == "landcover" and is_worldcereal:
        band_idx = _find_band_index(tiff_path, "landcover_classification", fallback=4)

    with Reader(tiff_path) as src:
        try:
            if band_idx:
                img = src.tile(x, y, z, tilesize=256, indexes=(band_idx,))
            else:
                img = src.tile(x, y, z, tilesize=256)
        except Exception:
            return _transparent_tile()

        if is_worldcereal:
            # Classification output: integer labels, no rescaling needed
            # Build a colormap for integer values
            if product_type == "landcover":
                cm_dict = LANDCOVER_COLORMAP
            elif product_type == "cropland":
                cm_dict = CROPLAND_COLORMAP
            else:
                cm_dict = CROPTYPE_COLORMAP
            # rio-tiler expects colormap values as 0-255 uint8
            from rio_tiler.colormap import ColorMap
            color_map = ColorMap({k: (v[0], v[1], v[2]) for k, v in cm_dict.items()})
            # Clamp values to valid range
            img.rescale(in_range=((0, max(cm_dict.keys())),), out_range=((0, max(cm_dict.keys())),))
            png_bytes = img.render(img_format="PNG", colormap=color_map)
        else:
            # NDVI fallback: float values [-1, 1]
            cm = rio_cmap.get("rdylgn")
            img.rescale(in_range=((-1.0, 1.0),), out_range=((0, 255),))
            png_bytes = img.render(img_format="PNG", colormap=cm)

        return png_bytes


def _render_tile_rasterio(tiff_path: str, z: int, x: int, y: int,
                           product_type: str, is_worldcereal: bool) -> bytes | None:
    """Render a tile using rasterio directly (fallback)."""
    n = 2 ** z

    def merc_to_lat(yp):
        return math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * yp / n))))

    west  = x / n * 360.0 - 180.0
    east  = (x + 1) / n * 360.0 - 180.0
    north = merc_to_lat(y)
    south = merc_to_lat(y + 1)

    tile_size = 256

    with rasterio.open(tiff_path) as ds:
        file_west, file_south, file_east, file_north = transform_bounds(
            ds.crs, CRS.from_epsg(4326),
            ds.bounds.left, ds.bounds.bottom,
            ds.bounds.right, ds.bounds.top,
        )

        if east < file_west or west > file_east or north < file_south or south > file_north:
            return _transparent_tile()

        dst_transform = from_bounds(west, south, east, north, tile_size, tile_size)
        dst_data = np.full((tile_size, tile_size), np.nan, dtype=np.float32)

        # Select the right band based on product type
        # Band order in modified UDF output:
        #   1: cropland_classification
        #   2: probability_cropland
        #   3: probability_other
        #   4: landcover_classification  ← used for landcover product type
        #   5+: landcover_probability:* bands
        #   ... then croptype bands if applicable
        band_idx = 1
        if product_type == "landcover" and ds.count >= 4:
            # Try to find landcover_classification by band description
            for b in range(1, ds.count + 1):
                desc = ds.descriptions[b - 1] if ds.descriptions else None
                if desc and "landcover_classification" in str(desc):
                    band_idx = b
                    break
            else:
                # Fallback: band 4 is landcover_classification in standard order
                band_idx = 4

        reproject(
            source=rasterio.band(ds, band_idx),
            destination=dst_data,
            src_transform=ds.transform,
            src_crs=ds.crs,
            dst_transform=dst_transform,
            dst_crs=CRS.from_epsg(4326),
            resampling=Resampling.bilinear,
            src_nodata=ds.nodata,
            dst_nodata=np.nan,
        )

    if np.all(np.isnan(dst_data)):
        return _transparent_tile()

    # Colorize based on product type
    if is_worldcereal:
        rgba = _classification_to_rgba(dst_data, product_type)
    else:
        rgba = _ndvi_to_rgba(dst_data, vmin=-1.0, vmax=1.0)

    img = Image.fromarray(rgba, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _classification_to_rgba(data: np.ndarray, product_type: str) -> np.ndarray:
    """
    Convert 2D float32 classification array → RGBA uint8 array.
    Maps integer class labels to colors. NaN → transparent.
    """
    if product_type == "landcover":
        colormap = LANDCOVER_COLORMAP
    elif product_type == "cropland":
        colormap = CROPLAND_COLORMAP
    else:
        colormap = CROPTYPE_COLORMAP

    max_class = max(colormap.keys())
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Round to nearest integer class label
    int_data = np.round(data).astype(np.int32)

    for class_id, (r, g, b, a) in colormap.items():
        mask = int_data == class_id
        rgba[mask] = [r, g, b, a]

    # NaN → transparent
    nan_mask = np.isnan(data)
    rgba[nan_mask] = [0, 0, 0, 0]

    return rgba


def _ndvi_to_rgba(data: np.ndarray, vmin: float = -1.0, vmax: float = 1.0) -> np.ndarray:
    """
    Convert 2D float32 NDVI array → RGBA uint8 array.
    Uses a brown → yellow → green colormap. NaN → transparent.
    """
    lut = np.zeros((256, 4), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        if t < 0.5:
            s = t / 0.5
            r = int(139 + (255 - 139) * s)
            g = int(90 + (230 - 90) * s)
            b = int(43 + (0 - 43) * s)
        else:
            s = (t - 0.5) / 0.5
            r = int(255 + (0 - 255) * s)
            g = int(230 + (180 - 230) * s)
            b = int(0 + (0 - 0) * s)
        lut[i] = [r, g, b, 255]

    norm = (data - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0.0, 1.0)
    indices = (norm * 255).astype(np.uint8)

    rgba = lut[indices]

    nan_mask = np.isnan(data)
    rgba[nan_mask, 3] = 0

    return rgba


def _transparent_tile() -> bytes:
    """Return a 256x256 fully transparent PNG."""
    img = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Job List ────────────────────────────────────────────────────────────

def list_user_jobs(access_token: str, limit: int = 20) -> list:
    """List recent batch jobs for the authenticated user."""
    logger.info("Listing user jobs (limit=%d)...", limit)
    try:
        conn = connect_openeo(access_token)
        jobs = conn.list_jobs(limit=limit)
        logger.info("Found %d jobs", len(jobs))
        return [
            {
                "id": j.get("id"),
                "status": j.get("status"),
                "title": j.get("title"),
                "created": j.get("created"),
                "updated": j.get("updated"),
            }
            for j in jobs
        ]
    except Exception as exc:
        logger.error("Failed to list jobs: %s", exc)
        return []
