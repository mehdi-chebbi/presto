"""
Flask WorldCereal Explorer — Main Application

Handles CDSE authentication via Resource Owner Password Grant (cdse-public client),
session management, openEO job submission, result download, on-demand tile serving,
and the map interface.
"""

import logging
import os
import sys
import time
import traceback
from functools import wraps
from pathlib import Path

import requests as http_requests
from dotenv import load_dotenv

# Load .env file BEFORE importing config so env vars are available
load_dotenv(Path(__file__).parent / ".env")

from flask import (  # noqa: E402
    Flask,
    redirect,
    url_for,
    session,
    request,
    jsonify,
    render_template,
    send_from_directory,
    send_file,
    g,
    Response,
)

from config import Config  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════
#  LOGGING SETUP
# ═══════════════════════════════════════════════════════════════════════════

def setup_logging():
    """Configure Python logging with colored console output."""
    fmt = logging.Formatter(
        fmt="%(asctime)s │ %(levelname)-7s │ %(name)-16s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(fmt)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)

    # Suppress noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("openeo.rest.auth.oidc").setLevel(logging.INFO)
    logging.getLogger("werkzeug").setLevel(logging.INFO)
    logging.getLogger("rasterio").setLevel(logging.WARNING)
    logging.getLogger("rio_tiler").setLevel(logging.WARNING)

    return logging.getLogger("app")


logger = setup_logging()


# ═══════════════════════════════════════════════════════════════════════════
#  CDSE Password Grant Helpers
# ═══════════════════════════════════════════════════════════════════════════

def authenticate_cdse_user(username: str, password: str) -> dict:
    """Authenticate a CDSE user via Resource Owner Password Grant."""
    logger.info("Authenticating user '%s' via CDSE Password Grant...", username)
    logger.debug("Token endpoint: %s", Config.CDSE_TOKEN_ENDPOINT)

    response = http_requests.post(
        Config.CDSE_TOKEN_ENDPOINT,
        data={
            "grant_type": "password",
            "client_id": "cdse-public",
            "username": username,
            "password": password,
            "scope": Config.CDSE_SCOPES,
        },
    )

    logger.info("CDSE auth response status: %d", response.status_code)

    if response.status_code != 200:
        logger.error("CDSE auth failed: %d %s", response.status_code, response.text[:300])
        response.raise_for_status()

    data = response.json()

    if "error" in data:
        logger.error("CDSE auth error: %s — %s", data["error"], data.get("error_description", ""))
        raise Exception(data.get("error_description", data["error"]))

    expires_in = data.get("expires_in", "unknown")
    has_refresh = "refresh_token" in data
    logger.info(
        "Auth successful for '%s'. Expires in %ss. Refresh token: %s",
        username, expires_in, "yes" if has_refresh else "no",
    )

    return data


def fetch_cdse_userinfo(access_token: str) -> dict:
    """Fetch user profile info from CDSE using an access token."""
    logger.debug("Fetching CDSE user info...")
    response = http_requests.get(
        Config.CDSE_USERINFO_ENDPOINT,
        headers={"Authorization": f"Bearer {access_token}"},
    )
    response.raise_for_status()
    info = response.json()
    logger.info("User info: %s (%s)", info.get("preferred_username", "?"), info.get("email", "?"))
    return info


def refresh_cdse_token(refresh_token: str) -> dict:
    """Refresh an expired access token using a refresh token."""
    logger.info("Refreshing CDSE access token...")
    response = http_requests.post(
        Config.CDSE_TOKEN_ENDPOINT,
        data={
            "grant_type": "refresh_token",
            "client_id": "cdse-public",
            "refresh_token": refresh_token,
            "scope": Config.CDSE_SCOPES,
        },
    )
    response.raise_for_status()
    data = response.json()
    if "error" in data:
        logger.error("Token refresh failed: %s", data)
        raise Exception(data.get("error_description", data["error"]))
    logger.info("Token refreshed successfully")
    return data


# ═══════════════════════════════════════════════════════════════════════════
#  App Factory
# ═══════════════════════════════════════════════════════════════════════════

def create_app() -> Flask:
    logger.info("Creating Flask application...")
    logger.info("Environment: DEBUG")

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_object(Config)
    app.secret_key = Config.SECRET_KEY

    # Ensure data directories exist
    for d in [Config.RESULTS_DIR, Config.JOBS_DIR]:
        os.makedirs(d, exist_ok=True)
        logger.debug("Ensured directory exists: %s", d)

    # ── Request Logging Middleware ────────────────────────────────────

    @app.before_request
    def log_request():
        """Log every incoming request."""
        g.request_start = time.time()
        # Don't spam logs for tile requests
        if "/tiles/" not in request.path:
            logger.info(
                ">> %s %s %s",
                request.method,
                request.full_path.rstrip("?"),
                request.content_type or "",
            )

    @app.after_request
    def log_response(response):
        """Log every outgoing response."""
        duration_ms = (time.time() - g.get("request_start", time.time())) * 1000
        # Don't spam logs for tile requests
        if "/tiles/" not in request.path:
            logger.info(
                "<< %s %s → %d (%.0fms)",
                request.method,
                request.full_path.rstrip("?"),
                response.status_code,
                duration_ms,
            )
            if response.status_code >= 400 and response.content_type == "application/json":
                try:
                    body = response.get_json()
                    if body:
                        logger.warning("  Error response body: %s", body)
                except Exception:
                    pass
        return response

    # ── Proactive Token Refresh (before_request) ─────────────────────

    # Routes that should be skipped from auto-refresh
    _REFRESH_SKIP_PATHS = {"/login", "/auth/login", "/auth/logout", "/favicon.ico"}

    @app.before_request
    def auto_refresh_token():
        """Proactively refresh the access token before it expires."""
        # Skip for non-authenticated routes
        if request.path in _REFRESH_SKIP_PATHS:
            return None
        if request.path.startswith("/static/") or request.path.startswith("/tiles/"):
            return None

        # Only act if user has a session
        if "user" not in session:
            return None

        # Check if token is still valid
        token_expiry = session.get("token_expiry", 0)
        if time.time() < token_expiry:
            return None  # Token still valid, nothing to do

        # Token expired or about to expire — try to refresh
        refresh_token = session.get("refresh_token")
        if not refresh_token:
            logger.warning("Token expired and no refresh_token available — redirecting to login")
            session.clear()
            if request.path.startswith("/api/"):
                return jsonify({"error": "Session expired. Please log in again."}), 401
            return redirect(url_for("login_page"))

        try:
            logger.info("Access token expired — refreshing proactively")
            token_data = refresh_cdse_token(refresh_token)
            session["access_token"] = token_data.get("access_token", "")
            new_refresh = token_data.get("refresh_token")
            if new_refresh:
                session["refresh_token"] = new_refresh
            session["token_expiry"] = time.time() + token_data.get("expires_in", 600) - 60
            session["refresh_expires_in"] = token_data.get("refresh_expires_in")
            logger.info("Proactive token refresh successful")
        except Exception as exc:
            logger.error("Proactive token refresh failed: %s — redirecting to login", exc)
            session.clear()
            if request.path.startswith("/api/"):
                return jsonify({"error": "Session expired. Please log in again."}), 401
            return redirect(url_for("login_page"))

    # ── 401 Retry Helper for API Routes ─────────────────────────────

    def _refresh_and_retry(access_token, api_func, *args, **kwargs):
        """
        Retry an API call once after refreshing the token.
        Used as a safety net when the before_request hook misses an expiry edge case.
        """
        refresh_token = session.get("refresh_token")
        if not refresh_token:
            return None  # Can't retry, let the original error propagate

        logger.warning("API call may have failed with auth error — attempting token refresh and retry")
        try:
            token_data = refresh_cdse_token(refresh_token)
            session["access_token"] = token_data.get("access_token", "")
            new_refresh = token_data.get("refresh_token")
            if new_refresh:
                session["refresh_token"] = new_refresh
            session["token_expiry"] = time.time() + token_data.get("expires_in", 600) - 60
            session["refresh_expires_in"] = token_data.get("refresh_expires_in")

            # Retry with the new access token
            return api_func(session["access_token"], *args, **kwargs)
        except Exception as retry_exc:
            logger.error("Retry after refresh also failed: %s", retry_exc)
            return None

    # ── Login Required Decorator ─────────────────────────────────────

    def login_required(f):
        """Redirect to login page if user has no active session."""
        @wraps(f)
        def decorated(*args, **kwargs):
            if "user" not in session:
                logger.info("Unauthenticated access to %s — redirecting to login", request.path)
                return redirect(url_for("login_page"))
            return f(*args, **kwargs)
        return decorated

    # ══════════════════════════════════════════════════════════════════
    #  Authentication Routes
    # ══════════════════════════════════════════════════════════════════

    @app.route("/login")
    def login_page():
        """Render the login page. If already logged in, go to map."""
        if "user" in session:
            logger.info("User already logged in, redirecting to map")
            return redirect(url_for("map_page"))
        return render_template("auth.html")

    @app.route("/auth/login", methods=["POST"])
    def auth_login():
        """Handle login form submission — authenticate via Password Grant."""
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if not username or not password:
            logger.warning("Login attempt with empty username or password")
            return render_template(
                "auth.html",
                error="Please enter your CDSE username and password.",
            )

        try:
            logger.info("Processing login for user: %s", username)
            token_data = authenticate_cdse_user(username, password)
            user_info = fetch_cdse_userinfo(token_data["access_token"])

            session["user"] = {
                "sub": user_info.get("sub", ""),
                "preferred_username": user_info.get("preferred_username", username),
                "email": user_info.get("email", ""),
                "name": user_info.get("name", ""),
            }
            session["access_token"] = token_data.get("access_token", "")
            session["refresh_token"] = token_data.get("refresh_token", "")
            session["token_expiry"] = time.time() + token_data.get("expires_in", 600) - 60
            session["refresh_expires_in"] = token_data.get("refresh_expires_in")
            logger.info(
                "Token stored: expires_in=%ss, refresh_expires_in=%ss, offline=%s",
                token_data.get("expires_in"),
                token_data.get("refresh_expires_in"),
                token_data.get("refresh_expires_in") == 0,
            )

            logger.info("Login successful for '%s' — redirecting to map", username)
            return redirect(url_for("map_page"))

        except http_requests.exceptions.HTTPError as exc:
            status = exc.response.status_code
            logger.error("Login HTTP error (%d): %s", status, exc.response.text[:300])
            if status == 401:
                return render_template(
                    "auth.html",
                    error="Invalid username or password. Please check your CDSE credentials.",
                )
            return render_template(
                "auth.html",
                error=f"CDSE auth error ({status}): {exc.response.text[:200]}",
            )

        except Exception as exc:
            logger.error("Login failed: %s\n%s", exc, traceback.format_exc())
            return render_template(
                "auth.html",
                error=f"Authentication failed: {exc}",
            )

    @app.route("/auth/logout", methods=["POST", "GET"])
    def auth_logout():
        """Clear local session and invalidate CDSE remote session."""
        username = session.get("user", {}).get("preferred_username", "unknown")
        refresh_token = session.get("refresh_token")

        # Invalidate the CDSE remote session
        if refresh_token:
            try:
                http_requests.post(
                    Config.CDSE_LOGOUT_ENDPOINT,
                    data={
                        "client_id": "cdse-public",
                        "refresh_token": refresh_token,
                    },
                    timeout=5,
                )
                logger.info("CDSE remote session invalidated for '%s'", username)
            except Exception as exc:
                logger.warning("Failed to invalidate CDSE session (non-critical): %s", exc)

        session.clear()
        logger.info("User '%s' logged out", username)
        return redirect(url_for("login_page"))

    @app.route("/auth/me")
    @login_required
    def auth_me():
        """Return current user info as JSON (for debugging)."""
        return jsonify({
            "user": session.get("user", {}),
            "has_access_token": bool(session.get("access_token")),
            "token_preview": session.get("access_token", "")[:15] + "..." if session.get("access_token") else None,
        })

    @app.route("/auth/refresh")
    @login_required
    def auth_refresh():
        """Refresh the access token if it's about to expire. Returns JSON status."""
        # Only refresh if token expires within the next 2 minutes
        # This avoids unnecessary refresh token rotations when the heartbeat calls this endpoint
        token_expiry = session.get("token_expiry", 0)
        if time.time() < token_expiry - 120:
            return jsonify({"status": "ok", "message": "Token still valid"})

        refresh_token = session.get("refresh_token")
        if not refresh_token:
            logger.warning("Token refresh requested but no refresh_token in session")
            return jsonify({"error": "No refresh token available"}), 400
        try:
            token_data = refresh_cdse_token(refresh_token)
            session["access_token"] = token_data.get("access_token", "")
            new_refresh = token_data.get("refresh_token")
            if new_refresh:
                session["refresh_token"] = new_refresh
            session["token_expiry"] = time.time() + token_data.get("expires_in", 600) - 60
            session["refresh_expires_in"] = token_data.get("refresh_expires_in")
            logger.info("Token refreshed successfully (expires_in=%ss, refresh_expires_in=%ss)",
                        token_data.get("expires_in"), token_data.get("refresh_expires_in"))
            return jsonify({"status": "ok"})
        except Exception as exc:
            logger.error("Token refresh failed: %s", exc)
            return jsonify({"error": str(exc)}), 401

    # ══════════════════════════════════════════════════════════════════
    #  Main Page Routes
    # ══════════════════════════════════════════════════════════════════

    @app.route("/")
    def index():
        """Root redirects to map if logged in, otherwise to login."""
        if "user" in session:
            return redirect(url_for("map_page"))
        return redirect(url_for("login_page"))

    @app.route("/map")
    @login_required
    def map_page():
        """The main map interface (Leaflet + polygon drawing)."""
        return render_template(
            "map.html",
            user=session.get("user", {}),
        )

    # ══════════════════════════════════════════════════════════════════
    #  API Routes — WorldCereal Job Submission & Status
    # ══════════════════════════════════════════════════════════════════

    @app.route("/api/submit", methods=["POST"])
    @login_required
    def submit_job():
        """
        Submit a WorldCereal classification job.

        Expects JSON body:
        {
            "geometry": { ... },
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "product_type": "cropland"
        }
        """
        logger.info("=== /api/submit called ===")

        body = request.get_json(silent=True)
        if not body:
            logger.error("No JSON body in request")
            return jsonify({"error": "Request body must be JSON"}), 400

        logger.info("Request body keys: %s", list(body.keys()))

        geometry = body.get("geometry")
        start_date = body.get("start_date")
        end_date = body.get("end_date")
        product_type = body.get("product_type", "cropland")

        if not geometry:
            logger.error("Missing 'geometry' field in request body")
            return jsonify({"error": "'geometry' field is required"}), 400

        if not start_date or not end_date:
            logger.error("Missing date fields: start_date=%s, end_date=%s", start_date, end_date)
            return jsonify({"error": "'start_date' and 'end_date' are required"}), 400

        logger.info("Validated: geometry=%s, dates=%s→%s, product=%s",
                     geometry.get("type"), start_date, end_date, product_type)

        access_token = session.get("access_token")
        if not access_token:
            logger.error("No access_token in session — user may need to re-login")
            return jsonify({
                "error": "No access token found. Your session may have expired. Please log in again.",
            }), 401

        try:
            from tasks import submit_classification_job

            result = submit_classification_job(
                access_token=access_token,
                geometry=geometry,
                start_date=start_date,
                end_date=end_date,
                product_type=product_type,
            )

            logger.info("Job submitted successfully! job_id=%s", result.get("job_id"))
            return jsonify(result), 202

        except ValueError as exc:
            logger.error("Validation error: %s", exc)
            return jsonify({"error": str(exc)}), 400

        except RuntimeError as exc:
            error_str = str(exc).lower()
            if "401" in error_str or "unauthorized" in error_str or "authentication" in error_str:
                # 401 safety net: refresh token and retry once
                retry_result = _refresh_and_retry(
                    access_token, submit_classification_job,
                    geometry=geometry, start_date=start_date, end_date=end_date, product_type=product_type,
                )
                if retry_result is not None:
                    logger.info("Job submitted successfully on retry! job_id=%s", retry_result.get("job_id"))
                    return jsonify(retry_result), 202

            logger.error("Job submission runtime error: %s\n%s", exc, traceback.format_exc())
            return jsonify({"error": str(exc)}), 502

        except Exception as exc:
            logger.error("Unexpected error in job submission:\n%s", traceback.format_exc())
            return jsonify({
                "error": f"Internal server error: {exc}",
                "detail": traceback.format_exc()[-500:],
            }), 500

    @app.route("/api/status/<job_id>")
    @login_required
    def job_status(job_id: str):
        """
        Check the status of a submitted WorldCereal batch job.
        Returns JSON with status, progress, and whether results are downloaded.
        """
        logger.info("=== /api/status/%s called ===", job_id)

        access_token = session.get("access_token")
        if not access_token:
            logger.error("No access_token in session")
            return jsonify({"error": "Session expired. Please log in again."}), 401

        try:
            from tasks import get_job_status

            result = get_job_status(access_token, job_id)
            return jsonify(result)

        except Exception as exc:
            error_str = str(exc).lower()
            if "401" in error_str or "unauthorized" in error_str:
                # 401 safety net: refresh token and retry once
                retry_result = _refresh_and_retry(access_token, get_job_status, job_id)
                if retry_result is not None:
                    return jsonify(retry_result)

            logger.error("Error checking job status:\n%s", traceback.format_exc())
            return jsonify({
                "job_id": job_id,
                "status": "error",
                "error": str(exc),
            }), 500

    @app.route("/api/download/<job_id>")
    @login_required
    def download_result(job_id: str):
        """
        Download the GeoTIFF result for a finished job.
        This triggers the actual download from CDSE and saves locally.
        The frontend calls this once after the job status becomes "finished".
        """
        logger.info("=== /api/download/%s called ===", job_id)

        access_token = session.get("access_token")
        if not access_token:
            return jsonify({"error": "Session expired. Please log in again."}), 401

        try:
            from tasks import download_job_result, _load_job_store

            result = download_job_result(access_token, job_id)

            if "error" in result:
                return jsonify(result), 400

            # Include product_type and worldcereal flag for frontend legend
            local_jobs = _load_job_store()
            job_info = local_jobs.get(job_id, {})
            result["product_type"] = job_info.get("product_type", "cropland")
            result["worldcereal"] = job_info.get("worldcereal", False)

            return jsonify(result)

        except Exception as exc:
            error_str = str(exc).lower()
            if "401" in error_str or "unauthorized" in error_str:
                # 401 safety net: refresh token and retry once
                retry_result = _refresh_and_retry(access_token, download_job_result, job_id)
                if retry_result is not None:
                    # Add product_type and worldcereal flag for frontend legend
                    local_jobs = _load_job_store()
                    job_info = local_jobs.get(job_id, {})
                    retry_result["product_type"] = job_info.get("product_type", "cropland")
                    retry_result["worldcereal"] = job_info.get("worldcereal", False)
                    return jsonify(retry_result)

            logger.error("Error downloading results:\n%s", traceback.format_exc())
            return jsonify({
                "job_id": job_id,
                "error": str(exc),
            }), 500

    @app.route("/api/jobs")
    @login_required
    def list_jobs():
        """List recent jobs for the authenticated user."""
        logger.info("=== /api/jobs called ===")

        access_token = session.get("access_token")
        if not access_token:
            return jsonify({"error": "Session expired. Please log in again."}), 401

        try:
            from tasks import list_user_jobs

            jobs = list_user_jobs(access_token)
            return jsonify({"jobs": jobs})

        except Exception as exc:
            logger.error("Error listing jobs: %s\n%s", exc, traceback.format_exc())
            return jsonify({"error": str(exc)}), 500

    @app.route("/api/jobs/local")
    @login_required
    def list_local_jobs():
        """List jobs from the local job store (with result_path, bounds, etc.)."""
        logger.info("=== /api/jobs/local called ===")

        try:
            from tasks import _load_job_store

            local_jobs = _load_job_store()
            # Convert to a list sorted by created_at descending
            job_list = []
            for job_id, info in local_jobs.items():
                entry = {"job_id": job_id, **info}
                # Check if the result file still exists on disk
                result_path = info.get("result_path")
                if result_path:
                    entry["result_available"] = Path(result_path).exists()
                else:
                    entry["result_available"] = False
                job_list.append(entry)

            # Sort by created_at descending (newest first)
            job_list.sort(key=lambda j: j.get("created_at", ""), reverse=True)

            return jsonify({"jobs": job_list})

        except Exception as exc:
            logger.error("Error listing local jobs: %s\n%s", exc, traceback.format_exc())
            return jsonify({"error": str(exc)}), 500

    # ══════════════════════════════════════════════════════════════════
    #  Tile Serving — On-Demand from GeoTIFF
    # ══════════════════════════════════════════════════════════════════

    @app.route("/tiles/<job_id>/<int:z>/<int:x>/<int:y>.png")
    @login_required
    def serve_tile(job_id: str, z: int, x: int, y: int):
        """
        Serve a single XYZ map tile, rendered on-the-fly from the job's GeoTIFF.

        This does NOT read from pre-generated tile files.
        Instead, it uses rasterio/rio-tiler to read the relevant pixels
        from the GeoTIFF and colorize them into a 256x256 PNG.
        """
        try:
            from tasks import render_tile

            png_bytes = render_tile(job_id, z, x, y)

            if png_bytes is None:
                # No result available yet — return transparent tile
                from tasks import _transparent_tile
                png_bytes = _transparent_tile()

            return Response(png_bytes, mimetype="image/png")

        except Exception as exc:
            logger.error("Tile render error: %s/%d/%d/%d — %s", job_id, z, x, y, exc)
            # Return transparent tile on error
            try:
                from tasks import _transparent_tile
                return Response(_transparent_tile(), mimetype="image/png")
            except Exception:
                return "", 204

    # ══════════════════════════════════════════════════════════════════
    #  Error Handlers
    # ══════════════════════════════════════════════════════════════════

    @app.errorhandler(404)
    def not_found(e):
        logger.info("404 Not Found: %s %s", request.method, request.path)
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def server_error(e):
        logger.error("500 Server Error: %s\n%s", request.path, traceback.format_exc())
        return render_template("500.html"), 500

    # ══════════════════════════════════════════════════════════════════
    #  Favicon
    # ══════════════════════════════════════════════════════════════════

    @app.route("/favicon.ico")
    def favicon():
        return send_from_directory(
            os.path.join(app.static_folder, "img"),
            "favicon.svg",
            mimetype="image/svg+xml",
        )

    logger.info("Flask application created successfully")
    return app


# ═══════════════════════════════════════════════════════════════════════════
#  Run directly (dev only)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    app = create_app()
    logger.info("Starting development server on %s:%d", Config.APP_HOST, Config.APP_PORT)
    app.run(
        host=Config.APP_HOST,
        port=Config.APP_PORT,
        debug=True,
    )
