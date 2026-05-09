"""
Configuration for the WorldCereal Flask application.
Set these via environment variables or a .env file.
"""

import os


class Config:
    """Base configuration."""

    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "change-me-in-production")

    # ── CDSE OAuth2 / OIDC ──────────────────────────────────────────────
    # Register your client at: https://shapps.dataspace.copernicus.eu/dashboard/
    #   User Settings -> OAuth Clients -> Create New Client
    #   Set redirect URI to: http://localhost:5000/auth/callback

    CDSE_CLIENT_ID = os.environ.get("CDSE_CLIENT_ID", "")
    CDSE_CLIENT_SECRET = os.environ.get("CDSE_CLIENT_SECRET", "")

    CDSE_ISSUER = (
        "https://identity.dataspace.copernicus.eu"
        "/auth/realms/CDSE"
    )
    CDSE_AUTH_ENDPOINT = (
        "https://identity.dataspace.copernicus.eu"
        "/auth/realms/CDSE/protocol/openid-connect/auth"
    )
    CDSE_TOKEN_ENDPOINT = (
        "https://identity.dataspace.copernicus.eu"
        "/auth/realms/CDSE/protocol/openid-connect/token"
    )
    CDSE_USERINFO_ENDPOINT = (
        "https://identity.dataspace.copernicus.eu"
        "/auth/realms/CDSE/protocol/openid-connect/userinfo"
    )
    CDSE_LOGOUT_ENDPOINT = (
        "https://identity.dataspace.copernicus.eu"
        "/auth/realms/CDSE/protocol/openid-connect/logout"
    )
    CDSE_DISCOVERY_URL = (
        "https://identity.dataspace.copernicus.eu"
        "/auth/realms/CDSE/.well-known/openid-configuration"
    )

    CDSE_SCOPES = "openid profile email offline_access"

    # ── Flask settings ──────────────────────────────────────────────────
    APP_HOST = os.environ.get("APP_HOST", "localhost")
    APP_PORT = int(os.environ.get("APP_PORT", 5000))

    # ── openEO ──────────────────────────────────────────────────────────
    OPENEO_BACKEND = "openeo.dataspace.copernicus.eu"

    # ── WorldCereal defaults ────────────────────────────────────────────
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "data", "results")
    JOBS_DIR = os.path.join(os.path.dirname(__file__), "data", "jobs")
    TILES_DIR = os.path.join(os.path.dirname(__file__), "static", "tiles")

    # ── Tile generation ─────────────────────────────────────────────────
    GDAL_ZOOM_RANGE = "8-14"
