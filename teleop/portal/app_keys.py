"""Typed aiohttp app keys used across portal modules."""

from __future__ import annotations

from aiohttp import web

from teleop.portal.ice_config import BrowserIceConfig
from teleop.portal.settings import PortalAuthSettings

BROWSER_ICE_APP_KEY = web.AppKey("browser_ice", BrowserIceConfig)
PORTAL_AUTH_SETTINGS_APP_KEY = web.AppKey("portal_auth_settings", PortalAuthSettings)
