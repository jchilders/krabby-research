"""Unit tests for the teleop portal CLI entrypoint."""

from __future__ import annotations

from types import SimpleNamespace


def test_portal_main_uses_cli_host_port_and_bootstrap_builders(monkeypatch) -> None:
    from teleop.portal import main as portal_main

    browser_ice = {"iceServers": [{"urls": "stun:stun.example.com:19302"}]}
    auth_settings = {"http_token": "secret"}
    app = object()
    captured: dict[str, object] = {}

    class _FakeArgumentParser:
        def __init__(self, *args, **kwargs):
            captured["parser_description"] = kwargs.get("description")

        def add_argument(self, *args, **kwargs) -> None:
            return None

        def parse_args(self) -> SimpleNamespace:
            return SimpleNamespace(host="0.0.0.0", port=9010)

    def _fake_build_browser_ice_config() -> dict:
        return browser_ice

    def _fake_build_portal_auth_settings() -> dict:
        return auth_settings

    def _fake_create_portal_app(**kwargs):
        captured["create_kwargs"] = kwargs
        return app

    def _fake_run_app(run_app_arg, *, host: str, port: int, print):
        captured["run_app_arg"] = run_app_arg
        captured["host"] = host
        captured["port"] = port
        captured["print"] = print

    monkeypatch.setattr(portal_main.argparse, "ArgumentParser", _FakeArgumentParser)
    monkeypatch.setattr(portal_main, "build_browser_ice_config", _fake_build_browser_ice_config)
    monkeypatch.setattr(portal_main, "build_portal_auth_settings", _fake_build_portal_auth_settings)
    monkeypatch.setattr(portal_main, "create_portal_app", _fake_create_portal_app)
    monkeypatch.setattr(portal_main.web, "run_app", _fake_run_app)

    portal_main.main()

    assert captured["parser_description"] == "Krabby teleop portal — pair browser/edge signaling relay"
    assert captured["create_kwargs"] == {
        "browser_ice": browser_ice,
        "portal_auth_settings": auth_settings,
    }
    assert captured["run_app_arg"] is app
    assert captured["host"] == "0.0.0.0"
    assert captured["port"] == 9010
    assert captured["print"] is None
