"""One-time host setup: udev rule and dialout group membership."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

UDEV_RULE_PATH = Path("/etc/udev/rules.d/99-krabby-mega.rules")
UDEV_RULE = (
    'SUBSYSTEM=="tty", ATTRS{idVendor}=="2341", ATTRS{idProduct}=="0042",'
    ' MODE="0666", GROUP="dialout"\n'
)

_PRO_CONTROLLER_UDEV_PATH = Path("/etc/udev/rules.d/99-krabby-pro-controller.rules")
_PRO_CONTROLLER_UDEV_RULE = (
    'SUBSYSTEM=="usb", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="2009",'
    ' MODE="0666", TAG+="uaccess"\n'
    'KERNEL=="hidraw*", ATTRS{idVendor}=="057e", ATTRS{idProduct}=="2009",'
    ' MODE="0666", TAG+="uaccess"\n'
)

_HID_NINTENDO_DKMS_REPO = "https://github.com/nicman23/dkms-hid-nintendo"
_HID_NINTENDO_MODULES_LOAD = Path("/etc/modules-load.d/hid_nintendo.conf")

_PRO_CONTROLLER_LED_UDEV_PATH = Path("/etc/udev/rules.d/99-krabby-pro-controller-led.rules")
_PRO_CONTROLLER_LED_UDEV_RULE = (
    'ACTION=="add", SUBSYSTEM=="leds", KERNEL=="*057E:2009*:player1",'
    ' RUN+="/bin/sh -c \'echo 1 > /sys%p/brightness\'"\n'
    'ACTION=="add", SUBSYSTEM=="leds", KERNEL=="*057E:2009*:player2",'
    ' RUN+="/bin/sh -c \'echo 0 > /sys%p/brightness\'"\n'
    'ACTION=="add", SUBSYSTEM=="leds", KERNEL=="*057E:2009*:player3",'
    ' RUN+="/bin/sh -c \'echo 0 > /sys%p/brightness\'"\n'
    'ACTION=="add", SUBSYSTEM=="leds", KERNEL=="*057E:2009*:player4",'
    ' RUN+="/bin/sh -c \'echo 0 > /sys%p/brightness\'"\n'
)

_BT_INPUT_CONF = Path("/etc/bluetooth/input.conf")
# Required on L4T 5.15-tegra: kernel has no BTPROTO_HIDP socket, so BlueZ must
# use the userspace uhid path. ClassicBondedOnly=false is needed because the Pro
# Controller sends store_hint=0 (does not persist its link key), which would
# otherwise cause the HIDP connection to be rejected as non-bonded.
_BT_INPUT_CONF_SETTINGS = {
    "UserspaceHID": "true",
    "ClassicBondedOnly": "false",
}


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd).returncode


def _ensure_udev_rule() -> bool:
    if UDEV_RULE_PATH.exists() and UDEV_RULE_PATH.read_text() == UDEV_RULE:
        print(f"[ok]  udev rule already in place: {UDEV_RULE_PATH}")
        return True
    try:
        UDEV_RULE_PATH.write_text(UDEV_RULE)
        _run(["udevadm", "control", "--reload-rules"])
        _run(["udevadm", "trigger"])
        print(f"[+]   wrote udev rule: {UDEV_RULE_PATH}")
        return True
    except PermissionError:
        print(f"[err] cannot write {UDEV_RULE_PATH} — run with sudo", file=sys.stderr)
        return False


def _ensure_pro_controller_udev() -> bool:
    if _PRO_CONTROLLER_UDEV_PATH.exists() and _PRO_CONTROLLER_UDEV_PATH.read_text() == _PRO_CONTROLLER_UDEV_RULE:
        print(f"[ok]  Pro Controller udev rule already in place: {_PRO_CONTROLLER_UDEV_PATH}")
        return True
    try:
        _PRO_CONTROLLER_UDEV_PATH.write_text(_PRO_CONTROLLER_UDEV_RULE)
        _run(["udevadm", "control", "--reload-rules"])
        _run(["udevadm", "trigger"])
        print(f"[+]   wrote Pro Controller udev rule: {_PRO_CONTROLLER_UDEV_PATH}")
        return True
    except PermissionError:
        print(f"[err] cannot write {_PRO_CONTROLLER_UDEV_PATH} — run with sudo", file=sys.stderr)
        return False


def _ensure_hid_nintendo() -> bool:
    try:
        if subprocess.run(["modinfo", "hid_nintendo"], capture_output=True).returncode == 0:
            print("[ok]  hid_nintendo kernel module already present")
            return True
    except FileNotFoundError:
        print("[skip] modinfo not found — skipping hid_nintendo setup (not Linux?)")
        return True

    print("      hid_nintendo not found — installing via DKMS ...")

    for pkg in ("dkms", "git"):
        if not shutil.which(pkg):
            print(f"      apt-installing {pkg} ...")
            if _run(["apt-get", "install", "-y", pkg]) != 0:
                print(f"[err] apt-get install {pkg} failed", file=sys.stderr)
                return False

    with tempfile.TemporaryDirectory() as tmpdir:
        src = Path(tmpdir) / "dkms-hid-nintendo"
        print(f"      cloning {_HID_NINTENDO_DKMS_REPO} ...")
        if _run(["git", "clone", "--depth=1", _HID_NINTENDO_DKMS_REPO, str(src)]) != 0:
            print("[err] git clone dkms-hid-nintendo failed", file=sys.stderr)
            return False

        pkg_name = None
        version = None
        for line in (src / "dkms.conf").read_text().splitlines():
            if line.startswith("PACKAGE_NAME") and pkg_name is None:
                pkg_name = line.split("=", 1)[1].strip().strip('"')
            if line.startswith("PACKAGE_VERSION") and version is None:
                version = line.split("=", 1)[1].strip().strip('"')
        if not pkg_name or not version:
            print("[err] could not parse PACKAGE_NAME/VERSION from dkms.conf", file=sys.stderr)
            return False

        if _run(["dkms", "add", str(src)]) != 0:
            print("[err] dkms add failed", file=sys.stderr)
            return False
        print(f"      building {pkg_name}/{version} (this may take a minute) ...")
        if _run(["dkms", "build", "-m", pkg_name, "-v", version]) != 0:
            print("[err] dkms build failed", file=sys.stderr)
            return False
        if _run(["dkms", "install", "-m", pkg_name, "-v", version]) != 0:
            print("[err] dkms install failed", file=sys.stderr)
            return False

    if _run(["modprobe", "hid_nintendo"]) != 0:
        print("[err] modprobe hid_nintendo failed", file=sys.stderr)
        return False

    try:
        _HID_NINTENDO_MODULES_LOAD.write_text("hid_nintendo\n")
        print(f"[+]   wrote {_HID_NINTENDO_MODULES_LOAD} (auto-load on boot)")
    except PermissionError:
        print(f"[err] cannot write {_HID_NINTENDO_MODULES_LOAD} — run with sudo", file=sys.stderr)
        return False

    print("[+]   hid_nintendo installed and loaded")
    return True


def _ensure_pro_controller_led_udev() -> bool:
    if _PRO_CONTROLLER_LED_UDEV_PATH.exists() and _PRO_CONTROLLER_LED_UDEV_PATH.read_text() == _PRO_CONTROLLER_LED_UDEV_RULE:
        print(f"[ok]  Pro Controller LED udev rule already in place: {_PRO_CONTROLLER_LED_UDEV_PATH}")
        return True
    try:
        _PRO_CONTROLLER_LED_UDEV_PATH.write_text(_PRO_CONTROLLER_LED_UDEV_RULE)
        _run(["udevadm", "control", "--reload-rules"])
        print(f"[+]   wrote Pro Controller LED udev rule: {_PRO_CONTROLLER_LED_UDEV_PATH}")
        return True
    except PermissionError:
        print(f"[err] cannot write {_PRO_CONTROLLER_LED_UDEV_PATH} — run with sudo", file=sys.stderr)
        return False


def _ensure_bt_input_conf() -> bool:
    if not _BT_INPUT_CONF.exists():
        print(f"[skip] {_BT_INPUT_CONF} not found — skipping Bluetooth HID config")
        return True

    try:
        text = _BT_INPUT_CONF.read_text()
    except PermissionError:
        print(f"[err] cannot read {_BT_INPUT_CONF} — run with sudo", file=sys.stderr)
        return False

    changed = False
    for key, value in _BT_INPUT_CONF_SETTINGS.items():
        active_pattern = re.compile(rf"^{key}={value}$", re.MULTILINE)
        if active_pattern.search(text):
            print(f"[ok]  {_BT_INPUT_CONF}: {key}={value} already set")
            continue
        # Replace commented or wrong-value line, or append if absent.
        any_line = re.compile(rf"^#?{key}=.*$", re.MULTILINE)
        if any_line.search(text):
            text = any_line.sub(f"{key}={value}", text)
        else:
            text = text.rstrip("\n") + f"\n{key}={value}\n"
        print(f"[+]   {_BT_INPUT_CONF}: set {key}={value}")
        changed = True

    if not changed:
        return True

    try:
        _BT_INPUT_CONF.write_text(text)
    except PermissionError:
        print(f"[err] cannot write {_BT_INPUT_CONF} — run with sudo", file=sys.stderr)
        return False

    ret = _run(["systemctl", "restart", "bluetooth"])
    if ret != 0:
        print("[err] failed to restart bluetooth service", file=sys.stderr)
        return False
    print("[+]   bluetooth service restarted")
    return True


def _ensure_dialout() -> None:
    user = os.environ.get("SUDO_USER") or os.environ.get("USER") or ""
    if not user:
        print("[skip] could not determine invoking user for dialout group")
        return
    result = subprocess.run(["groups", user], capture_output=True, text=True)
    if "dialout" in result.stdout:
        print(f"[ok]  {user} already in dialout group")
    else:
        ret = _run(["usermod", "-aG", "dialout", user])
        if ret == 0:
            print(f"[+]   added {user} to dialout group (re-login to take effect)")
        else:
            print(f"[err] usermod failed (exit {ret})", file=sys.stderr)


def run_host_setup() -> None:
    ok = _ensure_udev_rule()
    ok &= _ensure_pro_controller_udev()
    ok &= _ensure_pro_controller_led_udev()
    ok &= _ensure_hid_nintendo()
    ok &= _ensure_bt_input_conf()
    _ensure_dialout()
    if ok:
        print("\nHost setup complete. Replug your Mega boards.")
    else:
        print("\nHost setup incomplete — fix errors above and re-run.", file=sys.stderr)
        sys.exit(1)
