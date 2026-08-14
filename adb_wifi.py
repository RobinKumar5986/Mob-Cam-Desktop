"""ADB wireless-debugging helpers.

Handles the mDNS side of Android's wireless debugging so the user never has to
type an IP address:

  * QR pairing  - we invent a service name + password, render it as a QR code
                  (in adb_wifi_dialog), then wait for the phone to broadcast
                  _adb-tls-pairing._tcp once the code is scanned.
  * Connect     - _adb-tls-connect._tcp is advertised continuously while
                  wireless debugging is on, which gives us the host:port to
                  `adb connect` without asking for it.
"""

import random
import string
import socket
import subprocess
import time

from zeroconf import Zeroconf, ServiceBrowser, ServiceListener

PAIRING_SERVICE = "_adb-tls-pairing._tcp.local."
CONNECT_SERVICE = "_adb-tls-connect._tcp.local."


def random_service_name():
    return "mobcam_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


def random_password():
    return "".join(random.choices(string.ascii_letters + string.digits, k=12))


def run_adb(*args):
    """Run an adb command. Returns (ok, combined_output)."""
    try:
        out = subprocess.run(["adb", *args], capture_output=True, text=True, timeout=15)
        return out.returncode == 0, (out.stdout + out.stderr).strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, str(e)


class _NameListener(ServiceListener):
    """Fires on_found(host, port) the first time a matching service appears.

    `match` is a substring test against the service name; pass "" to accept the
    first service of the browsed type.
    """

    def __init__(self, match, on_found):
        self.match = match
        self.on_found = on_found
        self.found = False

    def add_service(self, zc, type_, name):
        if self.found or self.match not in name:
            return
        info = zc.get_service_info(type_, name)
        if info is None or not info.addresses:
            return
        self.found = True
        self.on_found(socket.inet_ntoa(info.addresses[0]), info.port)

    def update_service(self, zc, type_, name):
        pass

    def remove_service(self, zc, type_, name):
        pass


def _browse_until_found(service_type, match, on_found, timeout):
    """Browse `service_type` until a match turns up or the timeout expires.

    Calls on_found(None, None) on timeout so callers always get exactly one
    callback and can report failure without a separate timer.
    """
    zc = Zeroconf()
    listener = _NameListener(match, on_found)
    browser = ServiceBrowser(zc, service_type, listener)

    start = time.time()
    try:
        while not listener.found and time.time() - start < timeout:
            time.sleep(0.3)
    finally:
        browser.cancel()
        zc.close()

    if not listener.found:
        on_found(None, None)


def wait_for_pairing_service(service_name, on_found, timeout=90):
    # phone broadcasts this once the QR code is scanned
    _browse_until_found(PAIRING_SERVICE, service_name, on_found, timeout)


def wait_for_connect_service(on_found, timeout=15):
    # advertised continuously while wireless debugging is on, gives us the
    # host:port to `adb connect` without asking the user for it
    _browse_until_found(CONNECT_SERVICE, "", on_found, timeout)