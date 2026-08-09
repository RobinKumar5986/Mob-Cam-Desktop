#File Info: name = adb_wifi.py

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
    try:
        out = subprocess.run(["adb", *args], capture_output=True, text=True, timeout=15)
        return out.returncode == 0, (out.stdout + out.stderr).strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        return False, str(e)


class _NameListener(ServiceListener):
    # fires on_found(host, port) the first time a matching service appears
    def __init__(self, match, on_found):
        self.match = match
        self.on_found = on_found
        self.found = False

    def add_service(self, zc, type_, name):
        if self.found or self.match not in name:
            return
        info = zc.get_service_info(type_, name)
        if info is None:
            return
        self.found = True
        self.on_found(socket.inet_ntoa(info.addresses[0]), info.port)

    def update_service(self, zc, type_, name):
        pass

    def remove_service(self, zc, type_, name):
        pass


def wait_for_pairing_service(service_name, on_found, timeout=90):
    # phone broadcasts this once the QR code is scanned
    zc = Zeroconf()
    listener = _NameListener(service_name, on_found)
    browser = ServiceBrowser(zc, PAIRING_SERVICE, listener)

    start = time.time()
    while not listener.found and time.time() - start < timeout:
        time.sleep(0.3)

    browser.cancel()
    zc.close()
    if not listener.found:
        on_found(None, None)


def wait_for_connect_service(on_found, timeout=15):
    # advertised continuously while wireless debugging is on, gives us the
    # host:port to `adb connect` without asking the user for it
    zc = Zeroconf()
    listener = _NameListener("", on_found)
    browser = ServiceBrowser(zc, CONNECT_SERVICE, listener)

    start = time.time()
    while not listener.found and time.time() - start < timeout:
        time.sleep(0.3)

    browser.cancel()
    zc.close()
    if not listener.found:
        on_found(None, None)