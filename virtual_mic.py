"""Virtual microphone device.

The counterpart of virtual_camera. That module leans on v4l2loopback to publish
a camera device; this one publishes a *capture* device so the phone's audio
appears in every app's microphone list exactly the way a headset mic does.

Two PulseAudio / PipeWire modules do the work:

    module-null-sink     a sink nothing is wired to, so writing to it is silent
                         on the speakers
    module-remap-source  wraps that sink's monitor and presents it as a plain
                         input device

The remap step is the part that matters. A bare null sink only gives you
"Monitor of <sink>", which some applications hide and most users never find
under that name. Remapping produces a real source called "Mob Cam Microphone"
that sits in the list next to the built-in mic.

Windows and macOS have no equivalent command line, so there the module only
reports what to install; the driver itself is a manual step.
"""

from __future__ import annotations

import platform
import shutil
import subprocess
from typing import List, Tuple

SINK_NAME = "mobcam_sink"
SOURCE_NAME = "mobcam_mic"
SINK_DESCRIPTION = "Mob_Cam_Output"
SOURCE_DESCRIPTION = "Mob_Cam_Microphone"

# What the user sees in their conferencing app. Underscores in the description
# above become spaces there, which is a PulseAudio quirk, not a typo.
FRIENDLY_NAME = "Mob Cam Microphone"


def current_os() -> str:
    system = platform.system()
    return {"Linux": "linux", "Darwin": "macos", "Windows": "windows"}.get(
        system, system.lower())


def has_pactl() -> bool:
    """True when the PulseAudio / PipeWire control tool is available."""
    return shutil.which("pactl") is not None


def can_create() -> bool:
    """True when Mob Cam can build the device itself, no manual install."""
    return current_os() == "linux" and has_pactl()


def _pactl(*args: str) -> Tuple[bool, str]:
    """Run pactl, returning success and the combined output."""
    try:
        result = subprocess.run(
            ["pactl", *args], capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, output.strip()


def list_sources() -> List[str]:
    """Names of every capture device the sound server knows about."""
    ok, output = _pactl("list", "short", "sources")
    if not ok:
        return []
    names = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            names.append(parts[1])
    return names


def list_sinks() -> List[str]:
    ok, output = _pactl("list", "short", "sinks")
    if not ok:
        return []
    names = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) >= 2:
            names.append(parts[1])
    return names


def is_installed() -> bool:
    """True when the Mob Cam microphone is currently loaded."""
    if not can_create():
        return False
    return SOURCE_NAME in list_sources() and SINK_NAME in list_sinks()


def sink_target() -> str:
    """The sink audio must be written to, or '' when the device is absent."""
    return SINK_NAME if is_installed() else ""


def create() -> Tuple[bool, str]:
    """Load the two modules. Safe to call when they are already loaded."""
    if not can_create():
        return False, (
            "Automatic setup needs pactl (PulseAudio or PipeWire) and is only "
            "available on Linux. See Setup help for the manual steps.")

    if is_installed():
        return True, f"'{FRIENDLY_NAME}' is already available."

    sinks = list_sinks()
    if SINK_NAME not in sinks:
        ok, output = _pactl(
            "load-module", "module-null-sink",
            f"sink_name={SINK_NAME}",
            f"sink_properties=device.description={SINK_DESCRIPTION}",
        )
        if not ok:
            return False, f"Could not create the audio sink:\n{output}"

    if SOURCE_NAME not in list_sources():
        ok, output = _pactl(
            "load-module", "module-remap-source",
            f"master={SINK_NAME}.monitor",
            f"source_name={SOURCE_NAME}",
            f"source_properties=device.description={SOURCE_DESCRIPTION}",
        )
        if not ok:
            # Roll the sink back so a half-built device is not left behind.
            remove()
            return False, f"Could not create the microphone source:\n{output}"

    if not is_installed():
        return False, "The modules loaded but the device did not appear."

    return True, (
        f"'{FRIENDLY_NAME}' created. Select it as the microphone in Zoom, "
        "Meet, Teams, OBS or your browser.\n\n"
        "It lasts until you log out. To make it permanent, add these to "
        "~/.config/pulse/default.pa:\n\n"
        f"  load-module module-null-sink sink_name={SINK_NAME} "
        f"sink_properties=device.description={SINK_DESCRIPTION}\n"
        f"  load-module module-remap-source master={SINK_NAME}.monitor "
        f"source_name={SOURCE_NAME} "
        f"source_properties=device.description={SOURCE_DESCRIPTION}")


def remove() -> Tuple[bool, str]:
    """Unload both modules, ignoring whichever is already gone."""
    if not can_create():
        return False, "pactl is not available."

    messages = []
    for module, key in (("module-remap-source", f"source_name={SOURCE_NAME}"),
                        ("module-null-sink", f"sink_name={SINK_NAME}")):
        index = _module_index(module, key)
        if index is None:
            continue
        ok, output = _pactl("unload-module", str(index))
        if not ok:
            messages.append(output)

    if messages:
        return False, "\n".join(messages)
    return True, f"'{FRIENDLY_NAME}' removed."


def _module_index(module: str, key: str):
    """Index of a loaded module whose arguments contain key, if any."""
    ok, output = _pactl("list", "short", "modules")
    if not ok:
        return None
    for line in output.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[1] == module and key in line:
            try:
                return int(parts[0])
            except ValueError:
                continue
    return None


def probe() -> Tuple[bool, str]:
    """Whether a usable virtual microphone exists, with a message for the UI."""
    system = current_os()
    if system == "linux":
        if is_installed():
            return True, f"'{FRIENDLY_NAME}' ready"
        if can_create():
            return False, "no virtual mic yet - click Create virtual mic"
        return False, "no virtual mic, and pactl is unavailable"
    # Elsewhere the driver is installed by hand; audio_output detects it by name.
    return False, "install a loopback driver - click Setup help"


def setup_instructions() -> str:
    """Per-OS instructions for getting a microphone device."""
    system = current_os()
    if system == "windows":
        return (
            "Virtual microphone - Windows\n"
            "----------------------------\n"
            "1. Install VB-CABLE (free): https://vb-audio.com/Cable/\n"
            "2. Reboot.\n"
            "3. In Mob Cam, set 'Send audio to' to\n"
            "   'CABLE Input (VB-Audio Virtual Cable)'.\n"
            "4. In Zoom / Meet / Teams / OBS, choose the MICROPHONE\n"
            "   'CABLE Output (VB-Audio Virtual Cable)'.\n\n"
            "The two names are different on purpose: Mob Cam writes to the\n"
            "input end of the cable, your app listens at the output end."
        )
    if system == "macos":
        return (
            "Virtual microphone - macOS\n"
            "--------------------------\n"
            "1. Install BlackHole 2ch:\n"
            "     brew install blackhole-2ch\n"
            "   or download from https://existential.audio/blackhole/\n"
            "2. In Mob Cam, set 'Send audio to' to 'BlackHole 2ch'.\n"
            "3. In your app, choose the MICROPHONE 'BlackHole 2ch'.\n\n"
            "Tick 'Also play on my speakers' to hear it yourself as well."
        )
    return (
        "Virtual microphone - Linux\n"
        "--------------------------\n"
        "Click 'Create virtual mic' and Mob Cam does all of this for you.\n"
        "It runs, in effect:\n\n"
        f"  pactl load-module module-null-sink sink_name={SINK_NAME} \\\n"
        f"        sink_properties=device.description={SINK_DESCRIPTION}\n"
        f"  pactl load-module module-remap-source master={SINK_NAME}.monitor \\\n"
        f"        source_name={SOURCE_NAME} \\\n"
        f"        source_properties=device.description={SOURCE_DESCRIPTION}\n\n"
        f"Then pick '{FRIENDLY_NAME}' as the microphone in your app.\n\n"
        "Do NOT choose your sound card here (anything like 'HDA Intel PCH' or\n"
        "'hw:0,0'). That is the speakers - audio sent there is audible but no\n"
        "application can record from it.\n\n"
        "The device lasts until you log out. Add the two load-module lines to\n"
        "~/.config/pulse/default.pa to keep it.\n\n"
        "ALSA-only systems, no PulseAudio or PipeWire:\n"
        "  sudo modprobe snd-aloop\n"
        "  Send audio to the 'Loopback' playback device and record from the\n"
        "  matching capture device."
    )