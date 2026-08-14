# Mob Cam — virtual camera setup

## Architecture

```
phone (JPEG over adb-forwarded TCP)
        │
        ▼
frame_receiver.py      socket + JPEG decode, auto-reconnect  → BGR numpy frames
        │
        ▼
image_processing.py    rotate → mirror → crop → fit → shape mask   (add stages here)
        │
        ├──────────────► receiver_gui.py   local preview window (optional, cosmetic)
        ▼
virtual_camera.py      pyvirtualcam → real OS camera device
        │
        ▼
Zoom / Meet / Teams / OBS / browsers
```

`stream_pipeline.py` wires the three together and owns the threads. The virtual
camera runs on its own sender thread behind a depth-1 queue, so a slow driver
drops frames instead of stalling the socket reader.

## Install

```
pip install -r requirements.txt
```

## Driver, per OS

The OS is detected at runtime (`platform.system()`) and the backend is chosen
from a table in `virtual_camera.py`. You still need the loopback driver
installed once.

### Windows
Install [OBS Studio](https://obsproject.com) and launch it once — that registers
the OBS Virtual Camera DirectShow/MF driver system-wide. OBS does not need to
stay open afterwards. UnityCapture works as a fallback.

### macOS
Install OBS Studio 26.1 or newer, open it once and click **Start Virtual
Camera** so macOS registers the device. The first time an app uses it, allow the
camera extension in **System Settings → Privacy & Security**.

### Linux
```
sudo apt install v4l2loopback-dkms v4l-utils
sudo modprobe v4l2loopback devices=1 video_nr=10 \
     card_label='Mob Cam' exclusive_caps=1
```
`exclusive_caps=1` is not optional — without it Chrome and Firefox refuse to
list the device. To persist it, add `v4l2loopback` to `/etc/modules-load.d/` and
the options to `/etc/modprobe.d/v4l2loopback.conf`.

The app probes for a driver at startup. If none is found, the toggle is
disabled and **Setup help** shows the instructions for the detected OS plus the
underlying driver error.

## Notes

- Output resolution is fixed per device session. Changing it in the UI makes
  `VirtualCamera.send()` notice the size change and transparently reopen the
  device — apps already consuming the feed may need to reselect the camera.
- Odd pixel dimensions are rounded down to even numbers, since some drivers'
  YUV conversion breaks on odd sizes.
- Virtual cameras carry **no alpha channel**. The circle shape is therefore sent
  as a circle on solid green; the receiving app has to chroma-key it if you want
  a true cut-out. For a real transparent overlay, keep using the preview window
  as an OBS window-capture source with a chroma key filter.
- Closing the preview window no longer stops the stream — the camera keeps
  publishing.

## Adding a processing stage

Write `fn(frame, config) -> frame` on BGR numpy arrays and register it:

```python
def blur_background(frame, config):
    ...
    return frame

stream.pipeline.insert_before("fit_output", "blur_bg", blur_background)
```

Stages that raise are logged and skipped, so an experimental filter cannot take
the camera down mid-call. Put per-stage options in `config.extra`.