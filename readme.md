# Mob Cam — desktop setup

## Architecture

```
phone camera
    │  (Use PC OFF)  phone applies effects            (Use PC ON) raw upright JPEG
    ▼
data_receiver.py     typed message stream, handshake, JPEG decode
    │
    ▼
image_processing.py  rotate → mirror → color_filter → head_lock → background
    │                → blur → crop_shape → fit_output → shape_mask
    │                        (AI stages call into ai_processing.py)
    ├──────────────► receiver_gui.py   preview window, optional
    ▼
virtual_camera.py    pyvirtualcam → real OS camera device
    ▼
Zoom / Meet / Teams / OBS / browsers
```

`stream_pipeline.py` wires it together and owns the threads. `protocol.py` is the
wire format, shared with the Android `FrameStreamer`.

## Handshake

```
phone → HELLO       every setting as JSON, including the Use PC flag
phone → BACKGROUND  the selected background image, when Use PC is on
PC    → ACK         models loaded, output size negotiated
phone → FRAME ...   frames, only after the ACK
```

The phone starts sending anyway after a 4 second timeout, so an older desktop
build still works. Bare length-prefixed JPEGs from an older APK are also still
accepted — the receiver sniffs the first byte to tell the two formats apart.

When Use PC is off, the desktop applies geometry only; the colour and AI stages
pass through so effects are never applied twice.

## Install

```
pip install -r requirements.txt
```

## Driver, per OS

The OS is detected at runtime. You still need the loopback driver once.

**Windows** — install [OBS Studio](https://obsproject.com) and launch it once to
register the OBS Virtual Camera driver. OBS does not need to stay open.

**macOS** — install OBS Studio 26.1+, open it once, click **Start Virtual
Camera**, then allow the camera extension in System Settings → Privacy & Security.

**Linux**
```
sudo apt install v4l2loopback-dkms v4l-utils linux-headers-$(uname -r)
sudo modprobe v4l2loopback devices=1 video_nr=10 \
     card_label='Mob Cam' exclusive_caps=1
```
`exclusive_caps=1` is required or Chrome and Firefox will not list the device.
Persist it via `/etc/modules-load.d/` and `/etc/modprobe.d/`.

## AI models

Downloaded on first use into `~/.mobcam/models`, or set `MOBCAM_MODEL_DIR`. A
`models/` folder next to the scripts is checked first, so an existing
`selfie_segmenter_landscape.tflite` can be dropped there and will be used
as-is.

| Purpose | File |
| --- | --- |
| Segmentation, best quality | `selfie_multiclass_256x256.tflite` |
| Segmentation, fastest | `selfie_segmenter_landscape.tflite` |
| Face tracking | `blaze_face_short_range.tflite` |

Pick the segmentation model in the GUI's Phone panel. If mediapipe or a model is
missing, face tracking falls back to an OpenCV Haar cascade and segmentation
switches off — frames keep flowing either way, and the reason shows in the AI
status line.

Desktop-only quality gains over the phone pipeline: full-resolution compositing,
temporal mask smoothing to stop edge flicker, guided-filter edge refinement
(install `opencv-contrib-python` to enable it, otherwise a bilateral filter is
used), a real separable background blur, and synchronous face tracking with no
one-frame lag.

## Output resolution

A camera device advertises one resolution for as long as it is open, so the
format is fixed at open time and every frame is letterboxed to fit. Changing the
resolution in the GUI reopens the device — apps already consuming the feed need
to reselect the camera. Odd pixel sizes are rounded down to even, since some
drivers' YUV conversion breaks on odd dimensions.

## Adding a processing stage

```python
def blur_background(frame, config):
    ...
    return frame

stream.pipeline.insert_before("fit_output", "blur_bg", blur_background)
```

Stages that raise are logged and skipped. Per-frame scratch shared between
stages goes in `config.frame_state`, which is cleared each frame — that is how
`background` and `blur` share one segmentation pass.