"""Record the dashboard demo GIF with Playwright (uses the installed Microsoft Edge).

Start the API first:   uvicorn anomaly.serve.api:app --port 8000
Then:                  pip install -e ".[demo]"; python scripts/record_demo.py

Story (about 12-15 s): a defective bottle with heatmap and score, a good bottle
showing OK, then a defect on a textured surface. Frames are page screenshots at
about 10 fps, assembled into docs/demo.gif with Pillow; also saves
docs/screenshot_defect.png. The images shown are from MVTec AD (CC BY-NC-SA 4.0).
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

from PIL import Image

from anomaly.config import PROJECT_ROOT

DATA = PROJECT_ROOT / "data" / "mvtec_ad"
DOCS = PROJECT_ROOT / "docs"
FPS = 10
MAX_GIF_BYTES = 5 * 1024 * 1024

# (caption, category, defect type, file stem, seconds to hold the result)
STORY = (
    ("1 / 3  Defective bottle", "bottle", "broken_large", "000", 2.8),
    ("2 / 3  Good bottle", "bottle", "good", "000", 2.6),
    ("3 / 3  Textured surface: carpet", "carpet", "hole", "000", 3.2),
)

CAPTION_JS = """(text) => {
  let el = document.getElementById('demo-caption');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-caption';
    el.style.cssText = 'position:fixed;top:12px;right:16px;z-index:9;padding:6px 12px;'
      + 'border-radius:6px;background:#1b1f24;color:#fff;font:600 15px system-ui';
    document.body.appendChild(el);
  }
  el.textContent = text;
}"""


CURSOR_JS = """([x, y, ms]) => {
  let el = document.getElementById('demo-cursor');
  if (!el) {
    el = document.createElement('div');
    el.id = 'demo-cursor';
    el.style.cssText = 'position:fixed;z-index:10;width:22px;height:22px;margin:-11px 0 0 -11px;'
      + 'border-radius:50%;background:rgba(47,111,222,.35);border:3px solid #2f6fde;'
      + 'pointer-events:none;left:450px;top:420px;transition:left ease-in-out,top ease-in-out';
    document.body.appendChild(el);
  }
  el.style.transitionDuration = ms + 'ms';
  el.style.left = x + 'px';
  el.style.top = y + 'px';
}"""


def capture(page, seconds: float, frames: list[tuple[Image.Image, float]]) -> None:
    """Append page screenshots with timestamps for `seconds` at about FPS frames per second."""
    end = time.perf_counter() + seconds
    while (now := time.perf_counter()) < end:
        png = page.screenshot()
        frames.append((Image.open(io.BytesIO(png)).convert("RGB"), now))
        time.sleep(max(0.0, 1 / FPS - (time.perf_counter() - now)))


def to_gif(frames: list[tuple[Image.Image, float]], path: Path, width: int) -> int:
    """Write an optimised GIF: one shared palette, identical consecutive frames merged."""
    times = [t for _, t in frames] + [frames[-1][1] + 1 / FPS]
    images, durations = [], []
    for (img, _), t0, t1 in zip(frames, times, times[1:], strict=False):
        if img.width != width:
            img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
        ms = max(20, round((t1 - t0) * 1000 / 10) * 10)
        if images and img.tobytes() == images[-1].tobytes():
            durations[-1] += ms
        else:
            images.append(img)
            durations.append(ms)
    # One palette for all frames, built from frames spread over the whole story so the
    # heatmap colours of every scene are represented.
    keys = [images[round(k * (len(images) - 1) / 5)] for k in range(1, 6)]
    mosaic = Image.new("RGB", (width, sum(k.height for k in keys)))
    y = 0
    for k in keys:
        mosaic.paste(k, (0, y))
        y += k.height
    palette = mosaic.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    paletted = [im.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG) for im in images]
    paletted[0].save(
        path,
        save_all=True,
        append_images=paletted[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    return path.stat().st_size


def main() -> None:
    """Drive the dashboard through the demo story and save the GIF and a screenshot."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--width", type=int, default=900)
    parser.add_argument("--height", type=int, default=860)
    args = parser.parse_args()
    from playwright.sync_api import sync_playwright

    DOCS.mkdir(exist_ok=True)
    frames: list[tuple[Image.Image, float]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": args.width, "height": args.height})
        page.goto(args.url)
        page.wait_for_selector("#cat option", state="attached")
        for _, category, defect, stem, _ in STORY:  # warm-up: show steady-state latency
            image = DATA / category / "test" / defect / f"{stem}.png"
            if not image.is_file():
                raise SystemExit(f"missing demo image {image}")
            page.request.post(
                f"{args.url}/api/predict",
                multipart={
                    "category": category,
                    "file": {
                        "name": image.name,
                        "mimeType": "image/png",
                        "buffer": image.read_bytes(),
                    },
                },
            )
        for i, (caption, category, defect, stem, seconds) in enumerate(STORY):
            page.select_option("#cat", category)
            page.evaluate(CAPTION_JS, caption)
            button = page.locator(
                f'#samples button[data-url="/api/samples/{category}-{defect}-{stem}"]'
            )
            box = button.bounding_box()
            if box is None:
                raise SystemExit(f"sample {category}-{defect}-{stem} not shown in the dashboard")
            capture(page, 0.4, frames)
            page.evaluate(
                CURSOR_JS, [box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, 800]
            )
            capture(page, 1.0, frames)
            with page.expect_response("**/api/predict"):
                button.click()
            page.wait_for_function(
                "() => document.getElementById('result').offsetHeight > 0 && "
                "document.getElementById('img').src.length > 100"
            )
            page.wait_for_timeout(150)
            capture(page, seconds, frames)
            if i == 0:
                page.screenshot(path=str(DOCS / "screenshot_defect.png"), full_page=True)
        browser.close()

    size = to_gif(frames, DOCS / "demo.gif", args.width)
    total = frames[-1][1] - frames[0][1]
    print(f"docs/demo.gif: {len(frames)} frames, {total:.1f} s, {size / 1e6:.2f} MB")
    if size > MAX_GIF_BYTES:
        raise SystemExit("GIF is larger than 5 MB; lower --width or shorten the story")


if __name__ == "__main__":
    main()
