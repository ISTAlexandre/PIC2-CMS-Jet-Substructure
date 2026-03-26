'''
python3 python/dpsi12_video.py
'''
import os
import re
import subprocess

BASE = "iterative_psi12"
duration_s = 10

def find_sequence_prefix(folder: str) -> str | None:
    # Find something like "name_001.png" and return "name_%03d.png"
    pngs = sorted(f for f in os.listdir(folder) if f.lower().endswith(".png"))
    for f in pngs:
        m = re.match(r"^(.*)_001\.png$", f)
        if m:
            return m.group(1)
    return None

for name in sorted(os.listdir(BASE)):
    folder = os.path.join(BASE, name)
    if not os.path.isdir(folder):
        continue

    prefix = find_sequence_prefix(folder)
    if prefix is None:
        print(f"Skipping {folder}: no *_001.png sequence found")
        continue

    # count frames matching that prefix
    frames = sorted(
        f for f in os.listdir(folder)
        if re.match(rf"^{re.escape(prefix)}_\d{{3}}\.png$", f)
    )
    if not frames:
        print(f"Skipping {folder}: no frames for prefix '{prefix}'")
        continue

    n_imgs = len(frames)
    fps = n_imgs / duration_s

    pattern = os.path.join(folder, f"{prefix}_%03d.png")
    out_mp4 = os.path.join(BASE, f"{name}.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", "1",
        "-i", pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-movflags", "+faststart",
        out_mp4,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)