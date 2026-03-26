'''
python3 python/lund_video.py
'''
import os
import subprocess

BASE = "iterative_lund"
duration_s = 10

for name in sorted(os.listdir(BASE)):
    folder = os.path.join(BASE, name)
    if not os.path.isdir(folder):
        continue
    #if empty folder, skip
    if len(os.listdir(folder)) == 0:
        continue

    n_imgs = len([f for f in os.listdir(folder) if f.endswith(".png")])
    fps = n_imgs / duration_s  # 15

    pattern = os.path.join(folder, "lund_%03d.png")
    out_mp4 = os.path.join(BASE, f"{name}.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-start_number", "1",
        "-i", pattern,
        # force even dimensions for x264
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", "30",
        "-movflags", "+faststart",
        out_mp4,
    ]
    #print(" ".join(cmd))
    subprocess.run(cmd, check=True)