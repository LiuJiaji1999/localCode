# conda activate base
# cd knowledge/imageProcess
# python webtomp4.py
import subprocess
from pathlib import Path

def webm_to_mp4(input_path, output_path=None):
    input_path = Path(input_path)

    if output_path is None:
        output_path = input_path.with_suffix(".mp4")
    else:
        output_path = Path(output_path)

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        str(output_path)
    ]

    subprocess.run(cmd, check=True)
    print(f"转换完成：{output_path}")

if __name__ == "__main__":
    webm_to_mp4("录屏 2026-06-26 19-50-26.webm")
