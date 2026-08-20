#!/usr/bin/env python3
"""Download user uploaded images."""
import urllib.request
import os

images = [
    ("https://aka.doubaocdn.com/s/sqiV1wfgEJ", "fig1.png"),
    ("https://aka.doubaocdn.com/s/1P2t1wfgEJ", "fig2.png"),
    ("https://aka.doubaocdn.com/s/1pP41wfgEJ", "fig3.png"),
    ("https://aka.doubaocdn.com/s/fBLT1wfgEJ", "fig4.png"),
    ("https://aka.doubaocdn.com/s/Q3Bb1wfgEJ", "fig5.png"),
]

save_dir = r"C:\Users\28130\Desktop\lyy-6.12\images"
os.makedirs(save_dir, exist_ok=True)

for url, filename in images:
    filepath = os.path.join(save_dir, filename)
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        size = os.path.getsize(filepath)
        print(f"  Done: {size} bytes")
    except Exception as e:
        print(f"  Error: {e}")

print("All downloads completed.")
