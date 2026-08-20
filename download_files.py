#!/usr/bin/env python3
"""Download user uploaded files."""
import urllib.request
import os

files = [
    ("https://aka.doubaocdn.com/s/aptC1wfflQ", "实验方法部分.docx"),
    ("https://aka.doubaocdn.com/s/0fUo1wfflQ", "paper_analysis.md"),
    ("https://aka.doubaocdn.com/s/i9kv1wfflQ", "summary.md"),
]

save_dir = r"C:\Users\28130\Desktop\lyy-6.12"

for url, filename in files:
    filepath = os.path.join(save_dir, filename)
    print(f"Downloading {filename}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        size = os.path.getsize(filepath)
        print(f"  Done: {size} bytes")
    except Exception as e:
        print(f"  Error: {e}")

print("All downloads completed.")
