import os
src = r"C:\Users\28130\Desktop\lyy-6.12\doubao_final.docx"
dst = r"C:\Users\28130\Desktop\lyy-6.12\doubao.docx"
if os.path.exists(dst):
    os.remove(dst)
os.rename(src, dst)
print(f"Renamed to: {dst}")
print(f"File size: {os.path.getsize(dst)} bytes")
