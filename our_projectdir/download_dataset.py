import requests
from pathlib import Path

url = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"

output = Path("data/real/download/Task01_BrainTumour.tar")
output.parent.mkdir(parents=True, exist_ok=True)

print("Starting download...")
print("Destination:", output)

response = requests.get(url, stream=True, timeout=60)
response.raise_for_status()

total = int(response.headers.get("content-length", 0))
downloaded = 0

with open(output, "wb") as f:
    for chunk in response.iter_content(chunk_size=1024 * 1024):
        if chunk:
            f.write(chunk)
            downloaded += len(chunk)

            if total:
                percent = downloaded / total * 100
                print(f"\rDownloaded: {percent:.1f}%", end="")

print("\nDownload complete!")
print("File:", output)
print("Size (GB):", round(output.stat().st_size / (1024**3), 2))