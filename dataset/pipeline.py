import zipfile
import requests
from io import BytesIO

# Step 0: Download and Extract Dataset from Dropbox
dropbox_link = "https://www.dropbox.com/scl/fi/dro2nnu6fg8n78q80iic8/archive.zip?rlkey=viz52a08sswvzj0wb6xje1efr&st=na7wo7ua&dl=1"  # Replace with your actual Dropbox link
print("starting")
def download_and_extract_from_dropbox(url, extract_to='./dataset'):
    print("Downloading dataset from Dropbox...")
    url = url.replace("?dl=0", "?dl=1")  # Ensure direct download
    response = requests.get(url, stream=True)

    if response.status_code == 200:
        print("Download successful. Extracting archive...")
        with zipfile.ZipFile(BytesIO(response.content)) as archive:
            archive.extractall(extract_to)
        print(f"Extraction completed. Files are stored in '{extract_to}'.")
    else:
        print("Failed to download the dataset. Please check the link.")
        exit()

download_and_extract_from_dropbox(dropbox_link)
print("ended")