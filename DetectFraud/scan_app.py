import requests
import hashlib
import os
import json

VIRUSTOTAL_API_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' # Set your VirusTotal API key


# Scan files by VirusTotal
def upload_file(apk_path):
    url = "https://www.virustotal.com/api/v3/files"
    apk_size = get_file_size(apk_path)
    if apk_size > 32:
        url = get_upload_url()    # URL for uploading files larger than 32MB
    # print(url)
    files = {"file": open(apk_path, "rb")}
    headers = {
        "Accept": "application/json",
        "x-apikey": VIRUSTOTAL_API_KEY
    }
    try:
        response = requests.post(url, files=files, headers=headers)
        # print(response.text)
    except Exception as e:
        print('Failed to upload the app to VirusTotal!')
        print(e)


# Get the size of the file
def get_file_size(file_path):
    fsize = os.path.getsize(file_path)
    fsize = fsize / float(1024 * 1024)
    return fsize


# Get a URL for uploading files larger than 32MB
def get_upload_url():
    url = "https://www.virustotal.com/api/v3/files/upload_url"
    headers = {
        "Accept": "application/json",
        "x-apikey": VIRUSTOTAL_API_KEY
    }
    response = requests.get(url, headers=headers)
    upload_url = json.loads(response.text)['data']
    return upload_url


# Calculate SHA-256
def get_sha256(apk_path, block_size=2 ** 8):
    sha256 = hashlib.sha256()
    f = open(apk_path, 'rb')
    while True:
        data = f.read(block_size)
        if not data:
            break
        sha256.update(data)
    # print(sha256.hexdigest())
    return sha256.hexdigest()


def get_file_report(apk_path):
    apk_sha256 = get_sha256(apk_path)
    request_url = "https://www.virustotal.com/api/v3/files/" + apk_sha256
    headers = {
        "Accept": "application/json",
        "x-apikey": VIRUSTOTAL_API_KEY
    }
    response = requests.request("GET", request_url, headers=headers)
    # print(json.loads(response.text)['data']['attributes']['last_analysis_stats'])
    return response
