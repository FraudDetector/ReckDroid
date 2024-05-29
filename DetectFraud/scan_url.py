import requests, base64

VIRUSTOTAL_API_KEY = 'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx' # Set your VirusTotal API key


# Scan urls by VirusTotal
def upload_url(url):
    request_url = "https://www.virustotal.com/api/v3/urls"
    payload = "url=" + url
    headers = {
        "Accept": "application/json",
        "x-apikey": VIRUSTOTAL_API_KEY,
        "Content-Type": "application/x-www-form-urlencoded"
    }
    response = requests.request("POST", request_url, data=payload, headers=headers)
    # print(response.text)
    return response.status_code


def get_url_report(url):
    response = None
    status = upload_url(url)
    # print('scan_url:', status)
    if status == 200:
        url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
        request_url = "https://www.virustotal.com/api/v3/urls/" + url_id
        headers = {
            "Accept": "application/json",
            "x-apikey": VIRUSTOTAL_API_KEY
        }
        response = requests.request("GET", request_url, headers=headers)
    return response
