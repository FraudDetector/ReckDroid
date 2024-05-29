import os
import datetime
import json
import urllib

from DetectFraud.recognize_ad import recog_ad_traffic, recog_ad_lib
from DetectFraud.scan_app import upload_file, get_file_report
from DetectFraud.scan_url import get_url_report

download_dir = '../ExploreRecket/output/downloads/'
utg_dir = '../ExploreRecket/output/utgs/'
traffic_dir = '../ExploreRecket/output/traffic/'


# Obtain the number of red packets in the app
def get_red_packet_num(apk_name):
    recket_view_dir = utg_dir + apk_name + '/candidates/red_packet/'
    recket_views = os.listdir(recket_view_dir)
    return len(recket_views)


# Fraud 1: Aggressive Advertising
def detect_ad_fraud(domains, pkgs):
    is_ad_fraud = False
    if recog_ad_traffic(domains) and recog_ad_lib(pkgs):
        is_ad_fraud = True
    return is_ad_fraud


# Fraud 2-1: Malicious App (automatically downloaded in the background)
def detect_auto_app(urls, pkg_name):
    is_app_fraud = False
    if len(urls) > 0:
        for url in urls:
            if url[-4:] == '.apk' or url[-5:] == '.xapk':
                print('app download link (automatically):', url)
                # Download the app and upload it to VirusTotal
                is_app_fraud = detect_app(url, pkg_name)
                break
    return is_app_fraud


# Fraud 2-2: Malicious App (recommended in the landing page)
def detect_recom_app(redirects, pkg_name):
    is_app_fraud = False
    # Look for the download link for the app
    for redirect in redirects:
        num = redirect.split('#')[0]
        response_path = traffic_dir + pkg_name + '/' + num + '/ResponseBody.txt'
        with open(response_path, "r+", encoding='UTF-8', errors="ignore") as f:
            response_body = f.read()
        apk_link = ""
        link_index = response_body.find('\"link\"')
        while link_index != -1:
            start_index = response_body.index('\"', link_index + 6)
            end_index = response_body.index('\"', start_index + 1)
            link = response_body[start_index + 1: end_index]
            if 'http' in link:
                apk_link = link
                break
            link_index = response_body.find('\"link\"', end_index)
        if apk_link == "":
            url_index = response_body.find('\"url')
            while url_index != -1:
                start_index = response_body.index('\"', url_index + 6)
                end_index = response_body.index('\"', start_index + 1)
                link = response_body[start_index + 1: end_index]
                if "http" in link:
                    apk_link = link
                    break
                url_index = response_body.find('\"url', end_index)

        if apk_link != "":
            # Download the app and upload it to VirusTotal
            print("app download link (recommending):", apk_link)
            is_app_fraud = detect_app(apk_link, pkg_name)
            break
    return is_app_fraud


# Download and upload an app
def detect_app(url, pkg_name):
    is_fraud = False
    # Download the app
    if not os.path.exists(download_dir + pkg_name):
        os.makedirs(download_dir + pkg_name)
    try:
        # filename = url.split('/')[-1]
        # filepath = download_dir + pkg_name + '/' + filename
        filename = datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S') + '.apk'
        filepath = download_dir + pkg_name + '/' + filename
        urllib.request.urlretrieve(url, filename=filepath)
    except Exception as e:
        print('Error occurred when downloading file!')
        print(e)

    # Upload the app to VirusTotal
    upload_file(filepath)
    response = get_file_report(filepath)
    if response.status_code == 200:
        analysis_results = json.loads(response.text)['data']['attributes']['last_analysis_stats']
        # print(analysis_results)
        if analysis_results['malicious'] >= 3:
            print('Malicious app:', filepath)
            is_fraud = True
    return is_fraud


# Fraud 3: Malicious Redirect
def detect_redirect_fraud(redirects):
    is_redirect_fraud = False
    # Remove duplicate redirects
    redirects_removal = set()
    for redirect in redirects:
        num = redirect.split('#')[1]
        redirects_removal.add(num)

    # Detect malicious redirects by VirusTotal
    for link in redirects_removal:
        # print('redirect:', link)
        response = get_url_report(link)
        if response and response.status_code == 200:
            analysis_results = json.loads(response.text)['data']['attributes']['last_analysis_stats']
            # print(analysis_results)
            if analysis_results['malicious'] >= 3:
                print('Malicious redirect:', link)
                is_redirect_fraud = True
                break
    return is_redirect_fraud


# Extract all domains from red packet traffic
def extract_recket_domain(pkg_name):
    domain_path = traffic_dir + pkg_name + '/hostnames.txt'
    with open(domain_path, "r+") as f:
        hostnames = f.read().split('\n')
    # Extract domains from each set of red packet traffic
    red_packet_num = get_red_packet_num(pkg_name)
    recket_domains = {}
    start = -1
    end = -1
    for i in range(red_packet_num):
        domains = set()
        try:
            start = hostnames.index('localhost', end + 1)
        except ValueError:
            start = -1
        if start != -1:
            try:
                end = hostnames.index('localhost', start + 1)
            except ValueError:
                end = -1
            if end != -1:
                domains = set(hostnames[start + 1: end])
        recket_domains[i] = domains
    return recket_domains


# Extract packages and methods from method call traces of the UI states containing red packets
def get_pkgs_methods(pkg_name):
    trace_dir = utg_dir + pkg_name + '/events/trace/'
    trace_files = os.listdir(trace_dir)
    recket_pkgs_methods = {}
    red_packet_num = get_red_packet_num(pkg_name)
    for i in range(red_packet_num):
        recket_pkgs_methods[i] = set()

    recket_num = 0
    for file_name in trace_files:
        if 'red' in file_name:
            file_dir = trace_dir + file_name
            with open(file_dir, "r+", encoding='utf-8', errors="ignore") as f:
                content = f.read()
            pkgs_methods = set()
            s_index = content.find('*methods')
            if s_index != -1:
                e_index = content.find('*end')
                method_calls = content[s_index + 9: e_index - 1].split('\n')
                for call in method_calls:
                    temp = call.split('\t')
                    pkg_method = temp[1] + '/' + temp[4] + '/' + temp[2]
                    pkgs_methods.add(pkg_method)
            recket_pkgs_methods[recket_num] = pkgs_methods
            recket_num += 1
    return recket_pkgs_methods


# Extract packages from the method call traces of the UI states containing red packets
def get_packages(pkg_name):
    trace_dir = utg_dir + pkg_name + '/events/trace/'
    trace_files = os.listdir(trace_dir)
    recket_pkgs = {}
    red_packet_num = get_red_packet_num(pkg_name)
    for i in range(red_packet_num):
        recket_pkgs[i] = set()

    recket_num = 0
    for file_name in trace_files:
        if 'red' in file_name:
            file_dir = trace_dir + file_name
            with open(file_dir, "r+", encoding='utf-8', errors="ignore") as f:
                content = f.read()
            pkgs = set()
            s_index = content.find('*methods')
            if s_index != -1:
                e_index = content.find('*end')
                method_calls = content[s_index + 9: e_index - 1].split('\n')
                for call in method_calls:
                    pkg = call.split('\t')[1]
                    pkgs.add(pkg)
            recket_pkgs[recket_num] = pkgs
            recket_num += 1
    return recket_pkgs


# Extract all urls from red packet traffic
def extract_recket_url(pkg_name):
    url_path = traffic_dir + pkg_name + '/urls.txt'
    with open(url_path, "r+") as f:
        urls = f.read().split('\n')
    # Extract urls from each set of red packet traffic
    red_packet_num = get_red_packet_num(pkg_name)
    recket_urls = {}
    start = -1
    end = -1
    for i in range(red_packet_num):
        temp_urls = set()
        try:
            start = urls.index('http://localhost:8080/?' + pkg_name + '&start', end + 1)
        except ValueError:
            start = -1
        if start != -1:
            try:
                end = urls.index('http://localhost:8080/?' + pkg_name + '&end', start + 1)
            except ValueError:
                end = -1
            if end != -1:
                temp_urls = set(urls[start + 1: end])
        recket_urls[i] = temp_urls
    return recket_urls


# Extract all redirects from red packet traffic
def extract_redirect(pkg_name):
    redirect_path = traffic_dir + pkg_name + '/redirects.txt'
    with open(redirect_path, "r+") as f:
        redirects = f.read().split('\n')
    # Extract redirects from each set of red packet traffic
    red_packet_num = get_red_packet_num(pkg_name)
    start = -1
    end = -1
    recket_redirects = {}
    for i in range(red_packet_num):
        temp_redirects = []
        try:
            start = redirects.index('RedPacket', end + 1)
        except ValueError:
            start = -1
        if start != -1:
            try:
                end = redirects.index('RedPacket', start + 1)
            except ValueError:
                end = -1
            if end != -1:
                temp_redirects = redirects[start + 1: end]
        recket_redirects[i] = temp_redirects
    return recket_redirects


def main():
    recket_app_file = utg_dir + 'red_packet_apps.txt'
    with open(recket_app_file, 'r+') as fp:
        recket_apps = fp.read().split('\n')
    # print(recket_apps)
    for apk_name in recket_apps:
        print("############### %s ###############" % apk_name)
        recket_num = get_red_packet_num(apk_name)
        domains = extract_recket_domain(apk_name)
        pkgs = get_packages(apk_name)
        urls = extract_recket_url(apk_name)
        redirects = extract_redirect(apk_name)
        for i in range(recket_num):
            time1 = datetime.datetime.now()
            b1 = detect_ad_fraud(domains[i], pkgs[i])
            time2 = datetime.datetime.now()
            time_cost1 = (time2 - time1).seconds + (time2 - time1).microseconds / 1000000

            b2 = detect_auto_app(urls[i], apk_name) or detect_recom_app(redirects[i], apk_name)
            time3 = datetime.datetime.now()
            time_cost2 = (time3 - time2).seconds + (time3 - time2).microseconds / 1000000

            b3 = detect_redirect_fraud(redirects[i])
            time4 = datetime.datetime.now()
            time_cost3 = (time4 - time3).seconds + (time4 - time3).microseconds / 1000000

            print('red packet%d fraud:' % i, b1, time_cost1, b2, time_cost2, b3, time_cost3)


if __name__ == "__main__":
    main()
