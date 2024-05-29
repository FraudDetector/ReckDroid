# coding:utf-8
import os

utg_dir = '../ExploreRecket/output/utgs/'
traffic_dir = '../ExploreRecket/output/traffic/'
ad_pkg_dir = 'resources/ad_packages.txt'
ad_method_dir = 'resources/ad_methods.txt'


# Recognize ad traffic from red packet traffic by the ad-domain list
def recog_ad_traffic(domains):
    is_ad = False
    # Obtain ad-domain list
    ad_domain_list = 'resources/ad-domains.txt'
    with open(ad_domain_list, "r+") as f:
        ad_list = f.read()
    if len(domains) > 0:
        # ad_domains = list()
        for domain in domains:
            if domain in ad_list:
                # Extract all ad domains from red packet traffic
                # ad_domains.append(domain)
                is_ad = True
                print('Ad traffic domain:', domain)
                break
        # if is_ad:
        #     print("Find %d ad domains:" % (len(ad_domains)), ad_domains)
    return is_ad


# Recognize ad library from method call traces of the UI state containing red packet
def recog_ad_lib(pkgs):
    is_ad_lib = False
    if len(pkgs) > 0:
        for pkg in pkgs:
            if match_ad_pkgs(pkg):
                is_ad_lib = True
                print('Ad library:', pkg)
                break
    return is_ad_lib


# Recognize ad load method (package + method) from method call traces of the UI state containing red packet
def recog_ad_load_method(pkgs_methods):
    is_ad = False
    if len(pkgs_methods) > 0:
        for pkg_method in pkgs_methods:
            if match_ad_pkgs(pkg_method) and match_ad_methods(pkg_method):
                is_ad = True
                print('Ad call stack:', pkg_method)
                break
    return is_ad


def match_ad_pkgs(pkg_method):
    is_ad_pkg = False
    with open(ad_pkg_dir, 'r+') as f:
        ad_pkgs = f.read().split('\n')
    for ad_pkg in ad_pkgs:
        if ad_pkg in pkg_method:
            is_ad_pkg = True
            # print('Ad Package:', ad_pkg)
            break
    return is_ad_pkg


def match_ad_methods(pkg_method):
    is_ad_method = False
    with open(ad_method_dir, 'r+') as f:
        ad_methods = f.read().split('\n')
    for ad_mtd in ad_methods:
        if ad_mtd.lower() in pkg_method.lower():
            is_ad_method = True
            # print('Ad Method:', ad_mtd)
            break
    return is_ad_method
