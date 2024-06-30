#!/usr/bin/env python3


""" Return list of ships"""

import requests
import sys
import time


if __name__ == "__main__":
    res = requests.get(sys.argv[1])

    if res.status_code == 403:
        rate_limit = int(res.headers.get('X-Ratelimit-Reset'))
        current_time = int(time.time())
        diff = (rate_limit - current_time) // 60
        print("Reset in {} min".format(diff))
        # get remaining rate

    elif res.status_code == 404:
        print("Not found")
    elif res.status_code == 200:
        res = res.json()
        print(res['location'])
