#!/usr/bin/env python3
"""
Can I join
"""


import requests


def availableShips(passengerCount):
    """
    Returns the number of ships that can accommodate the given number of passengers.
    """
    res = requests.get('https://swapi-api.alx-tools.com/api/starships')
    ships = res.json().get('results')
    count = 0
    for ship in ships:
        try:
            if int(ship.get('passengers')) >= passengerCount:
                count += 1
        except ValueError:
            pass
    return count


