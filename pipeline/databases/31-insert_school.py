#!/usr/bin/env python3
"""
Insert document
"""


def insert_school(mongo_collection, **kwargs):
    """ Insert document in a collection based on kwargs

    Args:
        mongo_collection (_type_): _description_
    Returns: new _id
    """
    new_id = mongo_collection.insert_one(kwargs).inserted_id
    return new_id
