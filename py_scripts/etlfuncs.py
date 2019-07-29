import pandas as pd
import numpy as np
import os
from dateutil.parser import parser
from sqlalchemy import create_engine


def test_categories(text, test_cols):
    """
    Test if the encoded categories haven't changed

    Args:
    text (str): string with categories and values encoded
    test_cols (list): list of category names to check against

    Output:
    (bool): True if the categories and their order haven't
            changed, False otherwise
    """
    t2 = test_cols
    t1 = [x[:-2] for x in text.split(';')]
    if t1 == t2:
        return True
    else:
        return False


def get_cat_values(text):
    """
    This basic function takes an encoded category string
    and returns a list of 0s and 1s indicating
    categories. It could be written as a lambda,
    but it would be less readable.

    Args:
    text (str): text with values encoded in them

    Output:
    val_list (list): a list of 0s and 1s flags
    """

    vals = [int(x[-1]) for x in text.split(';')]
    return vals