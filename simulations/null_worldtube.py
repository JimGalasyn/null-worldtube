#!/usr/bin/env python3
"""Backward compatibility wrapper. Use: python -m nwt"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from nwt import main

if __name__ == '__main__':
    main()
