## Backend IdCardScanner

This is a Flask application that used to be the backend of an OCR app (with Ionic) meant to turn an CEDEAO ID card into an array of data about the person concerned.

I can't remember the useful files so I push everything including the text images and the text files that I wanted to store.

In the app.py , the functions are a little bit commented.

Don't forget to install the packages required.

from flask import Flask, request, jsonify


_________________________

import numpy as np
import matplotlib.pyplot as plt
import io as io
from PIL import Image
import base64
import re
import json
from flask_cors import CORS

import os

from cv2 import cv2

import pytesseract
from pytesseract import Output

_________________________

#### Alassane Dial DIOP