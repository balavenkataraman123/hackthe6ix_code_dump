import asyncio # imports required libraries
from websockets.server import serve
import base64
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
from configs import getipkeys, getdisplay
