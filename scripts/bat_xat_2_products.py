from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import math
import shutil

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import planetary_computer
import rasterio
from pyproj import CRS, Transformer
from pystac_client import Client
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import Polygon, mapping
from shapely.ops import transform