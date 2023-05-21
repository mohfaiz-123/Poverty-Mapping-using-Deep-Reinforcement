import time
import os
import os.path
from osgeo import gdal, ogr, osr
from scipy import ndimage, misc
#import cStringIO
from io import StringIO
import io
gdal.UseExceptions()
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sns
import urllib
import pandas as pd
import numpy as np

# map
import folium
from ipyleaflet import Map, basemaps, basemap_to_tiles
import geopandas
from geopandas import GeoDataFrame
from shapely.geometry import LineString, Point
import contextily

def get_shp_extent(shp_file):
    """
    Function
    --------
    get_shp_extent
    Given a shapefile, get the extent (boundaries)
    Parameters
    ----------
    shp_file : string
        Path to the shapefile
    Returns
    -------
    extent : tuple
        Boundary location of the shapefile (x_min, x_max, y_min, y_max)
    """
    inDriver = ogr.GetDriverByName("ESRI Shapefile")
    inDataSource = inDriver.Open(shp_file, 0)
    if inDataSource is None:
        print(f'Could not open {shp_file}')
    else:
      print('Opened %s' % (shp_file))
      inLayer = inDataSource.GetLayer()
      extent = inLayer.GetExtent()
      # x_min_shp, x_max_shp, y_min_shp, y_max_shp = extent
      return extent
def read_raster(raster_file):
    """
    Function
    --------
    read_raster

    Given a raster file, get the pixel size, pixel location, and pixel value

    Parameters
    ----------
    raster_file : string
        Path to the raster file

    Returns
    -------
    x_size : float
        Pixel size
    top_left_x_coords : numpy.ndarray  shape: (number of columns,)
        Longitude of the top-left point in each pixel
    top_left_y_coords : numpy.ndarray  shape: (number of rows,)
        Latitude of the top-left point in each pixel
    centroid_x_coords : numpy.ndarray  shape: (number of columns,)
        Longitude of the centroid in each pixel
    centroid_y_coords : numpy.ndarray  shape: (number of rows,)
        Latitude of the centroid in each pixel
    bands_data : numpy.ndarray  shape: (number of rows, number of columns, 1)
        Pixel value
    """
    raster_dataset = gdal.Open(raster_file, gdal.GA_ReadOnly)
    # get project coordination
    proj = raster_dataset.GetProjectionRef()
    bands_data = []
    # Loop through all raster bands
    for b in range(1, raster_dataset.RasterCount + 1):
        band = raster_dataset.GetRasterBand(b)
        bands_data.append(band.ReadAsArray())
        no_data_value = band.GetNoDataValue()
    bands_data = np.dstack(bands_data)
    rows, cols, n_bands = bands_data.shape

    # Get the metadata of the raster
    geo_transform = raster_dataset.GetGeoTransform()
    (upper_left_x, x_size, x_rotation, upper_left_y, y_rotation, y_size) = geo_transform
    
    # Get location of each pixel
    x_size = 1.0 / int(round(1 / float(x_size)))
    y_size = - x_size
    y_index = np.arange(bands_data.shape[0])
    x_index = np.arange(bands_data.shape[1])
    top_left_x_coords = upper_left_x + x_index * x_size
    top_left_y_coords = upper_left_y + y_index * y_size
    # Add half of the cell size to get the centroid of the cell
    centroid_x_coords = top_left_x_coords + (x_size / 2)
    centroid_y_coords = top_left_y_coords + (y_size / 2)

    return (x_size, top_left_x_coords, top_left_y_coords, centroid_x_coords, centroid_y_coords, bands_data)


# Helper function to get the pixel index of the point
def get_cell_idx(lon, lat, top_left_x_coords, top_left_y_coords):
    """
    Function
    --------
    get_cell_idx

    Given a point location and all the pixel locations of the raster file,
    get the column and row index of the point in the raster

    Parameters
    ----------
    lon : float
        Longitude of the point
    lat : float
        Latitude of the point
    top_left_x_coords : numpy.ndarray  shape: (number of columns,)
        Longitude of the top-left point in each pixel
    top_left_y_coords : numpy.ndarray  shape: (number of rows,)
        Latitude of the top-left point in each pixel
    
    Returns
    -------
    lon_idx : int
        Column index
    lat_idx : int
        Row index
    """
    lon_idx = np.where(top_left_x_coords < lon)[0][-1]
    lat_idx = np.where(top_left_y_coords > lat)[0][-1]
    return lon_idx, lat_idx


path='/Users/rohitsharma/Desktop/poverty/India_Shape'
inShapefile = path+'/india_st.shp'
x_min_shp, x_max_shp, y_min_shp, y_max_shp = get_shp_extent(inShapefile)

print(x_min_shp, x_max_shp, y_min_shp, y_max_shp)

left_idx, top_idx = get_cell_idx(x_min_shp, y_max_shp, top_left_x_coords, top_left_y_coords)
right_idx, bottom_idx = get_cell_idx(x_max_shp, y_min_shp, top_left_x_coords, top_left_y_coords)

print(left_idx, top_idx, right_idx, bottom_idx)

from retrying import retry
import urllib.request
from io import BytesIO
from PIL import Image

@retry(wait_exponential_multiplier=1000, wait_exponential_max=3600000)
def save_img(url, file_path, file_name):
    """
    Function
    --------
    save_img
    Given a url of the map, save the image
    Parameters
    ----------
    url : string
        URL of the map from Google Map Static API
    file_path : string
        Folder name of the map
    file_name : string
        File name
    Returns
    -------
    None
    """
    a = urllib.request.urlopen(url).read()
    b = BytesIO(a)
    #image = ndimage.imageio.imread(b, mode='RGB')
    image = np.asarray(Image.open(b).convert("RGB"))
    # when no image exists, api will return an image with the same color.
    # and in the center of the image, it said'Sorry. We have no imagery here'.
    # we should drop these images if large area of the image has the same color.
    if np.array_equal(image[:,:10,:],image[:,10:20,:]):
        pass
    else:
        plt.imsave(file_path + file_name, image[50:450, :, :])
        
# print-out help / instructions
def help():
    print('\nusage:')
    print('use --keyid= for setting the Google Maps Static API key number.')
    print('use --topidx= for setting the top row to start looping over the country.')
    print('(note: the top-left index should not be altered.)\n')

key = 'API key'
start = time.time()
m = 1

for i in range(left_idx, right_idx + 1):
    for j in range(top_idx, bottom_idx + 1):
        lon = centroid_x_coords[i]
        lat = centroid_y_coords[j]
        url = 'https://maps.googleapis.com/maps/api/staticmap?center=' + str(lat) + ',' + \
               str(lon) + '&zoom=16&size=400x500&maptype=satellite&key=' + key
        lightness = bands_data[j, i, 0]
        file_path = path+'/google_images/' + str(lightness) + '/'
        if not os.path.isdir(file_path):
            os.makedirs(file_path)
        file_name = str(i) + '_' + str(j) +'.jpg'
        save_img(url, file_path, file_name)
        if m % 100 == 0:
            end = time.time()
            print(m)
            print('It takes  ' + str((end-start)/60) +' min ')
        m += 1


