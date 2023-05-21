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
#%matplotlib inline
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

path = '/Users/rohitsharma/Desktop/' 
df_dhs = pd.read_csv(path+'poverty/data.csv', index_col=False)

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

path = '/Users/rohitsharma/Desktop/poverty/'               

raster_file = path+'F182013.v4/F182013.v4c_web.stable_lights.avg_vis.tif'
x_size, top_left_x_coords, top_left_y_coords, centroid_x_coords, centroid_y_coords, bands_data = read_raster(raster_file)

# save the result in compressed format
np.savez(path+'nightlight.npz', top_left_x_coords=top_left_x_coords, top_left_y_coords=top_left_y_coords, bands_data=bands_data)

print(bands_data.shape)
print('Max light intensity: ',bands_data.max())
print('Min light intensity: ',bands_data.min())
print(bands_data[1,0,0])


def get_nightlight_feature(sample):
    idx, x, y = sample
    lon_idx, lat_idx = get_cell_idx(x, y, top_left_x_coords, top_left_y_coords)
    # Select the 10 * 10 pixels
    left_idx = lon_idx - 5
    right_idx = lon_idx + 4
    up_idx = lat_idx - 5
    low_idx = lat_idx + 4
    luminosity_100 = []
    for i in range(left_idx, right_idx + 1):
        for j in range(up_idx, low_idx + 1):
            #"" Get the luminosity of this pixel
            luminosity = bands_data[j, i, 0]
            luminosity_100.append(luminosity)
    luminosity_100 = np.asarray(luminosity_100)
    max_ = np.max(luminosity_100)
    min_ = np.min(luminosity_100)
    mean_ = np.mean(luminosity_100)
    median_ = np.median(luminosity_100)
    std_ = np.std(luminosity_100)
    return pd.Series({'id': idx, 'max_light': max_, 'min_light': min_, 'mean_light': mean_, 
                      'median_light': median_, 'std_light': std_})

print(df_dhs.head())
df_cluster_light = df_dhs.apply(lambda x: 
                                get_nightlight_feature([x['Cluster'], x['longitude'], x['latitude']]), axis=1)

print(df_cluster_light.head())
df_dhs_light = df_cluster_light.merge(df_dhs, left_on = 'id', right_on='Cluster')
print(df_dhs_light.describe())

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(2, 3, sharex=False, figsize=(15,7))


## set_matplotlib_formats('retina')



sns.regplot(x="mean_light", y="Wealth Index Factor Score", data=df_dhs_light,ax=ax[0][0])
ax[0][0].set_xlabel('Average nighttime luminosity')
ax[0][0].set_ylabel('Wealth Index Factor Score (median)')

sns.regplot(x="mean_light", y="Access to water", data=df_dhs_light, ax=ax[0][1])
ax[0][1].set_xlabel('Average nighttime luminosity')
ax[0][1].set_ylabel('Access to water (minute)')

sns.regplot(x="mean_light", y="Access to electricity", data=df_dhs_light, ax=ax[0][2])
ax[0][2].set_xlabel('Average nighttime luminosity')
ax[0][2].set_ylabel('Access to electricity (count)')

sns.regplot(x="mean_light", y="Acess to cellphone", data=df_dhs_light, ax=ax[1][0])
ax[1][0].set_xlabel('Average nighttime luminosity')
ax[1][0].set_ylabel('Acess to cellphone (count)')

sns.regplot(x="mean_light", y="Education completed", data=df_dhs_light, ax=ax[1][1])
ax[1][1].set_xlabel('Average nighttime luminosity')
ax[1][1].set_ylabel('Education completed (count)')

sns.regplot(x="mean_light", y="hiv blood test result", data=df_dhs_light, ax=ax[1][2])
ax[1][2].set_xlabel('Average nighttime luminosity')
ax[1][2].set_ylabel('HIV blood test result (count)')

print(plt.tight_layout())
plt.figure(figsize=(5,5))
g = sns.scatterplot(y='latitude', x='longitude', alpha=0.8,
                size='Wealth Index Factor Score', palette='Reds', 
                hue='Wealth Index Factor Score', sizes=(10, 1000), 
                data=df_dhs_light)

g.patch.set_facecolor('black')
g.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

center_x, center_y = df_dhs_light['latitude'].median(), df_dhs_light['longitude'].median()
print(center_x, center_y)

def generateBaseMap(default_location=[24.30846413 , 79.260475015], default_zoom_start=8, width=500, height=500):
    base_map = folium.Map(location=default_location, 
                          width=width,height=height,
                          control_scale=True, zoom_start=default_zoom_start)
    return base_map

from folium import plugins
from folium.plugins import HeatMap

base_map = generateBaseMap([center_x, center_y], 8, 400, 500)
folium.plugins.HeatMap(data=df_dhs_light[['latitude', 'longitude', 'Wealth Index Factor Score']]
                       .groupby(['latitude', 'longitude']).sum().reset_index().values.tolist()
                       , radius=10, max_zoom=10).add_to(base_map)
print('Wealth Index Factor Score')
base_map

base_map = generateBaseMap([center_x, center_y], 8,400,500)
folium.plugins.HeatMap(data=df_dhs_light[['latitude', 'longitude', 'mean_light']]
                       .groupby(['latitude', 'longitude']).sum().reset_index().values.tolist()
                       , radius=10, max_zoom=10).add_to(base_map)
print('Average Night Light Luminosity')
base_map
fig = plt.figure()
ax = fig.add_subplot(111)
_ = ax.hist(df_dhs_light[df_dhs_light['mean_light'] > 0]['Wealth Index Factor Score'], bins=30, alpha=0.5)
_ = ax.hist(df_dhs_light[df_dhs_light['mean_light'] == 0]['Wealth Index Factor Score'], 
            color='red', alpha=0.5, bins=30)
plt.legend(['mean light intensity >0','mean light intensity = 0'])
plt.ylabel('Count')
plt.xlabel('Wealth Index Factor Score')
df_dhs_light_pd = geopandas.GeoDataFrame(df_dhs_light, 
                                         geometry = geopandas.points_from_xy(df_dhs_light['longitude'], df_dhs_light['latitude']))
print(df_dhs_light_pd.head())
df_dhs_light_pd.crs = {'init' :'epsg:4326'}

lgnd_kwds = {'label': 'Wealth Index Factor Score'}

ax_clusters_1 = df_dhs_light_pd.to_crs(epsg=3857).plot(column = 'Wealth Index Factor Score', 
                                                     legend=True, legend_kwds=lgnd_kwds, alpha=0.5,
                                                     figsize=(12, 10), markersize = 100, cmap='hot')

#contextily.add_basemap(ax_clusters_1, url=contextily.sources.ST_TERRAIN_BACKGROUND)
#contextily.add_basemap(ax_clusters_1, url=contextily.providers.NASAGIBS.ViirsEarthAtNight2012)
ax_clusters_1.set_axis_off()
plt.show()
#df_shape = pd.read_csv(path+'test_2_shape.csv', index_col=False)
#df_shape.head()
list_wi=[]

for index, row in df_dhs_light_pd.iterrows():
    list_d = [row['latitude'], row['longitude'],row['Wealth Index Factor Score']/6.67233*1000]
    list_wi.append(list_d)

print(df_dhs_light_pd.head())

from ipyleaflet import Map, basemaps, basemap_to_tiles, Heatmap, GeoData, LayersControl, CircleMarker, LayerGroup
from ipywidgets import Button, Layout

center = [center_x, center_y]
zoom = 8

#shapefile
#geo_data = GeoData(geo_dataframe = shape_BU,
 #                  style={'color': 'gray', 'fillColor': '#3366cc', 
  #                        'opacity':1, 'weight':1, 'dashArray':'2', 'fillOpacity':0},
   #                name = 'Countries')

def create_marker(row):
    name = row["id"]
    size = int(row["Wealth Index Factor Score"]/6.67233*20)
    lat_lon = (row['latitude'], row['longitude'])
    return CircleMarker(location=lat_lon,
                    draggable=False,
                    title=name, opacity=0.5,
                    radius=size,color = "red",fill = False,
                    weight=1)

markers = df_dhs_light_pd.apply(create_marker, axis=1)
layer_group = LayerGroup(layers=tuple(markers.values))


#m=Map(basemap=basemaps.Stamen.Terrain, center=center, zoom=zoom)
m=Map(basemap=basemaps.NASAGIBS.ViirsEarthAtNight2012, center=center, zoom=zoom,
     scroll_wheel_zoom=True,
     layout=Layout(width='400px', height='550px'))

heatmap = Heatmap(
    locations=list_wi, max_zoom=18,
    gradient={.1: 'blue', .2: 'white', .8: 'orange', .9: 'yellow', 1: 'red'},
    radius=15
)

m.add_layer(layer_group)
#m.add_layer(heatmap);
#m.add_layer(geo_data)
#m

#Map(basemap=basemaps.NASAGIBS.ViirsEarthAtNight2012, center=center, zoom=zoom)
#m

df_dhs_light[df_dhs_light['mean_light']>1]['mean_light'].hist(bins=20)#.head()
plt.xlabel('light intensity')
plt.ylabel('count')
#nightlights = pd.read_csv(nightlights_file)
#nightlights.head(3)


def get_nightlight_luminosity(sample):
    idx, x, y = sample
    lon_idx, lat_idx = get_cell_idx(x, y, top_left_x_coords, top_left_y_coords)
    # Select the 10 * 10 pixels
    left_idx = lon_idx - 5
    right_idx = lon_idx + 4
    up_idx = lat_idx - 5
    low_idx = lat_idx + 4
    luminosity_100 = []
    for i in range(left_idx, right_idx + 1):
        for j in range(up_idx, low_idx + 1):
            #"" Get the luminosity of this pixel
            luminosity = bands_data[j, i, 0]
            luminosity_100.append(luminosity)
    luminosity_100 = np.asarray(luminosity_100)
    return pd.Series({'id': idx, 'luminosity': luminosity_100})


df_luminosity = df_dhs.apply(lambda x: get_nightlight_luminosity([x['Cluster'], x['longitude'], x['latitude']]), axis=1)
list_luminosity=[]
for index, row in df_luminosity.iterrows():
    list_luminosity.append(np.array(row['luminosity']))
    
list_luminosity = np.array(list_luminosity).reshape(-1,1)


plt.hist(list_luminosity, bins=30)
plt.xlabel('luminosity')
plt.ylabel('count')
plt.title('luminosity distribution')
plt.xlim([2,62])
plt.ylim([0,1000])
plt.show()

print('zero light intensity =', round(np.sum(list_luminosity==0)/len(list_luminosity)*100,2), '%')
print('total number of images =', len(list_luminosity))

from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(
        n_components=3,
        max_iter=1000,
        tol=1e-10,
        covariance_type='full',
        random_state=32).fit(list_luminosity)

# Predict
lum_pred = gmm.predict(list_luminosity)

pd_lum_pred = pd.DataFrame(list_luminosity,lum_pred).reset_index()#, 'pred_clusters':lum_pred})


pd_lum_pred.columns=['pred_cluster', 'luminosity']
pd_lum_pred.head()

bin_caps = {}
for x in range(3):
    bin_caps[x] = pd_lum_pred[pd_lum_pred['pred_cluster'] == x]['luminosity'].max()

bin_labels = ['low', 'medium', 'high']

assign_labels = {}
for val, label in zip(bin_caps, bin_labels):
    assign_labels[val] = label
print(bin_caps)
print(assign_labels)



def ad_hoc_binning(
    intensity, 
    bin_caps=[0,10,63],
    bin_labels=['low', 'medium', 'high']):

    #bin_caps.append(1e100)
    for val, label in zip(bin_caps, bin_labels):
        if intensity <= val:
            return label


pd_lum_pred['label'] = pd_lum_pred['luminosity'].apply(lambda x: ad_hoc_binning(x))
pd_lum_pred.head()

pd_lum_pred['label'].value_counts(normalize=True)
df_dhs_light_1 = df_dhs_light[['Cluster', 'median_light','Wealth Index Factor Score']]
df_dhs_light.head()

df_dhs_light.to_csv(path+'/DHS_light.csv')





