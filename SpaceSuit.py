
# coding: utf-8

# In[1]:


import os, json, requests, math
from skimage import io
from io import BytesIO
import matplotlib.pyplot as plt
import warnings
import random
import numpy as np
import glob
import tensorflow as tf

get_ipython().run_line_magic('matplotlib', 'inline')

random.seed(1)

TOKEN = "4b6a3111-3232-41de-ac00-c439b873aa47"

def get_ASNARO_scene(min_lat, min_lon, max_lat, max_lon):
    url = "https://gisapi.tellusxdp.com/api/v1/asnaro1/scene"         + "?min_lat={}&min_lon={}&max_lat={}&max_lon={}".format(min_lat, min_lon, max_lat, max_lon)
    
    headers = {
        "content-type": "application/json",
        "Authorization": "Bearer " + TOKEN
    }
    
    r = requests.get(url, headers=headers)
    return r.json()

def get_tile_num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def get_tile_bbox(z, x, y):
   
    def num2deg(xtile, ytile, zoom):       #https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames#Python
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lon_deg, lat_deg)
    
    right_top = num2deg(x + 1, y, z)
    left_bottom = num2deg(x, y + 1, z)
    return (left_bottom[0], left_bottom[1], right_top[0], right_top[1])

def get_ASNARO_image(scene_id, zoom, xtile, ytile):
    url = " https://gisapi.tellusxdp.com/ASNARO-1/{}/{}/{}/{}.png".format(scene_id, zoom, xtile, ytile)
    headers = {
        "Authorization": "Bearer " + TOKEN
    }
    
    r = requests.get(url, headers=headers)
    return io.imread(BytesIO(r.content))


# In[3]:


# We conduct the following codes under the development environment of Tellus.

# This is the directory to put the satellite data.
os.mkdir('Data_ASNARO1_spaceapp')


# In[5]:


os.mkdir('out_gogh')


# In[6]:


os.mkdir('out_monet')


# In[ ]:


pip install tensorflow==1.15.0


# In[7]:


# Register by yourself.
my_lat =  35.01 # Kyoto


# In[8]:


# Find the places with similar latitude to where you are living.
scene = get_ASNARO_scene(my_lat-5.0, -180.0,my_lat+5.0, 180.0)


# In[ ]:


random_scenes = random.sample(scene,10)
zoom = 17


# In[ ]:


count = 0

for scene in random_scenes: 
    (xtile, ytile) = get_tile_num(scene['max_lat'], scene['min_lon'], zoom)
    
    x_len = 0
    while(True):
        bbox = get_tile_bbox(zoom, xtile + x_len + 1, ytile)
        tile_lon = bbox[0]
        if(tile_lon > scene['max_lon']):
            break
        x_len += 1

    y_len = 0
    while(True):
        bbox = get_tile_bbox(zoom, xtile, ytile + y_len + 1)
        tile_lat = bbox[3]
        if(tile_lat < scene['min_lat']):
            break
        y_len += 1

    for x in range(x_len):
        for y in range(y_len):    
            try:
                img = get_ASNARO_image(scene['entityId'], zoom, xtile + x, ytile + y)
                if(np.count_nonzero(np.ravel(img))/len(np.ravel(img)) > 0.999):
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')                   
                        io.imsave(os.getcwd()+'/Data_ASNARO1_spaceapp/{}_{}_{}.png'.format(zoom,
 xtile + x, ytile + y), img)
                        count += 1
                        break
            except Exception as e:
                print(e)
            #取得枚数が1枚を超えたら次のシーンへ移る。
        else:
            continue
        break
    if count >= 4:
        break


# In[ ]:


file_path = glob.glob("./Data_ASNARO1_spaceapp/*") 
file_path


# In[ ]:


# Show satellite data for you to pick up.
import cv2

plt.figure(figsize=(11,11))

count=1
for item in file_path:
    img = cv2.imread(item)
    plt.subplot(2,2,count)
    plt.imshow(img)
    count+=1


# In[ ]:


print("Which satellite data did you like?")
num_sat = int(input())
print(num_sat)


# In[ ]:


print("Do you wanna make it Art using AI?")
print("1. Convert to Gogh.")
print("2. Convert to Monet.")
print("3. Not use AI.")

num = int(input())
print(num)


# In[ ]:


img_path = file_path[num_sat]

if num==1:
    print("1. Convert to Gogh.")
    get_ipython().run_line_magic('run', './neural_style_transfer.py ./Data_ASNARO1_spaceapp/17_116393_51635.png ./gogh.jpg out_gogh/gogh')
elif num==2:
    print("2. Convert to Monet.")
    get_ipython().run_line_magic('run', './neural_style_transfer.py ./Data_ASNARO1_spaceapp/17_116393_51635.png ./monet_2.jpg out_monet/monet')
else: 
    print("3. Not use AI.")


# In[ ]:


img_path = file_path[num_sat]
print(img_path)


# In[ ]:


plt.figure(figsize=(11,11))

img_or = cv2.imread(img_path)
plt.subplot(1,2,1)
plt.imshow(img_or)
plt.title("Original")

img_conv = cv2.imread("./out_monet/monet_at_iteration_9.png")

plt.subplot(1,2,2)
plt.imshow(img_conv)
plt.title("Converted using AI")

