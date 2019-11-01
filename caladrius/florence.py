import os

import json
import numpy as np

import pandas as pd
from pandas.io.json import json_normalize
import rasterio

from tqdm import tqdm

import rasterio.mask
import rasterio.features
import rasterio.warp

from shutil import move

from shapely.ops import transform
import shapely.wkt
import pyproj


# Fraction of image pixels that must be non-zero
NONZERO_PIXEL_THRESHOLD = 0.90

ROOT_DIRECTORY = os.path.join('data', 'hurricane_florence')

# supported damage types
#DAMAGE_TYPES = ['destroyed', 'significant', 'partial', 'none']
DAMAGE_TYPES = ['destroyed', 'major-damage', 'minor-damage', 'no-damage']

BEFORE_FOLDER = os.path.join(ROOT_DIRECTORY, 'Before')
AFTER_FOLDER = os.path.join(ROOT_DIRECTORY, 'After')

JSON_FOLDER = os.path.join(ROOT_DIRECTORY, 'labels')

# output
TARGET_DATA_FOLDER = os.path.join('data', 'Florence')
os.makedirs(TARGET_DATA_FOLDER, exist_ok=True)

# cache
TEMP_DATA_FOLDER = os.path.join(TARGET_DATA_FOLDER, 'temp')
os.makedirs(TEMP_DATA_FOLDER, exist_ok=True)

LABELS_FILE = os.path.join(TEMP_DATA_FOLDER, 'labels.txt')
#ADDRESS_CACHE = os.path.join(TARGET_DATA_FOLDER, 'address_cache.esri')

def damage_quantifier(category):
    stats = {
        'none': {
            'mean': 0.2,
            'std': 0.2
        },
        'partial': {
            'mean': 0.55,
            'std': 0.15
        },
        'significant': {
            'mean': 0.85,
            'std': 0.15
        }
    }

    if category == 'no-damage':
        value = np.random.normal(stats['none']['mean'], stats['none']['std'])
    elif category == 'minor-damage':
        value = np.random.normal(stats['partial']['mean'], stats['partial']['std'])
    else:
        value = np.random.normal(stats['significant']['mean'], stats['significant']['std'])

    return np.clip(value, 0.0, 1.0)


def makesquare(minx, miny, maxx, maxy):
    rangeX = maxx - minx
    rangeY = maxy - miny

    # 20 refers to 5% added to each side
    extension_factor = 20

    # Set image to a square if not square
    if rangeX == rangeY:
        pass
    elif rangeX > rangeY:
        difference_range = rangeX - rangeY
        miny -= difference_range/2
        maxy += difference_range/2
    elif rangeX < rangeY:
        difference_range = rangeY - rangeX
        minx -= difference_range/2
        maxx += difference_range/2
    else:
        pass

    # update ranges
    rangeX = maxx - minx
    rangeY = maxy - miny

    # add some extra border
    minx -= rangeX/extension_factor
    maxx += rangeX/extension_factor
    miny -= rangeY/extension_factor
    maxy += rangeY/extension_factor
    geoms = [{
        "type": "MultiPolygon",
        "coordinates": [[[
            [minx, miny],
            [minx, maxy],
            [maxx, maxy],
            [maxx, miny],
            [minx, miny]
        ]]]
    }]

    return geoms


def saveImage(image, transform, out_meta, folder, name):
    out_meta.update({
            "driver": "PNG",
            "height": image.shape[1],
            "width": image.shape[2],
            "transform": transform
        })
    directory = os.path.join(TEMP_DATA_FOLDER, folder)
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, name)
    with rasterio.open(file_path, 'w', **out_meta) as dest:
        dest.write(image)
    return file_path

def getImage(source_before_image, geometry, moment, name):
    with rasterio.open(source_before_image) as source:
        image, transform = rasterio.mask.mask(source, geometry, crop=True)
        out_meta = source.meta.copy()
        good_pixel_frac = np.count_nonzero(image) / image.size
        if np.sum(image) > 0 and good_pixel_frac > NONZERO_PIXEL_THRESHOLD:
            return saveImage(image, transform, out_meta, moment, name)
        return None


def createDatapoints(df):

    #logger.info('Feature Size {}'.format(len(df)))

    #BEFORE_FILE = os.path.join(BEFORE_FOLDER, 'IGN_Feb2017_20CM.tif')

    before_files = [os.path.join(BEFORE_FOLDER, before_file) for before_file in os.listdir(BEFORE_FOLDER)]
    before_files.sort()

    with open(LABELS_FILE, 'w+') as labels_file:
        #with rasterio.open(BEFORE_FILE) as source_before_image:

        count = 0

        for index, row in tqdm(df.iterrows(), total=df.shape[0]):

            # filter based on damage
            damage = row['_damage']
            if damage not in DAMAGE_TYPES:
                continue

            # pre geom
            bounds_pre = row['geometry_pre'].bounds
            geoms_pre = makesquare(*bounds_pre)

            # post geom
            bounds_post = row['geometry_post'].bounds
            geoms_post = makesquare(*bounds_post)

            # identify data point
            objectID = row['OBJECTID']



            try:
                before_file = getImage(os.path.join(BEFORE_FOLDER, row['file_pre']), geoms_pre,'before','{}.png'.format(objectID))
                after_file = getImage(os.path.join(AFTER_FOLDER, row['file_post']), geoms_post,'after', '{}.png'.format(objectID))
                if (before_file is not None) and os.path.isfile(before_file) and (after_file is not None) \
                        and os.path.isfile(after_file):
                    labels_file.write('{0}.png {1:.4f}\n'.format(objectID, damage_quantifier(damage)))
                    count += 1
            except ValueError as ve:
                    continue

    #logger.info('Created {} Datapoints'.format(count))

def splitDatapoints(filepath):

    with open(filepath) as file:
        datapoints = file.readlines()

    allIndexes = list(range(len(datapoints)))

    np.random.shuffle(allIndexes)

    training_offset = int(len(allIndexes) * 0.8)

    validation_offset = int(len(allIndexes) * 0.9)

    training_indexes = allIndexes[:training_offset]

    validation_indexes = allIndexes[training_offset:validation_offset]

    testing_indexes = allIndexes[validation_offset:]

    split_mappings = {
        'train': [datapoints[i] for i in training_indexes],
        'validation': [datapoints[i] for i in validation_indexes],
        'test': [datapoints[i] for i in testing_indexes]
    }

    for split in split_mappings:

        split_filepath = os.path.join(TARGET_DATA_FOLDER, split)
        os.makedirs(split_filepath, exist_ok=True)

        split_labels_file = os.path.join(split_filepath, 'labels.txt')

        split_before_directory = os.path.join(split_filepath, 'before')
        os.makedirs(split_before_directory, exist_ok=True)

        split_after_directory = os.path.join(split_filepath, 'after')
        os.makedirs(split_after_directory, exist_ok=True)

        with open(split_labels_file, 'w+') as split_file:
            for datapoint in tqdm(split_mappings[split]):
                datapoint_name = datapoint.split(' ')[0]

                before_src = os.path.join(TEMP_DATA_FOLDER, 'before', datapoint_name)
                after_src = os.path.join(TEMP_DATA_FOLDER, 'after', datapoint_name)

                before_dst = os.path.join(split_before_directory, datapoint_name)
                after_dst = os.path.join(split_after_directory, datapoint_name)

                # print('{} => {} !! {}'.format(before_src, before_dst, os.path.isfile(before_src)))
                move(before_src, before_dst)

                # print('{} => {} !! {}'.format(after_src, after_dst, os.path.isfile(after_src)))
                move(after_src, after_dst)

                split_file.write(datapoint)

    return split_mappings


def florence_preprocess():
    #df = pd.DataFrame(columns=['geometry_post', 'feature_type', 'subtype', 'uid'])


    json_files = os.listdir(JSON_FOLDER)
    json_files.sort()

    post_df = pd.DataFrame()
    pre_df = pd.DataFrame()

    for file in json_files:
        json_file = os.path.join(JSON_FOLDER, file)
        with open(json_file, 'r') as f:
            data = json.load(f)

        df_temp = json_normalize(data['features'], 'xy')

        # No buildings on image
        if df_temp.empty:
            continue

        # if pre file, only get coordinates for creating before image stamps
        if 'pre' in file:
            df_temp['file_pre'] = file[0:-4] + 'png'
            df_temp = df_temp.rename(columns={'wkt': 'geometry_pre'})
            pre_df = pre_df.append(df_temp[['geometry_pre','file_pre']], ignore_index=True)
            continue

        # post file, get all relevant info
        df_temp = df_temp.rename(
            columns={'wkt': 'geometry_post', 'properties.feature_type': 'feature_type', 'properties.subtype': '_damage',
                     'properties.uid': 'uid'})
        df_temp.insert(1, "file_post", file[0:-4] + 'png', True)


        post_df = post_df.append(df_temp, ignore_index=True)

    # concatenate pre and post
    df = pd.concat([pre_df, post_df], axis = 1)

    df['geometry_pre'] = df['geometry_pre'].apply(lambda x: shapely.wkt.loads(x))
    df['geometry_post'] = df['geometry_post'].apply(lambda x: shapely.wkt.loads(x))
    df.insert(0, "OBJECTID", range(0, df.shape[0]), True)

    # TODO: Check for empty buildings
    # Remove any empty building shapes
    # df = df.loc[~df['geometry'].is_empty]

    return df

def main():

    df = florence_preprocess()
    ######################################################## START PROCESSING ###################################
    # TODO: why do before images not work
    # Create datapoints
    createDatapoints(df)
    splitDatapoints(LABELS_FILE)


if __name__ == '__main__':
    if __name__ == '__main__':
        main()
