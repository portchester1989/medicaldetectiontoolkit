#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

'''
This preprocessing script loads nrrd files obtained by the data conversion tool: https://github.com/MIC-DKFZ/LIDC-IDRI-processing/tree/v1.0.1
After applying preprocessing, images are saved as numpy arrays and the meta information for the corresponding patient is stored
as a line in the dataframe saved as info_df.pickle.
'''

import os
import openslide
import numpy as np
from multiprocessing import Pool
import pandas as pd
import numpy.testing as npt
from skimage.transform import resize
import subprocess
import pickle
import openslide
import configs
cf = configs.configs()



def resample_array(src_imgs, level = 1):
    this_dimention = src_imgs.level_dimentions[level]
    img = src_imgs.read_region((0,0),level,this_dimention)

    return img,this_dimention


def pp_patient(inputs):

    ix, path = inputs
    pid = path.split('/')[-1]
    img = openslide.OpenSlide(os.path.join(path, '{}.svs'.format(pid)))
    img_arr,this_dimention = resample_array(img)
    #img_arr = np.clip(img_arr, -1200, 600)
    #img_arr = (1200 + img_arr) / (600 + 1200) * 255  # a+x / (b-a) * (c-d) (c, d = new)
    img_arr = img_arr.astype(np.float32)
    img_arr = (img_arr - np.mean(img_arr)) / np.std(img_arr).astype(np.float16)

    df = pd.read_csv(os.path.join(path, '{}.csv'.format(pid)))
    mal_labels = df.label.values
    np.save(os.path.join(cf.pp_dir, '{}_img.npy'.format(pid)), img_arr)

    with open(os.path.join(cf.pp_dir, 'meta_info_{}.pickle'.format(pid)), 'wb') as handle:
        meta_info_dict = {'pid': pid, 'class_target': mal_labels, 'y1':df.y, 'x1':df.x, 'y2':df.y + df.h, 'x2':df.x + df.w}
        pickle.dump(meta_info_dict, handle)



def aggregate_meta_info(exp_dir):

    files = [os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f]
    df = pd.DataFrame(columns=['pid', 'class_target', 'y1', 'x1', 'y2','x2'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print ("aggregated meta info to df with length", len(df))


if __name__ == "__main__":

    paths = [os.path.join(cf.raw_data_dir, ii) for ii in os.listdir(cf.raw_data_dir)]

    if not os.path.exists(cf.pp_dir):
        os.mkdir(cf.pp_dir)

    pool = Pool(processes=12)
    p1 = pool.map(pp_patient, enumerate(paths), chunksize=1)
    pool.close()
    pool.join()
    # for i in enumerate(paths):
    #     pp_patient(i)

    aggregate_meta_info(cf.pp_dir)
    subprocess.call('cp {} {}'.format(os.path.join(cf.pp_dir, 'info_df.pickle'), os.path.join(cf.pp_dir, 'info_df_bk.pickle')), shell=True)
