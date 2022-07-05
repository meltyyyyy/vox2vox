#!/usr/bin/env python
# coding: utf-8

# In[13]:


class Config:
    notebook = "Starter"
    script = "starter"

    seed = 2022

    # Colab Env
    api_path = "/content/drive/MyDrive/workspace/kaggle.json"
    drive_path = "/content/drive/MyDrive/workspace/kaggle-amex"

    # Local Env
    dir_path = "/Users/takeru.abe/Development/research/vox2vox"

    def is_notebook():
        if 'get_ipython' not in globals():
            return False
        env_name = get_ipython().__class__.__name__
        if env_name == 'TerminalInteractiveShell':
            return False
        return True


# In[14]:


import numpy as np
import os
import subprocess
from subprocess import PIPE
import ntpath
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use("seaborn-pastel")
warnings.filterwarnings('ignore')
sns.set_palette("winter_r")


# In[15]:


INPUT = os.path.join(Config.dir_path, 'vim-1')
OUTPUT = os.path.join(Config.dir_path, 'output')
OUTPUT_EXP = os.path.join(OUTPUT, Config.script)
EXP_MODEL = os.path.join(OUTPUT_EXP, "model")
EXP_FIG = os.path.join(OUTPUT_EXP, "fig")
NOTEBOOK = os.path.join(Config.dir_path, "Notebooks")
SCRIPT = os.path.join(Config.dir_path, "scripts")

# make dirs
for dir in [INPUT, OUTPUT_EXP, EXP_MODEL, EXP_FIG, NOTEBOOK, SCRIPT]:
    os.makedirs(dir, exist_ok=True)

if Config.is_notebook():
    notebook_path = os.path.join(NOTEBOOK, Config.notebook + ".ipynb")
    script_path = os.path.join(SCRIPT, Config.script + ".py")
    dir, _ = ntpath.split(script_path)
    subprocess.run(f"mkdir -p {dir}; touch {script_path}",
                   shell=True,
                   stdout=PIPE,
                   stderr=PIPE,
                   text=True)
    subprocess.run(
        f"jupyter nbconvert --to python {notebook_path} --output {script_path}",
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
        text=True)
    subprocess.run(
        f"sed -i '/#\sIn\[[0-9]\]:/d' {script_path}",
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
        text=True)


# In[4]:


import nibabel as nib

file = "Sub1_Ses1_Run1_Trn.nii.gz"
img = nib.load(os.path.join(INPUT, file))
img = img.get_data()
print(img.shape)


# In[5]:


plt.imshow(img[:, :, 9, 336].T, origin='lower')

