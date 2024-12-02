import shutil

src = '/data/ephemeral/home/data'
dst = '/data/ephemeral/home/data_v1'

shutil.copytree(src, dst)