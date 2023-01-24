import numpy as np
import os
import fnmatch

def find(pattern, path):
    result = []
    result_I = []
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                result.append(root)
                root_I = root.replace('/home/ubuntu/MyFiles/vimeo_septuplet/sequences', '/home/ubuntu/user_space/I-frames', )
                result_I.append(root_I)
                print(root)
                print(root_I)
    return result, result_I


folder,  folder_I = find('im1.png', '/home/ubuntu/MyFiles/vimeo_septuplet/sequences/')
# print(folder)
# np.save('folder.npy', folder)
# np.save('folder_I.npy', folder_I)