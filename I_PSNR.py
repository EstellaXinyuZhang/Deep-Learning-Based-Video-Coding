import os
import fnmatch

def generate(pattern, path):
    for root, dirs, files in os.walk(path, topdown=True):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                root_I = root.replace('/home/ubuntu/user_space/vimeo_septuplet/sequences', '/home/ubuntu/user_space/I-frames', )
                if not os.path.exists(root_I):
                    os.makedirs(root_I)
                os.system('bpgenc -f 444 -m 9 ' + root + '/im1.png -o '+ root_I +'/im1_QP27.bpg -q 27')
                os.system('bpgdec '+root_I+'/im1_QP27.bpg -o '+root_I+'/im1_bpg444_QP27.png')


generate('im1.png', '/home/ubuntu/user_space/vimeo_septuplet/sequences/')
