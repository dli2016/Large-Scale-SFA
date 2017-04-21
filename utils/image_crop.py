
import os
import sys
import Image

def listAllFiles(dir_name):
    full_names = []
    files = []
    for (parent,dirname,filenames) in os.walk(dir_name):
        full_names = [parent+fn for fn in filenames]
        files.extend(filenames)
        break
    return files, full_names


def cropImage(in_filenames, in_path, out_path, crop_size):
    idx = 0
    for full_filename in in_path:
        filename = in_filenames[idx]
        print filename
        save_name = out_path + "/" + filename
        img_obj = Image.open(full_filename)
        img_crp = img_obj.crop(crop_size)

        img_crp.save(save_name)
        idx = idx + 1
    return



if __name__ == '__main__':
    dir_name = sys.argv[1]
    out_path = sys.argv[2]

    crop_bb = (16, 16, 80, 144)

    files, full_names = listAllFiles(dir_name)
    cropImage(files, full_names, out_path, crop_bb)


