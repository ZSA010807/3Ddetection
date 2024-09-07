import os
import shutil

# 定义原始文件夹和目标文件夹的名称
source_folder = '/home/work/Downloads/waymo/kitti_format/training'
dst_folder = '/home/work/Downloads/waymo/ped'
target_filenames = ['1124081']

for j in range(5):
    folder_name = 'image_' + str(j)
    src_folder = os.path.join(source_folder, folder_name)
    for filename in target_filenames:
        folder_path = os.path.join(dst_folder, filename)
        os.makedirs(folder_path, exist_ok=True)
        file = filename + '.jpg'
        source_file = os.path.join(src_folder, file)
        if os.path.exists(source_file):
            new_filename = 'image_' + str(j) + '+' + file  # 构建新的文件名
            target_file = os.path.join(folder_path, new_filename)
            shutil.copyfile(source_file, target_file)
    # # # 复制目标文件到目标文件夹，并重命名
    # for filename in target_filenames:
    #     source_file = os.path.join(folder, filename)
    #     if os.path.exists(source_file):
    #         new_filename = folder + '_' + filename  # 构建新的文件名
    #         target_file = os.path.join(target_folder, new_filename)
    #         shutil.copyfile(source_file, target_file)

