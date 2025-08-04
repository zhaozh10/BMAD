import os
import shutil
from tqdm import tqdm
# 源路径和目标路径
src_dir = "/home/z0055k5c/BMAD/data/chest-rsna/val/Ungood"
dst_dir = "/home/z0055k5c/BMAD/data/chest-rsna/val/LungOpacity"

# 如果目标文件夹不存在，自动创建
os.makedirs(dst_dir, exist_ok=True)

# 遍历源文件夹
for filename in tqdm(os.listdir(src_dir)):
    if filename.startswith("Lung_Opacity") and filename.endswith(".png"):
        src_path = os.path.join(src_dir, filename)
        dst_path = os.path.join(dst_dir, filename)
        shutil.copyfile(src_path, dst_path)

print("复制完成。")
