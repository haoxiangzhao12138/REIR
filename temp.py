import os
import json
from tqdm import tqdm
# 定义文件路径
annotations_file = '/public/haoxiangzhao/datasets/Objects365v2/annotations/zhiyuan_objv2_train.json'  # 替换成你的标注文件路径
images_folder = '/public/haoxiangzhao/datasets/Objects365v2/images'  # 替换成你的images文件夹路径
new_annotations_file = '/public/haoxiangzhao/datasets/Objects365v2/annotations/objects365v2_train.json' 
# 读取标注文件
with open(annotations_file, 'r') as f:
    annotations = json.load(f)

# annotation_image_names = set()
# for image in tqdm(annotations['images'], desc="Processing annotation file"):
#     annotation_image_names.add(os.path.basename(image['file_name']))

# # 获取images文件夹中实际存在的图片文件名
# existing_image_names = set(os.listdir(images_folder))
# # for img in tqdm(os.listdir(images_folder), desc="Scanning images folder"):
# #     if os.path.isfile(os.path.join(images_folder, img)):
# #         existing_image_names.add(img)

# # 找出缺少的图片文件
# missing_images = annotation_image_names - existing_image_names

missing_images = {"objects365_v1_00320532.jpg", "objects365_v2_00908726.jpg", "objects365_v1_00320534.jpg"}
# 筛选标注文件中的图片信息，排除缺失的图片
annotations['images'] = [image for image in annotations['images'] if os.path.basename(image['file_name']) not in missing_images]

with open(new_annotations_file, 'w') as f:
    json.dump(annotations, f)


# with open(new_annotations_file, 'w') as f:
#     # 写入开头大括号
#     f.write('{\n')
    
#     # 写入 "images" 字段
#     f.write('"images": [\n')
#     for i, image in enumerate(tqdm(annotations['images'], desc="Writing images")):
#         json.dump(image, f)
#         if i < len(annotations['images']) - 1:
#             f.write(',\n')
#         else:
#             f.write('\n')
#     f.write('],\n')

#     # 写入 "annotations" 字段
#     f.write('"annotations": [\n')
#     for i, annotation in enumerate(tqdm(annotations['annotations'], desc="Writing annotations")):
#         json.dump(annotation, f)
#         if i < len(annotations['annotations']) - 1:
#             f.write(',\n')
#         else:
#             f.write('\n')
#     f.write('],\n')
    
#     # 写入 "categories" 字段
#     f.write('"categories": ')
#     json.dump(annotations['categories'], f)
#     f.write(',\n')

#     # 写入 "licenses" 字段
#     f.write('"licenses": ')
#     json.dump(annotations['licenses'], f)
#     f.write('\n}')

print("缺失图片已从标注文件中移除，新文件已保存。")# 提取标注文件中的文件名，只保留文件名部分
pass