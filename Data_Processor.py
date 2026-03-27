# import os
# import cv2
# import json
# import zipfile
# import shutil
# import glob
# import xml.etree.ElementTree as ET
# from collections import defaultdict, Counter
# from pathlib import Path
# from concurrent.futures import ThreadPoolExecutor
# import tqdm

# class TrafficDatasetProcessor:
#     """
#     交通异常检测数据集处理工具类集
#     包含：视频转图、抽帧、重命名、CVAT XML转JSON、数据统计与标签映射
#     """

#     def __init__(self, root_path="/data/qh/Trffic_Demo/Car_accident_detection/Dataset/manual/"):
#         self.root = Path(root_path)
#         # 默认的12分类映射表
#         self.category_mapping = {
#             "normal": "normal",
#             "Collision:car2car": "Collision:car2car",
#             "Collision:bike2car": "Collision:car2bike",
#             "Collision:car2tricycle": "Collision:car2bike",
#             "Collision:car2person": "Collision:car2person",
#             "Collision:car2truck": "Collision:car2large",
#             "Collision:bus2car": "Collision:car2large",
#             "Collision:truck2truck": "Collision:large2large",
#             "Collision:bus2bus": "Collision:large2large",
#             "Collision:bus2truck": "Collision:large2large",
#             "Collision:bike2truck": "Collision:large2vru",
#             "Collision:bike2bus": "Collision:large2vru",
#             "Collision:bus2person": "Collision:large2vru",
#             "Collision:person2truck": "Collision:large2vru",
#             "Collision:bus2tricycle": "Collision:large2vru",
#             "Collision:bike2bike": "Collision:bike2bike",
#             "Collision:bike2tricycle": "Collision:bike2bike",
#             "Collision:bike2person": "Collision:bike2person",
#             "Collision:person2tricycle": "Collision:bike2person",
#             "Collision:car2obstacle": "Collision:obstacle",
#             "Roll over:car": "Rollover", "Roll over:truck": "Rollover",
#             "Roll over:bike": "Rollover", "Roll over:bus": "Rollover",
#             "Roll over:others": "Rollover", "Roll over:None": "Rollover",
#             "Collision:car2others": "Collision:others",
#             "Others": "Unknown", "Collision:None": "Unknown"
#         }

#     # --- 1. 文件管理模块 ---
#     def rename_frames(self, folder_root, padding=6, reverse=False):
#         """
#         重命名文件夹下的图片。
#         用法: processor.rename_frames("extracted_frames", padding=6, reverse=False)
#         :param reverse: True 则将 000001.jpg 转回 1.jpg；False 则转为 000001.jpg
#         """
#         path = self.root / folder_root
#         for folder in tqdm.tqdm(sorted(os.listdir(path)), desc="Renaming"):
#             folder_path = path / folder
#             if not folder_path.is_dir(): continue
            
#             for file in os.listdir(folder_path):
#                 if not file.endswith(".jpg"): continue
#                 num_str = file.split(".")[0]
#                 try:
#                     num = int(num_str)
#                     new_name = f"{num}.jpg" if reverse else f"{num:0{padding}d}.jpg"
#                     os.rename(folder_path / file, folder_path / new_name)
#                 except ValueError: continue

#     def check_sequence_integrity(self, folder_root):
#         """
#         检查文件夹编号是否连续。
#         用法: processor.check_sequence_integrity("extracted_frames")
#         """
#         path = self.root / folder_root
#         folders = sorted([f for f in os.listdir(path) if (path / f).is_dir()])
#         if not folders: return
#         start, end = int(folders[0]), int(folders[-1])
#         missing = [f"{i:06d}" for i in range(start, end + 1) if f"{i:06d}" not in folders]
#         print(f"检查完成。范围: {start}-{end}。缺失数: {len(missing)}")
#         return missing

#     # --- 2. 视频与抽帧模块 ---
#     def video_to_jpg(self, video_path, save_dir_name):
#         """
#         视频转图片。
#         用法: processor.video_to_jpg("raw.mp4", "extracted_frames/video_01")
#         """
#         save_path = self.root / save_dir_name
#         save_path.mkdir(parents=True, exist_ok=True)
#         cap = cv2.VideoCapture(video_path)
#         frame_id = 0
#         while True:
#             ret, frame = cap.read()
#             if not ret: break
#             cv2.imwrite(str(save_path / f"{frame_id:06d}.jpg"), frame)
#             frame_id += 1
#         cap.release()
#         print(f"完成，总帧数: {frame_id}")

#     def sample_frames_multithread(self, src_name, dst_name, sample_rate=3, max_workers=8):
#         """
#         多线程抽帧并重新编号（从000000开始）。
#         用法: processor.sample_frames_multithread("extracted_frames", "sampled_frames", sample_rate=3)
#         """
#         src_path, dst_path = self.root / src_name, self.root / dst_name
#         subfolders = [f for f in sorted(src_path.iterdir()) if f.is_dir()]

#         def _process(subfolder):
#             target_sub = dst_path / subfolder.name
#             target_sub.mkdir(parents=True, exist_ok=True)
#             images = sorted(list(subfolder.glob("*.jpg")))
#             for i, img_file in enumerate(images[::sample_rate]):
#                 shutil.copy2(img_file, target_sub / f"{i:06d}.jpg")
#             return f"{subfolder.name} Done"

#         with ThreadPoolExecutor(max_workers=max_workers) as exec:
#             list(tqdm.tqdm(exec.map(_process, subfolders), total=len(subfolders), desc="Sampling"))

#     # --- 3. 标注转换模块 ---
#     def extract_cvat_zips(self, zip_dir, xml_out_dir):
#         """
#         解压CVAT导出的zip包并重命名annotations.xml。
#         用法: processor.extract_cvat_zips("cvat_zips", "anno_xml")
#         """
#         zin, xout = self.root / zip_dir, self.root / xml_out_dir
#         xout.mkdir(parents=True, exist_ok=True)
#         for fn in os.listdir(zin):
#             if fn.endswith(".zip"):
#                 base = fn.replace("dataset_task_", "").replace(".zip", "")
#                 with zipfile.ZipFile(zin / fn, 'r') as z:
#                     if 'annotations.xml' in z.namelist():
#                         with z.open('annotations.xml') as s, open(xout / f"{base}.xml", 'wb') as t:
#                             t.write(s.read())

#     def convert_xml_to_json(self, xml_file, json_out_dir):
#         """
#         CVAT XML转自定义JSON。
#         用法: processor.convert_xml_to_json("anno_xml/TAD_01.xml", "annotations")
#         """
#         out_path = self.root / json_out_dir
#         out_path.mkdir(parents=True, exist_ok=True)
#         tree = ET.parse(self.root / xml_file)
#         videos = defaultdict(list)

#         for img in tree.getroot().findall('image'):
#             name = img.get('name')
#             parts = name.split('/')
#             v_name, f_name = parts[-2], parts[-1]
#             f_id = int(f_name.split('.')[0])
            
#             # 解析逻辑 (简略版，保留你原有的核心逻辑)
#             sur_view, abnormal, acc_name = True, False, "normal"
#             tag = img.find(".//tag[@label='Video_Tags']")
#             if tag is not None:
#                 attrs = {a.get('name'): a.text for a in tag.findall('attribute')}
#                 sur_view = attrs.get('Sur_View') != 'False'
#                 ab_type = attrs.get('Abnormal_Type', attrs.get('Abnornal_Type', 'None'))
#                 if attrs.get('Abnormal') == 'True' or ab_type != 'None':
#                     abnormal = True
#                     # 这里可以插入你复杂的 Collision_A/B 逻辑...
#                     acc_name = ab_type # 简化示例
            
#             videos[v_name].append({
#                 "frame_id": f_id, "image_path": name, "accident_name": acc_name, 
#                 "objects": [], "_sur_view": sur_view, "_abnormal": abnormal
#             })

#         for vn, frames in videos.items():
#             frames.sort(key=lambda x: x['frame_id'])
#             if not frames or not frames[0]['_sur_view']: continue
            
#             ab_ids = [f['frame_id'] for f in frames if f['_abnormal']]
#             json_data = {
#                 "video_name": vn, "num_frames": len(frames),
#                 "anomaly_start": min(ab_ids) if ab_ids else None,
#                 "anomaly_end": max(ab_ids) if ab_ids else None,
#                 "labels": frames
#             }
#             with open(out_path / f"{vn}.json", 'w', encoding='utf-8') as f:
#                 json.dump(json_data, f, indent=2, ensure_ascii=False)

#     # --- 4. 统计与分析模块 ---
#     def analyze_and_map(self, json_dir, do_mapping=True):
#         """
#         分析数据集并执行12分类映射。
#         用法: processor.analyze_and_map("annotations", do_mapping=True)
#         """
#         path = self.root / json_dir
#         json_files = glob.glob(str(path / "*.json"))
#         f_counts, v_counts = Counter(), Counter()

#         for jf in json_files:
#             with open(jf, 'r+', encoding='utf-8') as f:
#                 data = json.load(f)
#                 v_accs = set()
#                 for lbl in data['labels']:
#                     raw = lbl.get('label_accident_name', lbl['accident_name'])
#                     mapped = self.category_mapping.get(raw, "Unknown") if do_mapping else raw
#                     if do_mapping:
#                         lbl['label_accident_name'], lbl['accident_name'] = raw, mapped
#                     mapped = lbl['accident_name']
#                     f_counts[mapped] += 1
#                     if mapped != 'normal': v_accs.add(mapped)
                
#                 for v_type in v_accs: v_counts[v_type] += 1
#                 if do_mapping:
#                     f.seek(0); json.dump(data, f, indent=2, ensure_ascii=False); f.truncate()

#         print("\n📊 统计报告:")
#         for k, v in v_counts.most_common(): print(f"视频级 - {k}: {v}")
#         # for k, v in f_counts.most_common(): print(f"帧级别 - {k}: {v}")
    
# if __name__ == "__main__":
#     '''
#     分析数据集
#     '''
#     tdp = TrafficDatasetProcessor(root_path="/data/qh/STDA/data/")
#     # tdp.analyze_and_map("annotations", do_mapping=False)
#     tdp.analyze_and_map("D_jsons", do_mapping=True)
#     pass

# # import os
# # import json
# # import glob

# # def batch_update_image_paths(json_dir):
# #     """
# #     批量将 JSON 文件中的 image_path 修改为 "文件夹名/图片名.jpg" 的格式。
# #     例如: "a/b/c/000000/000000.jpg" -> "000000/000000.jpg"
# #     """
# #     print("=" * 50)
# #     print("🚀 开始批量修改 image_path...")
# #     print("=" * 50)

# #     if not os.path.exists(json_dir):
# #         print(f"❌ 错误: 找不到目录 {json_dir}")
# #         return

# #     # 获取所有 json 文件
# #     json_files = glob.glob(os.path.join(json_dir, "*.json"))
    
# #     if not json_files:
# #         print(f"⚠️ 警告: 在 {json_dir} 下没有找到任何 .json 文件！")
# #         return

# #     total_files = len(json_files)
# #     modified_files = 0
# #     total_frames_updated = 0

# #     for filepath in json_files:
# #         try:
# #             # 读取 JSON 数据
# #             with open(filepath, 'r', encoding='utf-8') as f:
# #                 data = json.load(f)

# #             is_modified = False
# #             labels = data.get('labels', [])

# #             for label in labels:
# #                 old_path = label.get('image_path', '')
                
# #                 # 如果路径中包含 '/'，则进行切分截取
# #                 if '/' in old_path:
# #                     parts = old_path.split('/')
# #                     # 至少保证有两层结构才能取 [-2] 和 [-1]
# #                     if len(parts) >= 2:
# #                         new_path = f"{parts[-2]}/{parts[-1]}"
                        
# #                         # 只有当新旧路径不同时才修改
# #                         if old_path != new_path:
# #                             label['image_path'] = new_path
# #                             is_modified = True
# #                             total_frames_updated += 1

# #             # 如果该文件有被修改过，则写回磁盘
# #             if is_modified:
# #                 with open(filepath, 'w', encoding='utf-8') as f:
# #                     json.dump(data, f, indent=2, ensure_ascii=False)
# #                 modified_files += 1

# #         except Exception as e:
# #             print(f"❌ 处理文件 {os.path.basename(filepath)} 时出错: {e}")

# #     print("\n" + "=" * 50)
# #     print("✅ 处理完成！统计信息：")
# #     print(f"📄 扫描的总文件数: {total_files}")
# #     print(f"📝 发生修改的文件: {modified_files}")
# #     print(f"🎞️ 成功更新的帧数: {total_frames_updated}")
# #     print("=" * 50)

# # if __name__ == "__main__":
# #     # 请将此处替换为你的 annotations 文件夹的实际路径
# #     JSON_DIR = "/data/qh/STDA/data/annotations/"
    
# #     batch_update_image_paths(JSON_DIR)


# import os
# import json
# import glob
# import random
# from collections import defaultdict

# def generate_split_metadata(json_dir, train_output, val_output, val_weight=0.3, exclusive=False):
#     """
#     遍历 annotation 目录，按类别分层抽样，生成 train 和 val metadata。
    
#     参数:
#     - exclusive (bool): 
#         - False (默认): 训练集包含【全部数据】，测试集按比例抽取 (二者有交集)
#         - True: 训练集和测试集【完全互斥】，测试集抽完后，剩下的才给训练集
#     """
#     mode_str = "互斥划分 (Exclusive)" if exclusive else "包含式划分 (Overlapping)"
#     print("=" * 60)
#     print(f"🚀 开始解析数据集 | 当前模式: {mode_str}")
#     print("=" * 60)

#     if not os.path.exists(json_dir):
#         print(f"❌ 错误: 找不到目录 {json_dir}")
#         return

#     json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
#     if not json_files:
#         print(f"⚠️ 警告: 在 {json_dir} 下没有找到任何 .json 文件！")
#         return

#     parsed_data = {}
#     class_groups = defaultdict(list)

#     # 1. 完整解析所有 JSON 并按类别分组
#     for filepath in json_files:
#         try:
#             with open(filepath, 'r', encoding='utf-8') as f:
#                 data = json.load(f)

#             video_name = data.get("video_name", os.path.splitext(os.path.basename(filepath))[0])
#             num_frames = data.get("num_frames", 0)
#             anomaly_start = data.get("anomaly_start")
#             anomaly_end = data.get("anomaly_end")

#             # 提取 anomaly_class
#             anomaly_class = "normal" 
#             for label in data.get("labels", []):
#                 acc = label.get("accident_name", "normal")
#                 if acc != "normal":
#                     anomaly_class = acc
#                     break  
            
#             # 过滤正常视频或缺失标注的视频
#             if anomaly_class == "normal" or anomaly_start is None or anomaly_end is None:
#                 continue

#             info = {
#                 "anomaly_start": anomaly_start,
#                 "anomaly_end": anomaly_end,
#                 "anomaly_class": anomaly_class,
#                 "num_frames": num_frames
#             }
            
#             parsed_data[video_name] = info
#             class_groups[anomaly_class].append(video_name)

#         except Exception as e:
#             print(f"❌ 解析文件 {os.path.basename(filepath)} 时出错: {e}")

#     # 2. 分配 Train 和 Val 数据
#     train_metadata = {}
#     val_metadata = {}
    
#     random.seed(42)  # 固定种子保证划分结果可复现
#     print(f"🎲 开始按类别抽样 (Val 比例: {val_weight})")
    
#     for cls, v_names in class_groups.items():
#         n = len(v_names)
        
#         # 计算抽取数量 (上限20)
#         sample_size = min(5, int(n * val_weight)) if n >= 2 else 0
#         # sample_size = max(n, min(5, int(n * val_weight))) if n >= 2 else 0
        
#         # 随机打乱当前类别的所有视频
#         random.shuffle(v_names)
        
#         # 划分列表
#         val_names = v_names[:sample_size]
        
#         # 核心控制逻辑：根据 exclusive 参数决定 train 的内容
#         if exclusive:
#             train_names = v_names[sample_size:]  # 互斥：剩下的给 train
#         else:
#             train_names = v_names                # 包含：全部给 train
        
#         # 写入 Val 字典
#         for v in val_names:
#             info = parsed_data[v].copy()
#             info["subset"] = "val"
#             val_metadata[v] = info
            
#         # 写入 Train 字典
#         for v in train_names:
#             info = parsed_data[v].copy()
#             info["subset"] = "train"
#             train_metadata[v] = info
            
#         print(f"  - [{cls:25s}] 总数: {n:4d} -> Train: {len(train_names):4d} | Val: {len(val_names):4d}")

#     # 3. 分别保存两个 JSON 文件
#     def save_json(data_dict, path, name):
#         try:
#             os.makedirs(os.path.dirname(path), exist_ok=True)
#             with open(path, 'w', encoding='utf-8') as f:
#                 json.dump(data_dict, f, indent=4, ensure_ascii=False)
#             print(f"✅ {name} 生成成功! 共 {len(data_dict)} 个 -> {path}")
#         except Exception as e:
#             print(f"❌ 保存 {name} 时出错: {e}")

#     print("\n" + "=" * 60)
#     save_json(train_metadata, train_output, "Train Metadata")
#     save_json(val_metadata, val_output, "Val Metadata")
#     print("=" * 60)


# # if __name__ == "__main__":
# #     JSON_DIR = "/data/qh/STDA/data/annotations/"
    
# #     TRAIN_OUTPUT = "/data/qh/STDA/data/metadata/metadata_train.json"
# #     VAL_OUTPUT = "/data/qh/STDA/data/metadata/metadata_val.json"
    
# #     # 【测试模式A：保持原样，训练集包含验证集】
# #     # generate_split_metadata(JSON_DIR, TRAIN_OUTPUT, VAL_OUTPUT, val_weight=0.2, exclusive=False)
    
# #     # 【测试模式B：训练集和验证集互斥】
# #     generate_split_metadata(JSON_DIR, TRAIN_OUTPUT, VAL_OUTPUT, val_weight=0.2, exclusive=True)
        


# import os
# import json
# import glob
# import shutil

# def convert_dataset_format(src_json_dir, src_frame_dir, dst_json_dir, dst_frame_dir):
#     print("=" * 50)
#     print("🚀 开始数据格式对齐与帧重命名 (修复物理图像名脱节问题)...")
#     print(f"📂 JSON 输出目录: {dst_json_dir}")
#     print(f"📂 图像 输出目录: {dst_frame_dir}")
#     print("=" * 50)

#     os.makedirs(dst_json_dir, exist_ok=True)
#     os.makedirs(dst_frame_dir, exist_ok=True)

#     json_files = glob.glob(os.path.join(src_json_dir, "*.json"))
    
#     if not json_files:
#         print(f"❌ 找不到 JSON 文件，请检查路径: {src_json_dir}")
#         return

#     for json_path in json_files:
#         filename = os.path.basename(json_path)
#         json_basename = os.path.splitext(filename)[0] 
        
#         # 精准还原原视频文件夹名 (如 d2_000018_02 -> d2_000018)
#         parts = json_basename.split('_')
#         if len(parts) >= 3:
#             source_video_folder = f"{parts[0]}_{parts[1]}"
#         else:
#             source_video_folder = json_basename

#         with open(json_path, 'r', encoding='utf-8') as f:
#             data = json.load(f)

#         labels = data.get('labels', [])
#         if not labels:
#             print(f"⚠️ {filename} 没有 labels，跳过。")
#             continue
        
#         target_video_folder = os.path.join(dst_frame_dir, json_basename)
#         os.makedirs(target_video_folder, exist_ok=True)

#         new_labels = []
#         copied_frames = 0
#         abnormal_frame_ids = []
        
#         # 🎯 核心变更：使用 enumerate 强制生成绝对连续的 0, 1, 2... 帧号
#         for new_frame_id, label in enumerate(labels):
#             accident_name = label.get('accident_name', 'normal')
            
#             # --- 新增逻辑：将分离的 Collision_A 和 B 合并为标准字符串 ---
#             if accident_name == "Collision":
#                 col_a = label.get("Collision_A")
#                 col_b = label.get("Collision_B")
                
#                 # 确保 A 和 B 都存在且不是空值
#                 if col_a and col_b and str(col_a) != "None" and str(col_b) != "None":
#                     # 排序保证生成的格式统一，例如 ["others", "car"] -> ["car", "others"] -> car2others
#                     participants = sorted([str(col_a), str(col_b)])
#                     accident_name = f"Collision:{participants[0]}2{participants[1]}"
            
#             # --- 步骤 A：从老的 image_path 中强行抠出真实的源文件名 ---
#             old_image_path_str = label.get('image_path', '')
#             old_image_name = os.path.basename(old_image_path_str)
#             if not old_image_name:
#                 old_image_id = int(label.get('image_id', label.get('frame_id', 0)))
#                 old_image_name = f"{old_image_id:06d}.jpg"

#             # --- 步骤 B：生成新的文件名与路径 ---
#             new_image_name = f"{new_frame_id:06d}.jpg"
#             new_image_rel_path = f"{json_basename}/{new_image_name}"
            
#             if accident_name != "normal":
#                 abnormal_frame_ids.append(new_frame_id)
            
#             # --- 步骤 C：重构字典 ---
#             # 这里会自动丢弃掉原始冗余的 Collision_A 和 Collision_B 键值对，使结构保持纯净
#             new_label = {
#                 "frame_id": new_frame_id,
#                 "image_path": new_image_rel_path,
#                 "accident_name": accident_name,
#                 "objects": label.get("objects", []),
#                 "label_accident_name": accident_name
#             }
#             new_labels.append(new_label)
            
#             # --- 步骤 D：复制并重命名物理图片 (保持不变) ---
#             src_img_path = os.path.join(src_frame_dir, source_video_folder, old_image_name)
#             dst_img_path = os.path.join(target_video_folder, new_image_name)
            
#             if os.path.exists(src_img_path):
#                 shutil.copy2(src_img_path, dst_img_path)
#                 copied_frames += 1
#             else:
#                 print(f"❌ 警告: 找不到物理原图 {src_img_path}")

#         # --- 核心逻辑 2：动态重算 anomaly_start 和 anomaly_end ---
#         if abnormal_frame_ids:
#             data['anomaly_start'] = min(abnormal_frame_ids)
#             data['anomaly_end'] = max(abnormal_frame_ids)
#         else:
#             data['anomaly_start'] = None
#             data['anomaly_end'] = None

#         # 覆盖原数据并保存
#         data['labels'] = new_labels
#         data['video_name'] = json_basename 
#         data['num_frames'] = len(new_labels) 
        
#         dst_json_file = os.path.join(dst_json_dir, filename)
#         with open(dst_json_file, 'w', encoding='utf-8') as f:
#             json.dump(data, f, indent=2, ensure_ascii=False)
            
#         print(f"✅ {filename} -> 成功复制: {copied_frames} 帧 | 异常区间: {data['anomaly_start']} - {data['anomaly_end']}")

#     print("\n🎉 全部处理完成！图像与 JSON 完美对齐。")

# # if __name__ == "__main__":
# #     # --- 源数据路径 ---
# #     SRC_JSON_DIR = "/data/qh/STDA/data/filtered_jsons/"
# #     SRC_FRAME_DIR = "/data/qh/STDA/data/demo_frame/"
    
# #     # --- 目标数据路径 ---
# #     # JSON 输出路径 (你可以根据需要修改)
# #     DST_JSON_DIR = "/data/qh/STDA/data/D_jsons/"
# #     # 图像输出路径 (严格按照你的要求命名为 frames)
# #     DST_FRAME_DIR = "/data/qh/STDA/data/frames/"
    
# #     convert_dataset_format(SRC_JSON_DIR, SRC_FRAME_DIR, DST_JSON_DIR, DST_FRAME_DIR)


import os
import cv2
import json
import zipfile
import shutil
import glob
import random
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tqdm

# ==========================================
# 共享配置：12分类映射表
# ==========================================
CATEGORY_MAPPING = {
    "normal": "normal",
    "Collision:car2car": "Collision:car2car",
    "Collision:bike2car": "Collision:car2bike",
    "Collision:car2tricycle": "Collision:car2bike",
    "Collision:car2person": "Collision:car2person",
    "Collision:car2truck": "Collision:car2large",
    "Collision:bus2car": "Collision:car2large",
    "Collision:truck2truck": "Collision:large2large",
    "Collision:bus2bus": "Collision:large2large",
    "Collision:bus2truck": "Collision:large2large",
    "Collision:bike2truck": "Collision:large2vru",
    "Collision:bike2bus": "Collision:large2vru",
    "Collision:bus2person": "Collision:large2vru",
    "Collision:person2truck": "Collision:large2vru",
    "Collision:bus2tricycle": "Collision:large2vru",
    "Collision:bike2bike": "Collision:bike2bike",
    "Collision:bike2tricycle": "Collision:bike2bike",
    "Collision:bike2person": "Collision:bike2person",
    "Collision:person2tricycle": "Collision:bike2person",
    "Collision:car2obstacle": "Collision:obstacle",
    "Roll over:car": "Rollover", "Roll over:truck": "Rollover",
    "Roll over:bike": "Rollover", "Roll over:bus": "Rollover",
    "Roll over:others": "Rollover", "Roll over:None": "Rollover",
    "Collision:car2others": "Collision:others",
    "Others": "Unknown", "Collision:None": "Unknown"
}

class MediaProcessor:
    """
    【模块 1】媒体文件处理类
    负责处理视频到图像的转换、图像的重命名以及序列完整性检查。
    """
    def __init__(self, root_path: str):
        self.root = Path(root_path)

    def rename_frames(self, folder_root: str, padding: int = 6, reverse: bool = False):
        """
        重命名指定文件夹下的所有图片。
        
        Args:
            folder_root (str): 存放图像文件夹的根目录相对路径。
            padding (int): 补零的位数，默认6位（如 000001.jpg）。
            reverse (bool): 如果为True，则去除前导零（000001.jpg -> 1.jpg）。
        """
        path = self.root / folder_root
        for folder in tqdm.tqdm(sorted(os.listdir(path)), desc="Renaming frames"):
            folder_path = path / folder
            if not folder_path.is_dir(): 
                continue
            
            for file in os.listdir(folder_path):
                if not file.endswith(".jpg"): 
                    continue
                num_str = file.split(".")[0]
                try:
                    num = int(num_str)
                    new_name = f"{num}.jpg" if reverse else f"{num:0{padding}d}.jpg"
                    os.rename(folder_path / file, folder_path / new_name)
                except ValueError: 
                    continue

    def check_sequence_integrity(self, folder_root: str) -> list:
        """
        检查文件夹中的图像编号是否连续，用于排查抽帧或转移时的丢帧问题。
        
        Args:
            folder_root (str): 要检查的图像文件夹路径。
            
        Returns:
            list: 缺失的图像编号列表。
        """
        path = self.root / folder_root
        folders = sorted([f for f in os.listdir(path) if (path / f).is_dir()])
        if not folders: 
            return []
            
        start, end = int(folders[0]), int(folders[-1])
        missing = [f"{i:06d}" for i in range(start, end + 1) if f"{i:06d}" not in folders]
        print(f"✅ 完整性检查完成。范围: {start}-{end}。缺失帧数: {len(missing)}")
        return missing

    def video_to_jpg(self, video_path: str, save_dir_name: str):
        """
        使用 OpenCV 将单个视频按原始帧率转换为图片序列。
        
        Args:
            video_path (str): 源视频文件的绝对或相对路径。
            save_dir_name (str): 保存图片序列的目标文件夹名称。
        """
        save_path = self.root / save_dir_name
        save_path.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            cv2.imwrite(str(save_path / f"{frame_id:06d}.jpg"), frame)
            frame_id += 1
        cap.release()
        print(f"✅ 视频 {os.path.basename(video_path)} 解码完成，总帧数: {frame_id}")

    def sample_frames_multithread(self, src_name: str, dst_name: str, sample_rate: int = 3, max_workers: int = 8):
        """
        多线程抽帧，并将抽取后的帧重新从 0 开始连续编号。
        
        Args:
            src_name (str): 源图像序列目录。
            dst_name (str): 抽帧后保存的目标目录。
            sample_rate (int): 抽帧间隔（例如3，表示每3帧取1帧）。
            max_workers (int): 线程池最大线程数。
        """
        src_path, dst_path = self.root / src_name, self.root / dst_name
        subfolders = [f for f in sorted(src_path.iterdir()) if f.is_dir()]

        def _process(subfolder):
            target_sub = dst_path / subfolder.name
            target_sub.mkdir(parents=True, exist_ok=True)
            images = sorted(list(subfolder.glob("*.jpg")))
            # 按设定频率抽取，并重命名为严格连续的序号
            for i, img_file in enumerate(images[::sample_rate]):
                shutil.copy2(img_file, target_sub / f"{i:06d}.jpg")
            return f"{subfolder.name} Done"

        with ThreadPoolExecutor(max_workers=max_workers) as exec:
            list(tqdm.tqdm(exec.map(_process, subfolders), total=len(subfolders), desc="Multithread Sampling"))


class AnnotationParser:
    """
    【模块 2】标注解析与统计类
    负责处理 CVAT 导出的原始标注数据，转换为模型训练所需的标准 JSON 格式，并进行基础分析。
    """
    def __init__(self, root_path: str):
        self.root = Path(root_path)
        self.category_mapping = CATEGORY_MAPPING

    def extract_cvat_zips(self, zip_dir: str, xml_out_dir: str):
        """
        批量解压 CVAT 导出的 zip 压缩包，并提取其中的 annotations.xml 重命名。
        
        Args:
            zip_dir (str): 存放 CVAT zip 文件的目录。
            xml_out_dir (str): 提取出的 XML 文件保存目录。
        """
        zin, xout = self.root / zip_dir, self.root / xml_out_dir
        xout.mkdir(parents=True, exist_ok=True)
        for fn in os.listdir(zin):
            if fn.endswith(".zip"):
                base = fn.replace("dataset_task_", "").replace(".zip", "")
                with zipfile.ZipFile(zin / fn, 'r') as z:
                    if 'annotations.xml' in z.namelist():
                        with z.open('annotations.xml') as s, open(xout / f"{base}.xml", 'wb') as t:
                            t.write(s.read())
        print(f"✅ CVAT ZIP 提取完成，保存在: {xml_out_dir}")

    def convert_xml_to_json(self, xml_file: str, json_out_dir: str):
        """
        将提取的 CVAT XML 文件解析为结构化的 JSON 格式，提取异常帧区间和类别信息。
        
        Args:
            xml_file (str): 待解析的单个 XML 文件路径。
            json_out_dir (str): 生成的 JSON 文件的存放目录。
        """
        out_path = self.root / json_out_dir
        out_path.mkdir(parents=True, exist_ok=True)
        tree = ET.parse(self.root / xml_file)
        videos = defaultdict(list)

        for img in tree.getroot().findall('image'):
            name = img.get('name')
            parts = name.split('/')
            v_name, f_name = parts[-2], parts[-1]
            f_id = int(f_name.split('.')[0])
            
            sur_view, abnormal, acc_name = True, False, "normal"
            tag = img.find(".//tag[@label='Video_Tags']")
            
            if tag is not None:
                attrs = {a.get('name'): a.text for a in tag.findall('attribute')}
                sur_view = attrs.get('Sur_View') != 'False'
                ab_type = attrs.get('Abnormal_Type', attrs.get('Abnornal_Type', 'None'))
                
                if attrs.get('Abnormal') == 'True' or ab_type != 'None':
                    abnormal = True
                    acc_name = ab_type 
            
            videos[v_name].append({
                "frame_id": f_id, 
                "image_path": name, 
                "accident_name": acc_name, 
                "objects": [], 
                "_sur_view": sur_view, 
                "_abnormal": abnormal
            })

        for vn, frames in videos.items():
            frames.sort(key=lambda x: x['frame_id'])
            if not frames or not frames[0]['_sur_view']: 
                continue
            
            ab_ids = [f['frame_id'] for f in frames if f['_abnormal']]
            json_data = {
                "video_name": vn, 
                "num_frames": len(frames),
                "anomaly_start": min(ab_ids) if ab_ids else None,
                "anomaly_end": max(ab_ids) if ab_ids else None,
                "labels": frames
            }
            with open(out_path / f"{vn}.json", 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"✅ XML 转 JSON 完成。")

    def analyze_and_map(self, json_dir: str, do_mapping: bool = True):
        """
        遍历并分析目标目录下的 JSON 文件，执行 12 分类映射，并输出数据集统计报告。
        
        Args:
            json_dir (str): 需要分析的 JSON 文件夹。
            do_mapping (bool): 是否将复杂的原始类别重写（映射）为标准 12 分类。
        """
        path = self.root / json_dir
        json_files = glob.glob(str(path / "*.json"))
        f_counts, v_counts = Counter(), Counter()

        for jf in json_files:
            with open(jf, 'r+', encoding='utf-8') as f:
                data = json.load(f)
                v_accs = set()
                
                for lbl in data['labels']:
                    raw = lbl.get('label_accident_name', lbl['accident_name'])
                    mapped = self.category_mapping.get(raw, "Unknown") if do_mapping else raw
                    
                    if do_mapping:
                        lbl['label_accident_name'] = raw
                        lbl['accident_name'] = mapped
                    
                    f_counts[mapped] += 1
                    if mapped != 'normal': 
                        v_accs.add(mapped)
                
                for v_type in v_accs: 
                    v_counts[v_type] += 1
                
                if do_mapping:
                    f.seek(0)
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    f.truncate()

        print("\n📊 数据集统计报告:")
        for k, v in v_counts.most_common(): 
            print(f"  视频级别 - {k:25s}: {v}")


class DatasetAligner:
    """
    【模块 3】数据对齐与清洗类
    用于后期对 JSON 内容和物理图像文件的二次对齐，修复路径问题，统一碰撞命名等。
    """
    @staticmethod
    def batch_update_image_paths(json_dir: str):
        """
        批量简化 JSON 文件中的 image_path 格式（例如: "a/b/c/000000/000000.jpg" -> "000000/000000.jpg"）。
        
        Args:
            json_dir (str): JSON 文件所在目录。
        """
        print("=" * 50)
        print("🚀 开始批量修改 image_path...")
        
        json_files = glob.glob(os.path.join(json_dir, "*.json"))
        total_files, modified_files, total_frames = len(json_files), 0, 0

        for filepath in json_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            is_modified = False
            for label in data.get('labels', []):
                old_path = label.get('image_path', '')
                if '/' in old_path:
                    parts = old_path.split('/')
                    if len(parts) >= 2:
                        new_path = f"{parts[-2]}/{parts[-1]}"
                        if old_path != new_path:
                            label['image_path'] = new_path
                            is_modified = True
                            total_frames += 1

            if is_modified:
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                modified_files += 1

        print(f"✅ 处理完成！扫描文件: {total_files} | 修改文件: {modified_files} | 更新帧数: {total_frames}")

    @staticmethod
    def convert_dataset_format(src_json_dir: str, src_frame_dir: str, dst_json_dir: str, dst_frame_dir: str):
        """
        数据核心对齐方法：
        1. 强制生成绝对连续的 0, 1, 2... 帧号。
        2. 将分离的 Collision_A 和 B 组合为标准字符串 (如 Collision:car2person)。
        3. 动态重算 anomaly_start 和 anomaly_end。
        4. 将旧图片名复制映射为新的图片名并保存到新目录。
        
        Args:
            src_json_dir (str): 原始 JSON 目录
            src_frame_dir (str): 原始图像序列目录
            dst_json_dir (str): 输出清洗后的 JSON 目录
            dst_frame_dir (str): 输出对齐重命名后的图像目录
        """
        print("=" * 50)
        print("🚀 开始数据格式对齐与帧重命名 (修复物理图像脱节)...")
        
        os.makedirs(dst_json_dir, exist_ok=True)
        os.makedirs(dst_frame_dir, exist_ok=True)
        json_files = glob.glob(os.path.join(src_json_dir, "*.json"))

        for json_path in json_files:
            filename = os.path.basename(json_path)
            json_basename = os.path.splitext(filename)[0] 
            
            # 还原原视频文件夹名
            parts = json_basename.split('_')
            source_video_folder = f"{parts[0]}_{parts[1]}" if len(parts) >= 3 else json_basename

            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            labels = data.get('labels', [])
            if not labels:
                continue
            
            target_video_folder = os.path.join(dst_frame_dir, json_basename)
            os.makedirs(target_video_folder, exist_ok=True)

            new_labels, copied_frames, abnormal_frame_ids = [], 0, []
            
            for new_frame_id, label in enumerate(labels):
                accident_name = label.get('accident_name', 'normal')
                
                # 合并 Collision_A 和 B
                if accident_name == "Collision":
                    col_a, col_b = label.get("Collision_A"), label.get("Collision_B")
                    if col_a and col_b and str(col_a) != "None" and str(col_b) != "None":
                        participants = sorted([str(col_a), str(col_b)])
                        accident_name = f"Collision:{participants[0]}2{participants[1]}"
                
                # 寻找真实的物理源文件名
                old_image_path_str = label.get('image_path', '')
                old_image_name = os.path.basename(old_image_path_str)
                if not old_image_name:
                    old_id = int(label.get('image_id', label.get('frame_id', 0)))
                    old_image_name = f"{old_id:06d}.jpg"

                # 建立新的路径规则
                new_image_name = f"{new_frame_id:06d}.jpg"
                new_image_rel_path = f"{json_basename}/{new_image_name}"
                
                if accident_name != "normal":
                    abnormal_frame_ids.append(new_frame_id)
                
                new_labels.append({
                    "frame_id": new_frame_id,
                    "image_path": new_image_rel_path,
                    "accident_name": accident_name,
                    "objects": label.get("objects", []),
                    "label_accident_name": accident_name
                })
                
                # 复制并重命名物理图片
                src_img_path = os.path.join(src_frame_dir, source_video_folder, old_image_name)
                dst_img_path = os.path.join(target_video_folder, new_image_name)
                if os.path.exists(src_img_path):
                    shutil.copy2(src_img_path, dst_img_path)
                    copied_frames += 1

            # 动态更新 metadata 属性
            data['anomaly_start'] = min(abnormal_frame_ids) if abnormal_frame_ids else None
            data['anomaly_end'] = max(abnormal_frame_ids) if abnormal_frame_ids else None
            data['labels'] = new_labels
            data['video_name'] = json_basename 
            data['num_frames'] = len(new_labels) 
            
            with open(os.path.join(dst_json_dir, filename), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        print("\n🎉 数据格式转换与物理对齐全部完成！")


class DatasetSplitter:
    """
    【模块 4】数据集划分与 Metadata 生成类
    按类别分布对标注数据进行分层抽样，生成最终供给 PyTorch Dataset 调用的训练和验证索引。
    """
    @staticmethod
    def generate_split_metadata(json_dir: str, train_output: str, val_output: str, val_weight: float = 0.3, exclusive: bool = False):
        """
        遍历 annotation 目录，按类别分层抽样，生成 train 和 val metadata。
        
        Args:
            json_dir (str): 存放已对齐 JSON 文件的目录。
            train_output (str): 输出的训练集元数据文件路径。
            val_output (str): 输出的验证集元数据文件路径。
            val_weight (float): 验证集抽取比例，默认为 0.3。
            exclusive (bool): 
                - False (默认): 包含式。训练集包含【全部数据】，测试集按比例抽取 (二者有交集)。
                - True: 互斥式。训练集和验证集【完全互斥】，测试集抽完后剩下的才给训练集。
        """
        mode_str = "互斥划分 (Exclusive)" if exclusive else "包含式划分 (Overlapping)"
        print("=" * 60)
        print(f"🚀 开始生成 Metadata | 模式: {mode_str}")
        
        json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
        parsed_data = {}
        class_groups = defaultdict(list)

        for filepath in json_files:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)

            v_name = data.get("video_name", os.path.splitext(os.path.basename(filepath))[0])
            anomaly_start = data.get("anomaly_start")
            anomaly_end = data.get("anomaly_end")

            anomaly_class = "normal" 
            for label in data.get("labels", []):
                acc = label.get("accident_name", "normal")
                if acc != "normal":
                    anomaly_class = acc
                    break  
            
            # 过滤正常视频或缺失标注的视频
            if anomaly_class == "normal" or anomaly_start is None or anomaly_end is None:
                continue

            parsed_data[v_name] = {
                "anomaly_start": anomaly_start,
                "anomaly_end": anomaly_end,
                "anomaly_class": anomaly_class,
                "num_frames": data.get("num_frames", 0)
            }
            class_groups[anomaly_class].append(v_name)

        train_metadata, val_metadata = {}, {}
        random.seed(42)  # 固定种子保证划分复现性
        
        for cls, v_names in class_groups.items():
            n = len(v_names)
            sample_size = max(5, int(n * val_weight)) if n >= 2 else 0
            
            random.shuffle(v_names)
            val_names = v_names[:sample_size]
            train_names = v_names[sample_size:] if exclusive else v_names
            
            for v in val_names:
                info = parsed_data[v].copy()
                info["subset"] = "val"
                val_metadata[v] = info
                
            for v in train_names:
                info = parsed_data[v].copy()
                info["subset"] = "train"
                train_metadata[v] = info
                
            print(f"  - [{cls:25s}] 总数: {n:4d} -> Train: {len(train_names):4d} | Val: {len(val_names):4d}")

        def _save_json(data_dict, path, name):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data_dict, f, indent=4, ensure_ascii=False)
            print(f"✅ {name} 生成成功! 共 {len(data_dict)} 个序列 -> {path}")

        _save_json(train_metadata, train_output, "Train Metadata")
        _save_json(val_metadata, val_output, "Val Metadata")


# ==========================================
# 统一调用示例 (Facade)
# ==========================================
# if __name__ == "__main__":
#     ROOT = "/data/qh/STDA/data/"
    
#     # --- 1. 初始化类 ---
#     media_proc = MediaProcessor(ROOT)
#     anno_proc = AnnotationParser(ROOT)
    
#     # --- 2. 统计并映射初始标注 ---
#     # anno_proc.analyze_and_map("annotations", do_mapping=False)
#     # anno_proc.analyze_and_map("D_jsons", do_mapping=True)
    
#     # --- 3. 批量修正图像路径 ---
#     # DatasetAligner.batch_update_image_paths(os.path.join(ROOT, "annotations"))
    
#     # --- 4. 图像与标注终极对齐 ---
#     # DatasetAligner.convert_dataset_format(
#     #     src_json_dir="/data/qh/STDA/data/filtered_jsons/",
#     #     src_frame_dir="/data/qh/STDA/data/demo_frame/",
#     #     dst_json_dir="/data/qh/STDA/data/D_jsons/",
#     #     dst_frame_dir="/data/qh/STDA/data/frames/"
#     # )
    
#     # --- 5. 划分数据集 (准备投入 PyTorch) ---
#     DatasetSplitter.generate_split_metadata(
#         json_dir="/data/qh/STDA/data/annotations/",
#         # json_dir="/data/qh/STDA/data/demo_jsons/",
#         train_output="/data/qh/STDA/data/metadata/[ex]_metadata_train.json",
#         val_output="/data/qh/STDA/data/metadata/[ex]_metadata_val.json",
#         val_weight=1,
#         exclusive=False
#     )
        
import os
import json
from runner.src.metrics import subclass_ranking_per_scene
'''
single model : high ACC for visualization
'''
val_json = "/data/qh/STDA/data/metadata/[in]_metadata_val.json"
pkl_path = "/data/qh/STDA/output/in,subcls,ep=24,lr=1e-5,plain/eval/results-24.pkl"

# 获取每个子类下按 AP 降序排列的视频字典
auc_scenes = subclass_ranking_per_scene(pkl_path) 

# 加载原始 metadata
with open(val_json, 'r', encoding='utf-8') as f:
    data = json.load(f)

default_dir = "/data/qh/STDA/output/acceptance_workspace/metadata"
os.makedirs(default_dir, exist_ok=True)  # 确保输出目录存在，防止报错
save_path = os.path.join(default_dir, "infer_metadata_val.json")

# 修复后的推导式：先外层循环，后内层循环，并加入存在性校验
selected_data = {
    vid: data[vid] 
    for video_names in auc_scenes.values() 
    for vid in video_names[:20] 
    if vid in data
}

# 保存筛选后的 metadata
with open(save_path, 'w', encoding='utf-8') as f:     
    json.dump(selected_data, f, indent=4, ensure_ascii=False)

print(f"成功去重并提取了 {len(selected_data)} 个视频的 metadata。")
print(f"文件已保存至: {save_path}")