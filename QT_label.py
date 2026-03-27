import sys
import os
import json
import random
from collections import defaultdict
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QTreeWidgetItem,
                             QPushButton, QLabel, QListWidget, QSlider, QFileDialog, QGroupBox,QTreeWidget,QSpinBox,
                             QComboBox, QLineEdit, QMessageBox, QRadioButton, QScrollArea, QButtonGroup)
from PyQt5.QtGui import QPixmap, QPainter, QColor, QFont, QPen
from PyQt5.QtCore import Qt

class AnnotationEditor(QMainWindow):
    def __init__(self, default_json_dir="", default_image_root=""):
        super().__init__()
        self.setWindowTitle("交通事故标签编辑与可视化工具")
        self.setGeometry(100, 100, 1400, 900)

        # 数据路径
        self.json_dir = default_json_dir
        self.image_root = default_image_root
        
        # 类别映射表
        self.category_mapping = [
            "normal", "Collision:car2car", "Collision:car2bike", "Collision:car2person",
            "Collision:car2large", "Collision:large2large", "Collision:large2vru",
            "Collision:bike2bike", "Collision:bike2person", "Collision:obstacle",
            "Rollover", "Collision:others", "Unknown"
        ]

        self.all_videos_info = []
        self.current_video_data = None
        self.current_frame_idx = 0
        self.current_json_path = ""

        self.initUI()
        if self.json_dir and os.path.exists(self.json_dir):
            self._load_jsons_from_path(self.json_dir)

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ================= 左侧：图像显示区域 =================
        left_layout = QVBoxLayout()
        self.image_label = QLabel("加载数据后显示图像")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: #1a1a1a; color: #ffffff;")
        self.image_label.setMinimumSize(800, 600)
        left_layout.addWidget(self.image_label)

        # 帧控制
        slider_layout = QHBoxLayout()
        self.prev_frame_btn = QPushButton("上一帧 (D)")
        self.next_frame_btn = QPushButton("下一帧 (F)")
        self.frame_slider = QSlider(Qt.Horizontal)
        
        # 【修复】：增加 Label 提示，并用 setFixedWidth 强制固定宽度，防止被滑块挤没
        self.jump_label = QLabel("跳转帧号:")
        self.jump_input = QLineEdit()
        self.jump_input.setPlaceholderText("回车跳转")
        self.jump_input.setFixedWidth(80)
        self.jump_input.returnPressed.connect(self.jump_to_frame)
        
        # 取消按钮和滑块的焦点获取，防止它们抢走 D、F 快捷键
        self.prev_frame_btn.setFocusPolicy(Qt.NoFocus)
        self.next_frame_btn.setFocusPolicy(Qt.NoFocus)
        self.frame_slider.setFocusPolicy(Qt.NoFocus)

        self.prev_frame_btn.clicked.connect(self.prev_frame)
        self.next_frame_btn.clicked.connect(self.next_frame)
        self.frame_slider.valueChanged.connect(self.slider_changed)

        slider_layout.addWidget(self.prev_frame_btn)
        slider_layout.addWidget(self.frame_slider)
        slider_layout.addWidget(self.jump_label)   # 显示文本提示
        slider_layout.addWidget(self.jump_input)   # 显示跳转输入框
        slider_layout.addWidget(self.next_frame_btn)
        left_layout.addLayout(slider_layout)

        # ================= 中间：标注编辑面板 =================
        edit_layout = QVBoxLayout()
        edit_group = QGroupBox("标签编辑 (当前视频)")
        edit_vbox = QVBoxLayout()

        # 1. 异常区间设置
        range_layout = QHBoxLayout()
        self.btn_set_start = QPushButton("设为起始帧")
        self.btn_set_end = QPushButton("设为终止帧")
        
        self.btn_set_start.setFocusPolicy(Qt.NoFocus)
        self.btn_set_end.setFocusPolicy(Qt.NoFocus)

        self.edit_start = QLineEdit()
        self.edit_end = QLineEdit()
        self.btn_set_start.clicked.connect(lambda: self.edit_start.setText(str(self.current_frame_idx)))
        self.btn_set_end.clicked.connect(lambda: self.edit_end.setText(str(self.current_frame_idx)))
        
        range_layout.addWidget(QLabel("Start:"))
        range_layout.addWidget(self.edit_start)
        range_layout.addWidget(self.btn_set_start)
        range_layout.addWidget(QLabel("End:"))
        range_layout.addWidget(self.edit_end)
        range_layout.addWidget(self.btn_set_end)
        edit_vbox.addLayout(range_layout)

        # 2. 类别选择 (Radio Buttons)
        edit_vbox.addWidget(QLabel("选择异常类别 (accident_name):"))
        self.cat_group = QButtonGroup()
        scroll = QScrollArea()
        scroll_widget = QWidget()
        scroll_vbox = QVBoxLayout(scroll_widget)
        
        for i, cat in enumerate(self.category_mapping):
            rb = QRadioButton(cat)
            rb.setFocusPolicy(Qt.NoFocus)  # 防止单选框抢夺焦点
            self.cat_group.addButton(rb, i)
            scroll_vbox.addWidget(rb)
        
        scroll.setWidget(scroll_widget)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(380)  # 增大滚动区高度
        edit_vbox.addWidget(scroll)

        # 3. 操作按钮
        self.btn_apply_range = QPushButton("应用类别到异常区间并广播")
        self.btn_apply_range.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold; height: 40px;")
        self.btn_apply_range.setFocusPolicy(Qt.NoFocus)
        self.btn_apply_range.clicked.connect(self.apply_labels_to_range)
        edit_vbox.addWidget(self.btn_apply_range)

        self.btn_save_json = QPushButton("保存修改到 JSON 文件")
        self.btn_save_json.setStyleSheet("background-color: #2196F3; color: white; height: 30px;")
        self.btn_save_json.setFocusPolicy(Qt.NoFocus)
        self.btn_save_json.clicked.connect(self.save_current_json)
        edit_vbox.addWidget(self.btn_save_json)

        edit_group.setLayout(edit_vbox)
        edit_layout.addWidget(edit_group)

        # 信息面板
        info_group = QGroupBox("状态信息")
        info_vbox = QVBoxLayout()
        self.info_video_name = QLabel("视频: -")
        self.info_frame_id = QLabel("帧号: -")
        self.info_accident = QLabel("当前帧标签: -")
        info_vbox.addWidget(self.info_video_name)
        info_vbox.addWidget(self.info_frame_id)
        info_vbox.addWidget(self.info_accident)
        info_group.setLayout(info_vbox)
        edit_layout.addWidget(info_group)
        edit_layout.addStretch()

        # ================= 右侧：列表与搜索 =================
        right_layout = QVBoxLayout()
        
        search_vbox = QVBoxLayout()
        self.category_filter = QComboBox()
        self.category_filter.addItem("All (全部)")
        self.category_filter.currentIndexChanged.connect(self.filter_videos)
        search_vbox.addWidget(QLabel("按类别筛选:"))
        search_vbox.addWidget(self.category_filter)
        
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索视频名...")
        self.search_input.returnPressed.connect(self.search_video)
        search_vbox.addWidget(self.search_input)
        
        self.video_list = QListWidget()
        self.video_list.itemSelectionChanged.connect(self.video_selected)
        
        right_layout.addLayout(search_vbox)
        right_layout.addWidget(self.video_list)

        # 合并布局
        main_layout.addLayout(left_layout, 5)
        main_layout.addLayout(edit_layout, 2)
        main_layout.addLayout(right_layout, 2)

    # ================= 核心逻辑：标签广播与保存 =================
    
    def jump_to_frame(self):
        if not self.current_video_data:
            return
        try:
            target_frame = int(self.jump_input.text().strip())
            max_frame = self.frame_slider.maximum()
            if 0 <= target_frame <= max_frame:
                self.frame_slider.setValue(target_frame)
                self.jump_input.clear()
                self.setFocus()  # 将焦点交回给主窗口
            else:
                QMessageBox.warning(self, "越界", f"请输入 0 到 {max_frame} 之间的帧号！")
        except ValueError:
            QMessageBox.warning(self, "格式错误", "请输入有效的数字！")

    def apply_labels_to_range(self):
        if not self.current_video_data: return
        
        try:
            start_f = int(self.edit_start.text())
            end_f = int(self.edit_end.text())
            selected_id = self.cat_group.checkedId()
            if selected_id == -1:
                QMessageBox.warning(self, "警告", "请先选择一个异常类别！")
                return
            
            new_acc_name = self.category_mapping[selected_id]
        except ValueError:
            QMessageBox.warning(self, "错误", "起始/终止帧必须是数字！")
            return

        self.current_video_data['anomaly_start'] = start_f
        self.current_video_data['anomaly_end'] = end_f

        labels = self.current_video_data.get('labels', [])
        modified_count = 0
        for frame in labels:
            f_id = frame.get('frame_id')
            if start_f <= f_id <= end_f:
                if 'label_accident_name' not in frame:
                    frame['label_accident_name'] = frame.get('accident_name', 'normal')
                frame['accident_name'] = new_acc_name
                modified_count += 1
            else:
                frame['accident_name'] = "normal"

        QMessageBox.information(self, "成功", f"已更新区间 {start_f}-{end_f} 共 {modified_count} 帧为 {new_acc_name}。\n别忘了点击保存！")
        self.update_display()
        self.setFocus() 

    def save_current_json(self):
        if not self.current_video_data or not self.current_json_path: return
        try:
            with open(self.current_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.current_video_data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "保存成功", f"已保存至:\n{self.current_json_path}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存失败: {e}")
        self.setFocus()

    # ================= 数据加载与显示 =================

    def video_selected(self):
        selected = self.video_list.selectedItems()
        if not selected: return
        
        filename = selected[0].text()
        self.current_json_path = os.path.join(self.json_dir, filename)
        
        with open(self.current_json_path, 'r', encoding='utf-8') as f:
            self.current_video_data = json.load(f)
        
        # --- 【新增功能】：自动回显当前视频异常类别 ---
        labels = self.current_video_data.get('labels', [])
        main_accident = "normal"
        # 遍历所有帧，寻找第一个非 normal 的异常标签
        for frame in labels:
            acc = frame.get('accident_name', 'normal')
            if acc != 'normal':
                main_accident = acc
                break

        # 如果找出的标签在映射表中，将其对应的单选框打勾
        if main_accident in self.category_mapping:
            idx = self.category_mapping.index(main_accident)
            btn = self.cat_group.button(idx)
            if btn:
                btn.setChecked(True)
        else:
            # 如果是未知的怪异标签，选择 Unknown 或清空
            idx = self.category_mapping.index("Unknown")
            self.cat_group.button(idx).setChecked(True)
        # -----------------------------------------------

        start = self.current_video_data.get('anomaly_start', "")
        end = self.current_video_data.get('anomaly_end', "")
        self.edit_start.setText(str(start) if start is not None else "")
        self.edit_end.setText(str(end) if end is not None else "")
        
        num_frames = len(self.current_video_data.get('labels', []))
        self.frame_slider.setMaximum(num_frames - 1)
        self.frame_slider.setValue(0)
        self.current_frame_idx = 0
        self.info_video_name.setText(f"视频: {filename}")
        self.update_display()
        self.setFocus()

    def update_display(self):
        if not self.current_video_data: return
        
        frame_data = self.current_video_data['labels'][self.current_frame_idx]
        img_rel_path = frame_data['image_path']
        img_full_path = os.path.normpath(os.path.join(self.image_root, img_rel_path))
        
        acc = frame_data.get('accident_name', 'normal')
        self.info_frame_id.setText(f"帧号: {frame_data['frame_id']}")
        self.info_accident.setText(f"当前标签: {acc}")
        self.info_accident.setStyleSheet("color: red;" if acc != "normal" else "color: green;")

        pixmap = QPixmap(img_full_path)
        if pixmap.isNull():
            self.image_label.setText(f"图丢了:\n{img_full_path}")
            return
            
        pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        painter = QPainter(pixmap)
        painter.setPen(QPen(QColor(255, 0, 0) if acc != "normal" else QColor(0, 255, 0), 3))
        painter.setFont(QFont("Arial", 20, QFont.Bold))
        painter.drawText(30, 50, f"[{self.current_frame_idx}] {acc}")
        painter.end()
        self.image_label.setPixmap(pixmap)

    # ================= 辅助函数 =================
    def _load_jsons_from_path(self, directory):
        self.json_dir = directory
        self.all_videos_info = []
        categories_set = set()
        
        json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
        json_files.sort()
        
        for js in json_files:
            js_path = os.path.join(directory, js)
            try:
                with open(js_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    video_accidents = set()
                    for label in data.get('labels', []):
                        acc = label.get('accident_name', 'normal')
                        if acc != 'normal':
                            video_accidents.add(acc)
                    
                    self.all_videos_info.append({
                        "filename": js,
                        "categories": video_accidents
                    })
                    categories_set.update(video_accidents)
            except Exception as e:
                pass

        self.category_filter.blockSignals(True) 
        self.category_filter.clear()
        self.category_filter.addItem("All (全部)")
        for cat in sorted(list(categories_set)):
            self.category_filter.addItem(cat)
        self.category_filter.blockSignals(False)
        self.filter_videos()

    def filter_videos(self):
        selected_cat = self.category_filter.currentText()
        self.video_list.clear()
        for v_info in self.all_videos_info:
            if selected_cat == "All (全部)" or selected_cat in v_info["categories"]:
                self.video_list.addItem(v_info["filename"])
        if self.video_list.count() > 0:
            self.video_list.setCurrentRow(0)
        self.setFocus() 

    def slider_changed(self):
        self.current_frame_idx = self.frame_slider.value()
        self.update_display()
        self.setFocus()

    def prev_frame(self): self.frame_slider.setValue(self.current_frame_idx - 1)
    def next_frame(self): self.frame_slider.setValue(self.current_frame_idx + 1)
    
    def search_video(self):
        query = self.search_input.text()
        for i in range(self.video_list.count()):
            if query in self.video_list.item(i).text():
                self.video_list.setCurrentRow(i); break
        self.setFocus()

    def keyPressEvent(self, event):
        if (self.search_input.hasFocus() or 
            self.edit_start.hasFocus() or 
            self.edit_end.hasFocus() or 
            self.jump_input.hasFocus()): 
            return
        if event.key() == Qt.Key_D: 
            self.prev_frame()
        elif event.key() == Qt.Key_F: 
            self.next_frame()


class SceneSelector(QMainWindow):
    def __init__(self, metadata_path):
        super().__init__()
        self.setWindowTitle("验收场景挑选器 (Sub-class Selector)")
        self.setGeometry(200, 200, 800, 700)
        self.metadata_path = metadata_path
        
        # 数据结构: { "Collision:bike2bike": ["000768", "000729", ...], ... }
        self.grouped_scenes = defaultdict(list)
        self.total_scenes = 0
        
        self.initUI()
        self.load_metadata()

    def initUI(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # ================= 左侧：树形选择器 =================
        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("<b>视频场景列表 (按异常类别分组)</b>"))
        
        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("异常类别 / 视频ID")
        self.tree.itemChanged.connect(self.on_item_changed)
        left_layout.addWidget(self.tree)

        # ================= 右侧：操作面板 =================
        right_layout = QVBoxLayout()
        
        # 随机抽样组
        sample_group = QGroupBox("自动随机抽样")
        sample_vbox = QVBoxLayout()
        
        box_layout = QHBoxLayout()
        box_layout.addWidget(QLabel("每类抽取数量:"))
        self.spin_n = QSpinBox()
        self.spin_n.setRange(1, 100)
        self.spin_n.setValue(3)
        box_layout.addWidget(self.spin_n)
        sample_vbox.addLayout(box_layout)
        
        self.btn_random = QPushButton("一键随机抽样")
        self.btn_random.setStyleSheet("background-color: #2196F3; color: white; height: 35px;")
        self.btn_random.clicked.connect(self.random_sample)
        sample_vbox.addWidget(self.btn_random)
        
        self.btn_clear = QPushButton("清空所有选择")
        self.btn_clear.clicked.connect(self.clear_selection)
        sample_vbox.addWidget(self.btn_clear)
        
        sample_group.setLayout(sample_vbox)
        right_layout.addWidget(sample_group)
        
        # 状态与保存组
        status_group = QGroupBox("导出与保存")
        status_vbox = QVBoxLayout()
        
        self.lbl_status = QLabel("已选视频: 0")
        self.lbl_status.setStyleSheet("font-size: 14px; font-weight: bold; color: green;")
        status_vbox.addWidget(self.lbl_status)
        
        self.btn_save = QPushButton("导出选中场景至 JSON")
        self.btn_save.setStyleSheet("background-color: #4CAF50; color: white; height: 40px; font-weight: bold;")
        self.btn_save.clicked.connect(self.save_selection)
        status_vbox.addWidget(self.btn_save)
        
        status_group.setLayout(status_vbox)
        right_layout.addWidget(status_group)
        right_layout.addStretch()

        # 比例布局
        main_layout.addLayout(left_layout, 2)
        main_layout.addLayout(right_layout, 1)

    def load_metadata(self):
        if not os.path.exists(self.metadata_path):
            QMessageBox.warning(self, "错误", f"找不到 Metadata 文件: \n{self.metadata_path}")
            return
            
        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
            
        for vid, info in self.data.items():
            cat = info.get('anomaly_class', 'Unknown')
            self.grouped_scenes[cat].append(vid)
            self.total_scenes += 1
            
        self.populate_tree()

    def populate_tree(self):
        self.tree.clear()
        self.tree.blockSignals(True)
        
        for cat, vids in sorted(self.grouped_scenes.items()):
            # 父节点 (类别)
            parent_item = QTreeWidgetItem(self.tree)
            parent_item.setText(0, f"{cat} ({len(vids)} 个)")
            parent_item.setFlags(parent_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsTristate)
            parent_item.setCheckState(0, Qt.Unchecked)
            parent_item.setExpanded(False)
            
            # 子节点 (视频ID)
            for vid in sorted(vids):
                child_item = QTreeWidgetItem(parent_item)
                child_item.setText(0, vid)
                child_item.setFlags(child_item.flags() | Qt.ItemIsUserCheckable)
                child_item.setCheckState(0, Qt.Unchecked)
                # 藏一个真实 ID 到 UserRole 方便提取
                child_item.setData(0, Qt.UserRole, vid) 
                
        self.tree.blockSignals(False)
        self.update_status()

    def on_item_changed(self, item, column):
        self.update_status()

    def clear_selection(self):
        self.tree.blockSignals(True)
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            parent.setCheckState(0, Qt.Unchecked)
            for j in range(parent.childCount()):
                parent.child(j).setCheckState(0, Qt.Unchecked)
        self.tree.blockSignals(False)
        self.update_status()

    def random_sample(self):
        self.clear_selection()
        n = self.spin_n.value()
        self.tree.blockSignals(True)
        
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            child_count = parent.childCount()
            
            # 随机挑选索引
            sample_size = min(n, child_count)
            sampled_indices = random.sample(range(child_count), sample_size)
            
            for idx in sampled_indices:
                parent.child(idx).setCheckState(0, Qt.Checked)
                
        self.tree.blockSignals(False)
        self.update_status()

    def get_selected_scenes(self):
        selected = []
        for i in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(i)
            for j in range(parent.childCount()):
                child = parent.child(j)
                if child.checkState(0) == Qt.Checked:
                    selected.append(child.data(0, Qt.UserRole))
        return selected

    def update_status(self):
        selected = self.get_selected_scenes()
        self.lbl_status.setText(f"已选视频: {len(selected)} / {self.total_scenes}")

    def save_selection(self):
        selected = self.get_selected_scenes()
        if not selected:
            QMessageBox.warning(self, "警告", "当前未选中任何视频！")
            return
        # 设置默认保存目录和默认文件名
        default_dir = "/data/qh/STDA/output/acceptance_workspace/metadata"
        os.makedirs(default_dir, exist_ok=True)   
        default_path = os.path.join(default_dir, "selected_metadata_val.json")
        
        save_path, _ = QFileDialog.getSaveFileName(
            self, 
            "保存选中场景", 
            default_path,  # 传入完整的绝对路径
            "JSON Files (*.json)"
        )
        if save_path:
            selected_data = {k:self.data[k] for k in selected}
            with open(save_path, 'w', encoding='utf-8') as f:     
                json.dump(selected_data, f, indent=4)
            QMessageBox.information(self, "成功", f"成功导出了 {len(selected)} 个场景的列表！\n可以在 AcceptancePipeline 中读取该文件。")


if __name__ == '__main__':
    '''
    检查和修改标签 
    Xmobal直接运行python /home/qh/TDD/SurTAD/QT_label.py
    '''
    # app = QApplication(sys.argv)
    # JSON_DIR = "/data/qh/STDA/data/annotations/"
    # # IMG_ROOT = "/data/qh/STDA/data/frames/"
    # # JSON_DIR = "/data/qh/STDA/data/demo_jsons/"
    # IMG_ROOT = "/data/qh/STDA/data/frames/"
    
    # editor = AnnotationEditor(JSON_DIR, IMG_ROOT)
    # editor.show()
    # sys.exit(app.exec_())

    '''
    手动选择测试集
    '''
    app = QApplication(sys.argv)
    # 指向你的 metadata json 文件
    META_PATH = "/data/qh/STDA/data/metadata/infer_metadata_val.json" 
    ex = SceneSelector(META_PATH)
    ex.show()
    sys.exit(app.exec_())
