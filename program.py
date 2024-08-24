import os
import threading
import abc

import dearpygui.dearpygui as dpg
import xdialog
import numpy as np
import onnxruntime as ort
import cv2
import yaml
import GPUtil
import psutil

from themes import ThemeHandler
from PIL import Image, ImageDraw
from itertools import zip_longest


class InterfaceVariables(abc.ABC):
    def __init__(self) -> None:
        self._files_opened = 0
        self._files_processed = 0
        self._objects_found = 0

    def files_opened_change(self, value) -> None:
        self._files_opened += value
        dpg.configure_item('filesOpenedText', default_value=f'Открыто изображений: {str(self._files_opened)}  |')

    def files_processed_change_by(self, value) -> None:
        self._files_processed += value
        dpg.configure_item('filesProcessedText', default_value=f'Обработано изображений: {str(self._files_processed)}  |')

    def files_processed_set(self, value) -> None:
        self._files_processed = value
        dpg.configure_item('filesProcessedText', default_value=f'Обработано изображений: {str(self._files_processed)}  |')

    def objects_found_change_by(self, value) -> None:
        self._objects_found += value
        dpg.configure_item('objectsFoundText', default_value=f'Найдено объектов: {str(self._objects_found)}')

    def objects_found_set(self, value) -> None:
        self._objects_found = value
        dpg.configure_item('objectsFoundText', default_value=f'Найдено объектов: {str(self._objects_found)}')


class Settings:
    def __init__(self, program) -> None:
        self.accelerator_list = [gpu.name for gpu in GPUtil.getGPUs()]
        self.accelerator_list.append('CPU')
        self.themes_dict = {'dark': 'Темная', 'light': 'Светлая', 'custom': 'Пользовательская'}
        settings_dict = self.__load_settings()
        self.accelerator = settings_dict[0]
        self.theme = settings_dict[1]
        self.multithreaded = settings_dict[2]
        self.custom_colors = settings_dict[3]
        self.threshold = settings_dict[4]
        self.nms = settings_dict[5]
        self.nms_threshold = settings_dict[6]
        self.theme_handler = ThemeHandler(self.theme, self.custom_colors)
        self.program = program

    def change_accelerator(self, sender) -> None:
        self.accelerator = dpg.get_value(sender)
        self.program.open_loading_window()

    def change_theme(self, sender) -> None:
        self.theme = [key for key, lang in self.themes_dict.items() if lang == dpg.get_value(sender)][0]
        self.theme_handler.change_theme(self.theme, self.custom_colors)
        dpg.configure_item('custom_colors', show=self.theme == 'custom')
        dpg.configure_item('custom_colors_spacer', show=self.theme != 'custom')

    def change_color(self, color_key: str, item) -> None:
        self.custom_colors[color_key] = [round(i) for i in dpg.get_value(item)]
        self.theme_handler.change_color(color_key, self.custom_colors[color_key])

    def change_multithreaded_flag(self, sender) -> None:
        self.multithreaded = dpg.get_value(sender)

    def change_threshold(self, sender) -> None:
        self.threshold = dpg.get_value(sender)

    def change_nms_flag(self, sender) -> None:
        self.nms = dpg.get_value(sender)

    def change_nms_threshold(self, sender) -> None:
        self.nms_threshold = dpg.get_value(sender)

    def __create_new_yaml_file(self):
        with open("settings.yaml", "w") as stream:
            default_settings_dict = {
                "accelerator": "CPU",
                "color_scheme": "dark",
                "multithreaded": True,
                "colors": {
                    "main": [37, 37, 38, 255],
                    "additional": [15, 86, 135, 255],
                    "borders": [78, 78, 78, 255],
                    "text": [255, 255, 255, 255],
                    "widgets": [51, 51, 55, 255]
                },
                "threshold": 5,
                "nms": True,
                "nms_threshold": 50
            }
            yaml.dump(default_settings_dict, stream)
        return list(default_settings_dict.values())

    def __load_settings(self):
        if os.path.isfile("./settings.yaml"):
            with open("./settings.yaml", "r") as stream:
                try:
                    loaded_settings = yaml.safe_load(stream)
                    for setting in ["accelerator", "color_scheme", "multithreaded",
                                    "colors", "threshold", "nms", "nms_threshold"]:
                        if setting not in loaded_settings:
                            return self.__create_new_yaml_file()
                    accelerator = loaded_settings["accelerator"]
                    color_scheme = loaded_settings["color_scheme"]
                    multithreaded = loaded_settings["multithreaded"]
                    colors = loaded_settings["colors"]
                    threshold = loaded_settings["threshold"]
                    nms = loaded_settings["nms"]
                    nms_threshold = loaded_settings["nms_threshold"]
                    if color_scheme not in ['dark', 'light', 'custom'] or multithreaded not in [True, False] \
                            or accelerator not in self.accelerator_list or not isinstance(threshold, int) \
                            or nms not in [True, False] or not isinstance(nms_threshold, int):
                        return self.__create_new_yaml_file()
                    for color_key in colors:
                        if color_key not in ['main', 'additional', 'borders', 'text', 'widgets']:
                            return self.__create_new_yaml_file()
                        elif not isinstance(colors[color_key], list):
                            return self.__create_new_yaml_file()
                        elif False in [isinstance(i, int) for i in colors[color_key]]:
                            return self.__create_new_yaml_file()
                    return [accelerator, color_scheme, multithreaded, colors, threshold, nms, nms_threshold]
                except yaml.YAMLError:
                    return self.__create_new_yaml_file()
        else:
            return self.__create_new_yaml_file()

    def save(self) -> None:
        with open("settings.yaml", "w") as stream:
            settings_dict = {
                "accelerator": self.accelerator,
                "color_scheme": self.theme,
                "multithreaded": self.multithreaded,
                "colors": self.custom_colors,
                "threshold": self.threshold,
                "nms": self.nms,
                "nms_threshold": self.nms_threshold
            }
            yaml.dump(settings_dict, stream)


class Program(InterfaceVariables):
    def __init__(self) -> None:
        super().__init__()
        self.settings = Settings(self)
        self.tools = Tools(self)
        self.last_selected_image = None
        self.opened_files = {}
        self.ort_session = None

    def image_selected(self, sender, app_data, tag) -> None:
        if self.last_selected_image == sender:
            dpg.set_value(self.last_selected_image, not app_data)
            return
        try:
            width, height, channels, data = dpg.load_image(tag)
        except:
            print(tag)
            if self.last_selected_image is None:
                dpg.set_value(sender, not app_data)
                dpg.delete_item(dpg.get_item_parent(sender))
                self.opened_files.pop(tag)
                self.files_opened_change(-1)
                return
            dpg.set_value(self.last_selected_image, not app_data)
            self.last_selected_image = None
            dpg.delete_item(dpg.get_item_parent(sender))
            dpg.delete_item(dpg.get_item_children("plot_axis_y", 1)[0])
            dpg.delete_item("plot_texture")
            if self.opened_files[tag] is not None:
                self.files_processed_change_by(-1)
                self.objects_found_change_by(-len(self.opened_files[tag]))
            self.opened_files.pop(tag)
            self.files_opened_change(-1)
            return
        if self.last_selected_image is None:
            dpg.add_static_texture(width, height, data, tag="plot_texture", parent="textureRegistry")
        else:
            dpg.set_value(self.last_selected_image, not app_data)
            if self.opened_files[tag] is not None:
                self.tools.draw_bboxes(tag)
                dpg.fit_axis_data('plot_axis_y')
                dpg.fit_axis_data('plot_axis_x')
                self.last_selected_image = sender
                return
            dpg.delete_item(dpg.get_item_children("plot_axis_y", 1)[0])
            dpg.delete_item("plot_texture")
            dpg.add_static_texture(width, height, data, tag="plot_texture", parent="textureRegistry")
        self.last_selected_image = sender
        dpg.add_image_series("plot_texture", [0, 0], [width, height], parent="plot_axis_y")
        dpg.fit_axis_data('plot_axis_y')
        dpg.fit_axis_data('plot_axis_x')

    def open_folder_dialog(self) -> None:
        self.tools.check_opened_files()
        directory = xdialog.directory()
        if directory != '':
            files = [f.path for f in os.scandir(directory) if f.name.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            threading.Thread(target=self.tools.open_images, args=(files, self.settings.multithreaded)).start()
            # self.tools.open_images(files, self.settings.multithreaded)

    def open_file_dialog(self) -> None:
        self.tools.check_opened_files()
        files = xdialog.open_file(filetypes=[('Файлы изображений', '*.jpg; *.jpeg; *.png; *.bmp')], multiple=True)
        threading.Thread(target=self.tools.open_images, args=(files, self.settings.multithreaded)).start()
        # self.tools.open_images(files, self.settings.multithreaded)

    def open_loading_window(self) -> None:
        with dpg.window(
            tag='loading_window',
            modal=True,
            no_close=True,
            no_move=True,
            no_resize=True,
            width=500,
            height=250
        ) as loading_window:
            dpg.add_text('Загрузка модели')
            loading_indicator = dpg.add_loading_indicator(pos=[225, 175], width=25, height=25)
        if self.settings.theme == "custom":
            dpg.configure_item(
                loading_indicator,
                color=self.settings.custom_colors['additional'],
                secondary_color=self.settings.custom_colors['additional']
            )
        else:
            dpg.configure_item(
                loading_indicator,
                color=self.settings.theme_handler.themes_colors_dict[self.settings.theme]['additional'],
                secondary_color=self.settings.theme_handler.themes_colors_dict[self.settings.theme]['additional']
            )
        self.tools.open_model()
        dpg.delete_item(loading_window)

    def predict_all(self) -> None:
        if self.tools.check_opened_files():
            return
        self.tools.disable_gui_items()
        self.files_processed_set(0)
        self.objects_found_set(0)
        threading.Thread(target=self.tools.predict_all, daemon=True).start()

    def predict_image(self) -> None:
        if self.last_selected_image is None or self.tools.check_opened_files():
            return
        self.tools.disable_gui_items()
        file_path = dpg.get_item_user_data(self.last_selected_image)
        predicted, objects_found = self.tools.predict_image(file_path)
        if predicted:
            dpg.bind_item_theme(self.last_selected_image, 'imageText_predicted')
            self.tools.draw_bboxes(file_path)
            self.tools.change_image_objects_found_text(self.last_selected_image, objects_found)
            self.files_processed_change_by(1)
        self.tools.enable_gui_items()

    def open_settings(self) -> None:
        main_width = dpg.get_viewport_width()
        main_height = dpg.get_viewport_height()
        with dpg.window(
            label="Настройки", width=650, height=400, modal=True,
            pos=[main_width//2 - 325, main_height//2 - 200],
            on_close=lambda: dpg.delete_item(SettingsWindow)
        ) as SettingsWindow:
            with dpg.tab_bar():
                with dpg.tab(label="Персонализация"):
                    color_scheme = dpg.add_combo(
                        label="Цветовая схема",
                        items=list(self.settings.themes_dict.values()),
                        callback=self.settings.change_theme
                    )
                    with dpg.child_window(
                        tag='custom_colors',
                        height=-28,
                        show=self.settings.theme == 'custom'
                    ):
                        main_color = dpg.add_color_edit(
                            label="Основной цвет", callback=lambda: self.settings.change_color('main', main_color)
                        )
                        additional_color = dpg.add_color_edit(
                            label="Дополнительный цвет",
                            callback=lambda: self.settings.change_color('additional', additional_color)
                        )
                        borders_color = dpg.add_color_edit(
                            label="Цвет границ",
                            callback=lambda: self.settings.change_color('borders', borders_color)
                        )
                        text_color = dpg.add_color_edit(
                            label="Цвет текста",
                            callback=lambda: self.settings.change_color('text', text_color)
                        )
                        widgets_color = dpg.add_color_edit(
                            label="Цвет элементов",
                            callback=lambda: self.settings.change_color('widgets', widgets_color)
                        )
                    dpg.add_child_window(
                        tag='custom_colors_spacer', height=-28, show=self.settings.theme != 'custom'
                    )
                with dpg.tab(label="Производительность"):
                    accelerator = dpg.add_combo(
                        label="Ускоритель",
                        items=self.settings.accelerator_list,
                        callback=self.settings.change_accelerator
                    )
                    multithreaded = dpg.add_checkbox(
                        label="Многопоточная загрузка изображений",
                        callback=self.settings.change_multithreaded_flag
                    )
                    threshold = dpg.add_slider_int(
                        label="Порог точности, %",
                        callback=self.settings.change_threshold
                    )
                    nms = dpg.add_checkbox(
                        label="Включить NMS",
                        callback=self.settings.change_nms_flag
                    )
                    nms_threshold = dpg.add_slider_int(
                        label="Порог NMS, %",
                        callback=self.settings.change_nms_threshold
                    )
                    dpg.add_child_window(height=-28, border=False)
            with dpg.table(height=15, header_row=False):
                dpg.add_table_column(width_stretch=True)
                dpg.add_table_column(width_fixed=True)
                with dpg.table_row():
                    dpg.add_spacer()
                    dpg.add_button(label='Закрыть', callback=lambda: dpg.delete_item(SettingsWindow))
        dpg.set_value(color_scheme, self.settings.themes_dict[self.settings.theme])
        dpg.set_value(accelerator, self.settings.accelerator)
        dpg.set_value(main_color, self.settings.custom_colors['main'])
        dpg.set_value(additional_color, self.settings.custom_colors['additional'])
        dpg.set_value(borders_color, self.settings.custom_colors['borders'])
        dpg.set_value(text_color, self.settings.custom_colors['text'])
        dpg.set_value(widgets_color, self.settings.custom_colors['widgets'])
        dpg.set_value(multithreaded, self.settings.multithreaded)
        dpg.set_value(threshold, self.settings.threshold)
        dpg.set_value(nms, self.settings.nms)
        dpg.set_value(nms_threshold, self.settings.nms_threshold)


class Tools:
    def __init__(self, program) -> None:
        self.program = program

    def draw_bboxes(self, file_path) -> None:
        dpg.delete_item(dpg.get_item_children("plot_axis_y", 1)[0])
        dpg.delete_item("plot_texture")
        if not os.path.isfile(file_path):
            return
        img = Image.open(file_path).copy()
        img_draw = ImageDraw.Draw(img, "RGBA")
        for (bbox, score) in self.program.opened_files[file_path]:
            bbox = bbox.tolist()
            img_draw.rectangle(tuple(bbox), outline=(255, 0, 0, 127), width=4)
        img.putalpha(255)
        dpg_image = np.frombuffer(img.tobytes(), dtype=np.uint8) / 255.0
        dpg.add_static_texture(img.width, img.height, dpg_image, tag="plot_texture", parent="textureRegistry")
        dpg.add_image_series("plot_texture", [0, 0], [img.width, img.height], parent="plot_axis_y")

    def get_model_info(self):
        model_inputs = self.program.ort_session.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        input_shape = model_inputs[0].shape
        model_output = self.program.ort_session.get_outputs()
        output_names = [model_output[i].name for i in range(len(model_output))]
        return input_names, input_shape, output_names

    def predict_all(self) -> None:
        opened_files_list = list(self.program.opened_files.keys())
        opened_files_list.sort()
        opened_files_list.sort(key=lambda x: len(x))
        for file_path in list(opened_files_list):
            predicted, objects_found = self.predict_image(file_path)
            if predicted:
                self.program.files_processed_change_by(1)
                dpg.bind_item_theme(file_path, 'imageText_predicted')
                self.change_image_objects_found_text(file_path, objects_found)
            if file_path == self.program.last_selected_image:
                self.draw_bboxes(file_path)
        self.enable_gui_items()

    def change_image_objects_found_text(self, image_selected, objects_found) -> None:
        image_list_row = dpg.get_item_parent(image_selected)
        objects_found_text = dpg.get_item_children(dpg.get_item_children(image_list_row, 1)[0], 1)[1]
        if objects_found > 0:
            dpg.set_value(objects_found_text, f' {objects_found} об. ')
        else:
            dpg.set_value(objects_found_text, '')

    def xywh2xyxy(self, xywh):
        xyxy = np.copy(xywh)
        xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
        xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
        xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
        xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2
        return xyxy

    def nms(self, boxes, threshold):
        if len(boxes) == 0:
            return []
        filtered_boxes_ids = []
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        indexes = np.argsort(y2)
        while len(indexes) > 0:
            last = len(indexes) - 1
            i = indexes[last]
            filtered_boxes_ids.append(i)
            xx1 = np.maximum(x1[i], x1[indexes[:last]])
            yy1 = np.maximum(y1[i], y1[indexes[:last]])
            xx2 = np.minimum(x2[i], x2[indexes[:last]])
            yy2 = np.minimum(y2[i], y2[indexes[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[indexes[:last]]
            indexes = np.delete(indexes, np.concatenate(([last], np.where(overlap > threshold)[0])))
        return boxes[filtered_boxes_ids].astype("int")

    def predict_image(self, file_path) -> tuple[bool, int]:
        input_names, input_shape, output_names = self.get_model_info()
        if not os.path.isfile(file_path):
            self.program.opened_files.pop(file_path)
            return False, 0
        image = cv2.imread(file_path)
        image_height, image_width = image.shape[:2]
        input_height, input_width = input_shape[2:]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (input_width, input_height))
        input_image = resized / 255.0
        input_image = input_image.transpose(2, 0, 1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        outputs = self.program.ort_session.run(output_names, {input_names[0]: input_tensor})[0]
        predictions = np.squeeze(outputs)
        conf_threshold = self.program.settings.threshold / 100
        scores = np.squeeze(predictions[:, 4:])
        predictions = predictions[scores >= conf_threshold, :]
        scores = scores[scores >= conf_threshold]
        predictions = predictions[np.argsort(scores)]
        boxes = predictions[:, :4]
        boxes = self.xywh2xyxy(boxes)
        boxes *= np.array([image_width, image_height, image_width, image_height])
        boxes = boxes.astype(np.int32)
        if self.program.settings.nms:
            boxes = self.nms(boxes, self.program.settings.nms_threshold / 100)
        objects_found = len(boxes)
        self.program.objects_found_change_by(objects_found)
        self.program.opened_files[file_path] = list(zip(boxes, scores))
        return True, objects_found

    def open_image(self, file_path: str) -> None:
        if os.path.basename(file_path).endswith(('.jpg', '.jpeg')):
            img = Image.open(file_path)
            img.draft('RGB', (img.width, img.height))
        else:
            img = Image.open(file_path).copy()
        img = img.resize((100, img.height * 100 // img.width))
        img.putalpha(255)
        dpg_image = np.frombuffer(img.tobytes(), dtype=np.uint8) / 255.0
        texture = dpg.add_static_texture(width=img.width, height=img.height,
                                         default_value=dpg_image, parent="textureRegistry")
        with dpg.table_row(height=100, parent="imagesList"):
            with dpg.group():
                dpg.add_image(texture)
                mark = dpg.add_text('')
            dpg.bind_item_theme(mark, 'image_objects_mark')
            sel = dpg.add_selectable(
                label=os.path.basename(file_path),
                span_columns=True,
                height=100,
                callback=self.program.image_selected,
                user_data=file_path,
                tag=file_path,
            )
            with dpg.tooltip(sel, delay=0.1):
                dpg.add_text(file_path)

    def sort_table(self) -> None:
        table = "imagesList"
        sort_specs = [['text_column', 1]]
        if sort_specs is None:
            return
        rows = dpg.get_item_children(table, 1)
        sortable_list = []
        for row in rows:
            second_cell = dpg.get_item_children(row, 1)[1]
            sortable_list.append([row, dpg.get_item_label(second_cell)])
        sortable_list.sort(key=lambda x: x[1], reverse=sort_specs[0][1] < 0)
        sortable_list.sort(key=lambda x: len(x[1]), reverse=sort_specs[0][1] < 0)
        new_order = []
        for pair in sortable_list:
            new_order.append(pair[0])
        dpg.reorder_items(table, 1, new_order)

    def check_opened_files(self) -> bool:
        if self.program.opened_files == {}:
            return True
        for file in list(self.program.opened_files.keys()):
            if not os.path.isfile(file):
                self.program.opened_files.pop(file)
        if self.program.opened_files == {}:
            return True
        return False

    def disable_gui_items(self) -> None:
        menu_bar = dpg.get_item_children('menuBar', 1)
        for i in menu_bar:
            dpg.disable_item(i)

    def enable_gui_items(self) -> None:
        menu_bar = dpg.get_item_children('menuBar', 1)
        for i in menu_bar:
            dpg.enable_item(i)

    def open_model(self) -> None:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        session_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL
        session_options.intra_op_num_threads = int(psutil.cpu_count(logical=True)*0.9)
        model_path = './model/best.onnx'

        if self.program.settings.accelerator != 'CPU':
            device_id = self.program.settings.accelerator_list.index(self.program.settings.accelerator)
            exe_provider_list = [
                (
                    'CUDAExecutionProvider',
                    {
                        'device_id': device_id,
                        'cudnn_conv_use_max_workspace': '1',
                        'cudnn_conv_algo_search': 'DEFAULT',
                        'gpu_mem_limit': int(GPUtil.getGPUs()[device_id].memoryFree) * 1024 * 1024
                    }
                ),
                'CPUExecutionProvider'
            ]
        else:
            exe_provider_list = ['CPUExecutionProvider']

        self.program.ort_session = ort.InferenceSession(model_path, session_options, providers=exe_provider_list)

    def open_images(self, files, multithreaded: bool) -> None:
        files = list(set(files) - set(self.program.opened_files.keys()))
        self.program.opened_files = {**dict(zip_longest(files, [])), **self.program.opened_files}
        if files:
            self.disable_gui_items()
            if multithreaded:
                self.open_images_multithreaded(files)
            else:
                self.open_images_singlethreaded(files)
            self.sort_table()
            self.enable_gui_items()

    def open_images_multithreaded(self, files) -> None:
        for i in range(0, len(files), 30):
            threads = [threading.Thread(target=self.open_image, args=(file_path, )) for file_path in files[i:i+30]]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
                self.program.files_opened_change(1)

    def open_images_singlethreaded(self, files) -> None:
        for file_path in files:
            self.open_image(file_path)
            self.program.files_opened_change(1)
