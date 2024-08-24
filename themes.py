import dearpygui.dearpygui as dpg


def load_themes():
    with dpg.theme(tag='bottom_panel_text'):
        with dpg.theme_component(dpg.mvButton):
            dpg.add_theme_style(dpg.mvStyleVar_ButtonTextAlign, 1, 0.5, category=dpg.mvThemeCat_Core)
            dpg.add_theme_color(dpg.mvThemeCol_Button, (0, 0, 0, 0))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (0, 0, 0, 0))
            dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (0, 0, 0, 0))
    
    with dpg.theme(tag='imageText_predicted'):
        with dpg.theme_component(dpg.mvSelectable):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (0, 255, 0, 255))
        with dpg.theme_component(dpg.mvSelectable, enabled_state=False):
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (0, 125, 0, 255))

    with dpg.theme(tag='image_objects_mark'):
        with dpg.theme_component(dpg.mvText):
            dpg.add_theme_color(dpg.mvThemeCol_Text, (255, 0, 255, 255))
        with dpg.theme_component(dpg.mvText, enabled_state=False):
            dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, (125, 0, 125, 255))


class ThemeHandler:
    themes_colors_dict = {
        "dark": {
            "main": [37, 37, 38, 255],
            "additional": [15, 86, 135, 255],
            "borders": [78, 78, 78, 255],
            "text": [255, 255, 255, 255],
            "widgets": [51, 51, 55, 255]
        },
        "light": {
            "main": [235, 235, 235, 255],
            "additional": [45, 116, 165, 255],
            "borders": [100, 100, 100, 255],
            "text": [0, 0, 0, 255],
            "widgets": [200, 200, 200, 255]
        }
    }

    def __init__(self, color_scheme: str, colors: dict) -> None:
        if color_scheme == "custom":
            self.theme = self.__create_global_theme(colors)
        else:
            self.theme = self.__create_global_theme(ThemeHandler.themes_colors_dict[color_scheme])
    
    def change_color(self, color_key: str, new_color: list) -> None:
        if color_key == 'widgets' or color_key == 'text':
            print(color_key)
            for color in self.theme[color_key + '_disabled']:
                nw_color = new_color[:-1].append(new_color[-1]//2)
                dpg.set_value(color, nw_color)
        for color in self.theme[color_key]:
            dpg.set_value(color, new_color)

    def change_theme(self, color_scheme: str, colors: dict) -> None:
        if color_scheme == "custom":
            self.__change_colors(colors)
        else:
            self.__change_colors(self.themes_colors_dict[color_scheme])

    def __change_colors(self, colors: dict) -> None:
        for color_key, new_color in colors.items():
            self.change_color(color_key, new_color)

    def __create_global_theme(self, colors: dict):
        with dpg.theme() as theme:
            with dpg.theme_component(dpg.mvAll) as main:
                dpg.add_theme_color(dpg.mvThemeCol_WindowBg, colors["main"])
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, colors["main"])
                dpg.add_theme_color(dpg.mvThemeCol_PopupBg, colors["main"])
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgCollapsed, colors["main"])
                dpg.add_theme_color(dpg.mvThemeCol_TitleBg, colors["main"])
                dpg.add_theme_color(dpg.mvThemeCol_ResizeGrip, colors["main"])
            with dpg.theme_component(dpg.mvAll) as additional:
                dpg.add_theme_color(dpg.mvThemeCol_TitleBgActive, colors["additional"])
                dpg.add_theme_color(dpg.mvThemeCol_CheckMark, colors["additional"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, colors["additional"])
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, colors["additional"])
                dpg.add_theme_color(dpg.mvThemeCol_TabHovered, colors["additional"])
                dpg.add_theme_color(dpg.mvThemeCol_TabActive, colors["additional"])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderHovered, colors["additional"])
                dpg.add_theme_color(dpg.mvThemeCol_HeaderActive, colors["additional"])
                dpg.add_theme_color(dpg.mvThemeCol_ResizeGripHovered, colors["additional"])
                dpg.add_theme_color(dpg.mvThemeCol_ResizeGripActive, colors["additional"])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgActive, colors["additional"])
                dpg.add_theme_color(dpg.mvThemeCol_FrameBgHovered, colors["additional"])
            with dpg.theme_component(dpg.mvAll) as borders:
                dpg.add_theme_color(dpg.mvThemeCol_Border, colors["borders"])
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrab, colors["borders"])
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabActive, colors["borders"])
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarGrabHovered, colors["borders"])
            with dpg.theme_component(dpg.mvAll) as text:
                dpg.add_theme_color(dpg.mvThemeCol_Text, colors["text"])
            with dpg.theme_component(dpg.mvAll) as widgets:
                dpg.add_theme_color(dpg.mvThemeCol_FrameBg, colors["widgets"])
                dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, colors["widgets"])
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, colors["widgets"])
                dpg.add_theme_color(dpg.mvThemeCol_Button, colors["widgets"])
                dpg.add_theme_color(dpg.mvThemeCol_Tab, colors["widgets"])
                dpg.add_theme_color(dpg.mvThemeCol_Header, colors["widgets"])
            with dpg.theme_component(dpg.mvAll, enabled_state=False) as text_disabled:
                dpg.add_theme_color(dpg.mvThemeCol_TextDisabled, colors["text"][:-1].append(colors["text"][-1]//2))
            with dpg.theme_component(dpg.mvAll, enabled_state=False) as widgets_disabled:
                dpg.add_theme_color(dpg.mvThemeCol_MenuBarBg, colors["widgets"][:-1].append(colors["widgets"][-1]//2))
                dpg.add_theme_color(dpg.mvThemeCol_ScrollbarBg, colors["widgets"][:-1].append(colors["widgets"][-1]//2))
                dpg.add_theme_color(dpg.mvThemeCol_Button, colors["widgets"][:-1].append(colors["widgets"][-1]//2))
                dpg.add_theme_color(dpg.mvThemeCol_Tab, colors["widgets"][:-1].append(colors["widgets"][-1]//2))
                dpg.add_theme_color(dpg.mvThemeCol_Header, colors["widgets"][:-1].append(colors["widgets"][-1]//2))
        dpg.bind_theme(theme)
        return {
            "main": dpg.get_item_children(main, 1),
            "additional": dpg.get_item_children(additional, 1),
            "borders": dpg.get_item_children(borders, 1),
            "text": dpg.get_item_children(text, 1),
            "widgets": dpg.get_item_children(widgets, 1),
            "text_disabled": dpg.get_item_children(text_disabled, 1),
            "widgets_disabled": dpg.get_item_children(widgets_disabled, 1)
        }
