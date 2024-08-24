import dearpygui.dearpygui as dpg
import themes

from program import Program


def main():
    dpg.create_context()
    clb = Program()
    dpg.add_texture_registry(tag='textureRegistry')
    with dpg.font_registry():
        with dpg.font("./fonts/Pixel-Digivolve-Cyrillic-font.otf", 14, pixel_snapH=True) as default_font:
            dpg.add_font_range_hint(dpg.mvFontRangeHint_Cyrillic)
    dpg.bind_font(default_font)
    themes.load_themes()
    with dpg.window() as MainWindow:
        with dpg.menu_bar(tag='menuBar'):
            with dpg.menu(label="Изображения"):
                dpg.add_menu_item(label="Открыть изображения", callback=clb.open_file_dialog)
                dpg.add_menu_item(label="Открыть изображения в папке", callback=clb.open_folder_dialog)
            with dpg.menu(label="Распознать"):
                dpg.add_menu_item(label="на выбранном изображении", callback=clb.predict_image)
                dpg.add_menu_item(label="на всех изображениях",
                                  callback=clb.predict_all)
            dpg.add_button(label="Настройки", callback=clb.open_settings)
        with dpg.group(horizontal=True, height=-28):
            with dpg.table(tag="imagesList", height=-1, header_row=False, width=200, scrollX=True,
                           policy=dpg.mvTable_SizingFixedFit):
                dpg.add_table_column()
                dpg.add_table_column()
            with dpg.plot(width=-1, height=-1, equal_aspects=True, no_menus=True):
                dpg.add_plot_axis(dpg.mvXAxis, label="x axis", tag="plot_axis_x")
                dpg.add_plot_axis(dpg.mvYAxis, label="y axis", tag="plot_axis_y")
        with dpg.table(height=15, header_row=False):
            dpg.add_table_column(width_fixed=True)
            dpg.add_table_column(width_fixed=True)
            dpg.add_table_column(width_fixed=True)
            dpg.add_table_column(width_stretch=True)
            with dpg.table_row():
                dpg.add_text('Открыто изображений: 0  |', tag='filesOpenedText')
                dpg.add_text('Обработано изображений: 0  |', tag='filesProcessedText')
                dpg.add_text('Найдено объектов: 0', tag='objectsFoundText')
                dpg.add_button(label='ver. 1.0', width=-1)
                dpg.bind_item_theme(dpg.last_item(), "bottom_panel_text")
    dpg.set_primary_window(window=MainWindow, value=True)
    dpg.create_viewport(
        title='Lost People Search System',
        min_width=500,
        min_height=500,
        small_icon='./icon.ico',
        large_icon='./icon.ico'
    )
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_exit_callback(callback=clb.settings.save)
    dpg.set_frame_callback(frame=3, callback=clb.open_loading_window)
    dpg.start_dearpygui()
    dpg.destroy_context()

if __name__ == '__main__':
    main()
