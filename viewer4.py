# -*- coding: utf-8 -*-
"""Главный интерфейсный модуль интерактивного 3D-просмотрщика."""

import argparse
import sys
import os
import cv2
import numpy as np

from anaglyph_processor import AnaglyphProcessor
from config import ViewerState
from gdepth_processor import GDepthProcessor
from keyboard_controller import KeyboardController
# Импортируем вынесенный класс графического рендерера
from visual_renderer import VisualRenderer

# Создаем центральный валидируемый Pydantic стейт
state = ViewerState()


def parse_cli_args():
    """Вынесенная логика инициализации и валидации аргументов CLI."""
    parser = argparse.ArgumentParser(description="ООП-просмотрщик Dynamic Depth.")
    parser.add_argument("file", help="Путь к файлу изображения JPEG")
    parser.add_argument("--mode", "-mode", choices=["anaglyph", "stereopair", "depthmap", "primary"], default="anaglyph")
    parser.add_argument("--method", "-method", default="thor_olson_optimized")
    parser.add_argument("--disparity", "-disp", type=int, default=16)
    parser.add_argument("--blur", "-blur", type=int, default=21)
    parser.add_argument("--opacity", "-op", type=float, default=0.5)
    parser.add_argument("--microrelief", "-mr", choices=["on", "off"], default="on")
    parser.add_argument("--geom", "-geom", choices=["linear", "logarithmic"], default="linear")
    
    args = parser.parse_args()
    try:
        state.view_mode = args.mode
        state.method_name = args.method
        state.disparity = args.disparity
        state.blur = args.blur
        state.opacity = args.opacity
        state.use_microrelief = (args.microrelief == "on")
        state.depth_interpretation = args.geom
    except Exception as err:
        print(f"[-] Ошибка Pydantic-валидации CLI: {err}", file=sys.stderr)
        sys.exit(1)
    return args


def scan_directory(target_file):
    """Вынесенная логика сканирования каталога и поиска индекса файла."""
    file_path = os.path.abspath(target_file)
    folder = os.path.dirname(file_path)
    valid_ext = ('.jpg', '.jpeg', '.png')
    
    try:
        files_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_ext)]
        files_list.sort(key=lambda x: x.lower())
    except Exception:
        files_list = [file_path]

    idx = files_list.index(file_path) if file_path in files_list else 0
    if file_path not in files_list:
        files_list.append(file_path)
        files_list.sort(key=lambda x: x.lower())
        idx = files_list.index(file_path)
    return files_list, idx


def main():
    """Точка входа в приложение. Дирижирует модулями."""
    # 1. Забираем аргументы командной строки
    args = parse_cli_args()

    # 2. Сканируем папку и определяем индекс стартового файла
    all_files, file_index = scan_directory(args.file)

    try:
        # 3. Инициализируем процессор глубин и передаем стартовый флаг в state
        dp = GDepthProcessor(all_files[file_index])
        state.has_depth = dp.has_depth
        ap = AnaglyphProcessor(dp)
        
        # 4. Инициализируем визуальный рендерер HUD и холста
        renderer = VisualRenderer(state, ap)
        
        # 5. Создаем окно и переводим в полноэкранный режим
        window_name = "Dynamic Depth Interactive Viewer"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Выводим пустой холст и ждем инициализации дескриптора окна в WinAPI
        cv2.imshow(window_name, np.zeros((100, 100, 3), dtype=np.uint8))
        cv2.waitKey(50)
        
        # Замеряем геометрию экрана строго внутри main после waitKey
        rect = cv2.getWindowImageRect(window_name)
        if rect is not None and len(rect) >= 4:
            win_x, win_y, win_w, win_h = rect
            if win_w > 0 and win_h > 0:
                state.screen_w, state.screen_h = win_w, win_h

        img_h, img_w, _ = dp.primary_img.shape
        state.scale = renderer.get_fit_to_screen_scale(img_w, img_h)

        # 6. Инициализируем контроллер ввода, передавая рендерер
        controller = KeyboardController(all_files, file_index, window_name, ap, state, renderer)

        print("[+] Просмотрщик запущен. Полная декомпозиция архитектуры SOLID выполнена.")
        image_cached = None

        # 7. Главный цикл отрисовки кадра
        while True:
            if state.need_update or image_cached is None:
                image_cached = controller.update_processors_and_render()
                state.need_update = False

            # Вызов метода рендеринга из вынесенного класса renderer
            display_frame = renderer.render_canvas(
                image_cached, 
                os.path.basename(controller.get_current_file()), 
                len(controller.files), controller.file_index
            )
            cv2.imshow(window_name, display_frame)

            # Опрашиваем клавиатуру
            if not controller.handle_input(image_cached):
                break

        cv2.destroyAllWindows()

    except Exception as e:
        print(f"[-] Критическая ошибка в работе программы: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
