# -*- coding: utf-8 -*-
"""Модуль расчета параллакса и модульного микширования 3D-анаглифов с автокропом."""

import cv2
import numpy as np


class AnaglyphProcessor(object):
    """Класс, содержащий изолированные математические методы

    формирования 3D-анаглифов и стереопар.
    """

    def __init__(self, gdepth_processor):
        # Используем инжектированный процессор для доступа к кадрам и картам
        self.dp = gdepth_processor

        # Словарь метаданных для HUD-интерфейса (ключ — имя метода)
        self.method_details = {
            "wimmer_pure": "Wimmer Pure Color Anaglyph (StereoPhoto Maker)",
            "wimmer_balanced": "Wimmer Balanced Color Anaglyph II (SPM)",
            "vu_tran_least_squares": "Vu Tran Least Squares Approximation (2005)",
            "wimmer_optimized_matrix": "Peter Wimmer Optimized Matrix (SPM)",
            "ducos_classic": "Ducos du Hauron Classic True Anaglyph (1891)",
            "sgi_gray": "Silicon Graphics ITU-R Rec.601 Gray Anaglyph",
            "wimmer_half_color": "Wimmer Mono Half-color Anaglyph (SPM)",
            "wimmer_green_optimized": "Wimmer Green-Channel Optimized Anaglyph (SPM)",
            "dubois_projection": "Eric Dubois Projection Method (2001)",
            "thor_olson_optimized": "Thor Olson True Optimized Method (2009)",
            "mcallister_midpoint": "McAllister Midpoint Optimization",
            "rossman_luminance": "Rossman Luminance-Compensation Method",
            "thor_olson_lcd": "LCD Display Optimized Anaglyph (Thor Olson)",
            "green_magenta_pure": "Green-Magenta Empirical Pure Anaglyph",
            "green_magenta_dubois": "Green-Magenta Dubois Linear Method (2001)",
        }

    def _get_linear_channels(self, left_eye, right_eye):
        """Вспомогательный метод дегаммирования sRGB -> Linear Float32."""
        left_linear = np.power(left_eye.astype(np.float32) / 255.0, 2.2)
        right_linear = np.power(right_eye.astype(np.float32) / 255.0, 2.2)
        return left_linear, right_linear

    def _to_srgb_8bit(self, anaglyph_linear):
        """Вспомогательный метод регаммирования Linear -> sRGB 8-bit."""
        anaglyph_linear = np.clip(anaglyph_linear, 0.0, 1.0)
        return (np.power(anaglyph_linear, 1.0 / 2.2) * 255.0).astype(np.uint8)

    # === ЭМПИРИЧЕСКИЕ И КЛАССИЧЕСКИЕ МЕТОДЫ (sRGB) ===

    def wimmer_pure(self, left_eye, right_eye):
        mat_L = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        mat_R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        return np.clip(cv2.add(cv2.transform(left_eye, mat_L), cv2.transform(right_eye, mat_R)), 0, 255).astype(np.uint8)

    def wimmer_balanced(self, left_eye, right_eye):
        mat_L = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.2, 0.9]])
        mat_R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        return np.clip(cv2.add(cv2.transform(left_eye, mat_L), cv2.transform(right_eye, mat_R)), 0, 255).astype(np.uint8)

    def vu_tran_least_squares(self, left_eye, right_eye):
        mat_L = np.array([[0.0128, -0.0615, -0.0547], [-0.0257, -0.0484, -0.0458], [0.1669, 0.4710, 0.4154]])
        mat_R = np.array([[1.2971, -0.1287, -0.0651], [0.0111, 0.7333, 0.3756], [-0.0060, -0.0364, -0.0109]])
        return np.clip(cv2.add(cv2.transform(left_eye, mat_L), cv2.transform(right_eye, mat_R)), 0, 255).astype(np.uint8)

    def wimmer_optimized_matrix(self, left_eye, right_eye):
        mat_L = np.array([[-0.01323, -0.04987, -0.01311], [-0.03816, -0.09151, -0.96979], [0.15569, 0.44181, 0.40250]])
        mat_R = np.array([[0.82200, -0.07570, -0.02914], [-0.01552, 0.61710, 0.31835], [-0.00104, -0.06100, -0.04834]])
        return np.clip(cv2.add(cv2.transform(left_eye, mat_L), cv2.transform(right_eye, mat_R)), 0, 255).astype(np.uint8)

    def ducos_classic(self, left_eye, right_eye):
        mat_L = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.114, 0.587, 0.299]])
        mat_R = np.array([[0.114, 0.587, 0.299], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        return np.clip(cv2.add(cv2.transform(left_eye, mat_L), cv2.transform(right_eye, mat_R)), 0, 255).astype(np.uint8)

    def sgi_gray(self, left_eye, right_eye):
        mat_L = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.114, 0.587, 0.299]])
        mat_R = np.array([[0.114, 0.587, 0.299], [0.114, 0.587, 0.299], [0.0, 0.0, 0.0]])
        return np.clip(cv2.add(cv2.transform(left_eye, mat_L), cv2.transform(right_eye, mat_R)), 0, 255).astype(np.uint8)

    def wimmer_half_color(self, left_eye, right_eye):
        mat_L = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.114, 0.587, 0.299]])
        mat_R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        return np.clip(cv2.add(cv2.transform(left_eye, mat_L), cv2.transform(right_eye, mat_R)), 0, 255).astype(np.uint8)

    def wimmer_green_optimized(self, left_eye, right_eye):
        mat_L = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.7, 0.3]])
        mat_R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        return np.clip(cv2.add(cv2.transform(left_eye, mat_L), cv2.transform(right_eye, mat_R)), 0, 255).astype(np.uint8)

    def green_magenta_pure(self, left_eye, right_eye):
        """Зелено-пурпурный классический метод для sRGB."""
        mat_L = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        mat_R = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        return np.clip(cv2.add(cv2.transform(left_eye, mat_L), cv2.transform(right_eye, mat_R)), 0, 255).astype(np.uint8)
    # === НАУЧНЫЕ И ПРОЕКЦИОННЫЕ МЕТОДЫ (Linear float32) ===

    def dubois_projection(self, left_eye, right_eye):
        mat_L = np.array([[0.0128, -0.0615, -0.0547], [-0.0257, -0.0484, -0.0458], [0.1669, 0.4710, 0.4154]])
        mat_R = np.array([[1.2971, -0.1287, -0.0651], [0.0111, 0.7333, 0.3756], [-0.0060, -0.0364, -0.0109]])
        lL, lR = self._get_linear_channels(left_eye, right_eye)
        return self._to_srgb_8bit(cv2.add(cv2.transform(lL, mat_L), cv2.transform(lR, mat_R)))

    def thor_olson_optimized(self, left_eye, right_eye):
        mat_L = np.array([[-0.00019, -0.00984, -0.03676], [-0.00035, -0.01469, -0.05501], [0.00628, 0.27112, 1.01492]])
        mat_R = np.array([[1.00019, 0.00984, 0.03676], [0.00035, 1.01469, 0.05501], [-0.00628, -0.27112, -0.01492]])
        lL, lR = self._get_linear_channels(left_eye, right_eye)
        return self._to_srgb_8bit(cv2.add(cv2.transform(lL, mat_L), cv2.transform(lR, mat_R)))

    def mcallister_midpoint(self, left_eye, right_eye):
        mat_L = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.114, 0.587, 0.299]])
        mat_R = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        lL, lR = self._get_linear_channels(left_eye, right_eye)
        return self._to_srgb_8bit(cv2.add(cv2.transform(lL, mat_L), cv2.transform(lR, mat_R)))

    def rossman_luminance(self, left_eye, right_eye):
        mat_L = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.45, 0.85]])
        mat_R = np.array([[0.9, 0.0, 0.0], [0.0, 0.95, 0.0], [0.0, 0.0, 0.0]])
        lL, lR = self._get_linear_channels(left_eye, right_eye)
        return self._to_srgb_8bit(cv2.add(cv2.transform(lL, mat_L), cv2.transform(lR, mat_R)))

    def thor_olson_lcd(self, left_eye, right_eye):
        mat_L = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.015, 0.325, 0.725]])
        mat_R = np.array([[1.015, -0.015, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        lL, lR = self._get_linear_channels(left_eye, right_eye)
        return self._to_srgb_8bit(cv2.add(cv2.transform(lL, mat_L), cv2.transform(lR, mat_R)))

    def green_magenta_dubois(self, left_eye, right_eye):
        """Зелено-пурпурный проекционный метод Дюбуа в линейном float32."""
        mat_L = np.array([[-0.062, -0.046, -0.013], [0.284, 0.641, 0.092], [-0.015, -0.021, -0.005]])
        mat_R = np.array([[0.523, 0.384, 0.082], [-0.042, -0.081, -0.015], [0.021, 0.141, 0.812]])
        lL, lR = self._get_linear_channels(left_eye, right_eye)
        return self._to_srgb_8bit(cv2.add(cv2.transform(lL, mat_L), cv2.transform(lR, mat_R)))

    # === ГЛАВНЫЙ КОНВЕЙЕР РЕНДЕРИНГА С АВТОРЕЗАНИЕМ ОРЕОЛОВ ===

    def render_view_mode(self, state_obj):
        """Рендерит кадр с математическим отсечением краевых дефектов."""
        if state_obj.view_mode == "primary" or not self.dp.has_depth:
            return self.dp.primary_img.copy()

        image = self.dp.primary_img
        img_h, img_w, _ = image.shape

        if state_obj.use_microrelief:
            depth_map = self.dp.get_detailed_depth(state_obj.blur, state_obj.opacity)
        else:
            depth_map = self.dp.raw_depth.copy()

        if depth_map.shape[:2] != (img_h, img_w):
            depth_map = cv2.resize(depth_map, (img_w, img_h))

        if state_obj.view_mode == "depthmap":
            return cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)

        # ИСПРАВЛЕНО: Сначала объявляем базовый нормализованный массив
        depth_normalized = depth_map.astype(float) / 255.0

        # Теперь безопасно применяем логарифмический сдвиг планов
        if state_obj.depth_interpretation == "logarithmic":
            depth_normalized = (np.exp(depth_normalized * 3.0) - 1.0) / (np.exp(3.0) - 1.0)
            depth_normalized = np.clip(depth_normalized, 0.0, 1.0)

        conv_bias = state_obj.convergence / 100.0

        x_indices, y_indices = np.meshgrid(np.arange(img_w), np.arange(img_h))

        # === ДИНАМИЧЕСКИЙ БЛОК FACEBOOK 3D LOOK ===
        if state_obj.fb_3d_mode:
            # В режиме Facebook 3D мы используем ОДИНАКОВЫЙ сдвиг для обоих глаз (монокулярный параллакс)
            # Величина сдвига зависит от положения мыши (mouse_dx, mouse_dy), disparity и глубины пикселя
            # Объекты в плоскости схождения (depth_normalized - 0.5 - conv_bias == 0) остаются неподвижными
            #depth_factor = (depth_normalized - 0.5 - conv_bias) * (state_obj.disparity)
            depth_factor = depth_normalized * state_obj.disparity
            
            shift_x = depth_factor * state_obj.mouse_dx
            shift_y = depth_factor * state_obj.mouse_dy

            map_x_left = (x_indices + shift_x).astype(np.float32)
            map_y = (y_indices + shift_y).astype(np.float32)
            left_eye = cv2.remap(image, map_x_left, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            return left_eye

        # Стандартный бинокулярный горизонтальный параллакс для стереопар и анаглифов
        #shift_val = (depth_normalized - 0.5 - conv_bias) * (state_obj.disparity / 2.0)
        shift_val = depth_normalized * (state_obj.disparity / 2.0)
        map_x_left = (x_indices + shift_val).astype(np.float32)
        map_x_right = (x_indices - shift_val).astype(np.float32)
        map_y = y_indices.astype(np.float32)

        # Пересчет ремаппинга с использованием константного заполнения границ кадра
        left_eye = cv2.remap(image, map_x_left, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        right_eye = cv2.remap(image, map_x_right, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

        # Алгоритм автоматического кропа мертвых зон (выполняется только в обычном 3D режиме)
        max_shift = int(np.ceil(max(abs(state_obj.disparity / 2.0 + (conv_bias * (state_obj.disparity / 2.0))), 2)))
        if max_shift < img_w // 4:
            left_eye = left_eye[:, max_shift : img_w - max_shift]
            right_eye = right_eye[:, max_shift : img_w - max_shift]
            left_eye = cv2.resize(left_eye, (img_w, img_h), interpolation=cv2.INTER_LINEAR)
            right_eye = cv2.resize(right_eye, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        if state_obj.view_mode == "stereopair":
            return np.hstack((left_eye, right_eye))

        elif state_obj.view_mode == "anaglyph":
            mixer_func = getattr(self, state_obj.method_name, None)
            if mixer_func is not None:
                anaglyph_frame = mixer_func(left_eye, right_eye)
                
                # === АППАРАТНОЕ ПОДВАЛЕНИЕ ХРОМАТИЧЕСКИХ ПРИЗРАКОВ (GHOST REDUCTION) ===
                if state_obj.anaglyph_ghost_reduction:
                    # Разделяем анаглиф на физические каналы BGR
                    b, g, r = cv2.split(anaglyph_frame)
                    
                    # Математическая компенсация: утечка красного в правый глаз и сине-зеленого в левый
                    # Приглушаем экстремальные выбросы яркости на стыках за счет перекрестной фильтрации
                    r_compensated = cv2.subtract(r, cv2.scaleAdd(g, 0.12, cv2.scaleAdd(b, 0.08, np.zeros_like(b))))
                    g_compensated = cv2.addWeighted(g, 0.95, r, 0.05, 0)
                    b_compensated = cv2.addWeighted(b, 0.92, r, 0.08, 0)
                    
                    anaglyph_frame = cv2.merge([b_compensated, g_compensated, r_compensated])
                
                return anaglyph_frame
            else:
                raise AttributeError(f"Метод микширования {state_obj.method_name} не найден.")

        return image

