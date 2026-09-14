# -*- coding: utf-8 -*-
"""Модуль извлечения и детализации карт глубин контейнера Dynamic Depth."""

import xml.etree.ElementTree as ET
import cv2
import numpy as np


class GDepthProcessor(object):
    """Класс для разбора контейнера Dynamic Depth и детализации карт."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.primary_img = None
        self.raw_depth = None
        self.gray_primary = None
        # Флаг наличия 3D-данных (True для Dynamic Depth, False для обычных фото)
        self.has_depth = True 
        self._extract_data()

    def _extract_data(self):
        """Внутренний метод сквозного разбора бинарного файла и XMP."""
        self.primary_img = cv2.imread(self.file_path)
        if self.primary_img is None:
            raise ValueError(f"Не удалось загрузить изображение: {self.file_path}")

        img_h, img_w, _ = self.primary_img.shape
        self.gray_primary = cv2.cvtColor(self.primary_img, cv2.COLOR_BGR2GRAY)

        try:
            with open(self.file_path, "rb") as f:
                file_bytes = f.read()

            xmp_start = file_bytes.find(b"<x:xmpmeta")
            xmp_end = file_bytes.find(b"</x:xmpmeta>") + len(b"</x:xmpmeta>")
            if xmp_start == -1 or xmp_end == -1:
                raise ValueError("XMP теги не найдены.")

            xmp_data = file_bytes[xmp_start:xmp_end].decode("utf-8", errors="ignore")
            root = ET.fromstring(xmp_data)

            depth_length = None
            for elem in root.iter():
                tag_name = elem.tag.split("}")[-1]
                if tag_name == "Length" and elem.text:
                    try:
                        val = int(elem.text.strip())
                        if val > 0:
                            depth_length = val
                            break
                    except ValueError:
                        pass
                for attr_key, attr_val in elem.attrib.items():
                    attr_name = attr_key.split("}")[-1]
                    if attr_name == "Length" and attr_val:
                        try:
                            val = int(attr_val.strip())
                            if val > 0:
                                depth_length = val
                                break
                        except ValueError:
                            pass

            if depth_length is None:
                raise ValueError("Длина карты глубин равна нулю.")

            depth_bytes = file_bytes[-depth_length:]
            depth_array = np.frombuffer(depth_bytes, dtype=np.uint8)
            self.raw_depth = cv2.imdecode(depth_array, cv2.IMREAD_GRAYSCALE)
            
            if self.raw_depth is None:
                raise ValueError("Ошибка декодирования JPEG карты.")

        except Exception:
            # Если это обычное изображение без метаданных — создаем плоскую нулевую карту глубин
            self.has_depth = False
            self.raw_depth = np.zeros((img_h, img_w), dtype=np.uint8)

    def _apply_high_pass_filter(self, blur_size):
        """Получение высокочастотного микрорельефа из яркостного канала."""
        low_pass = cv2.GaussianBlur(self.gray_primary, (blur_size, blur_size), 0)
        return cv2.addWeighted(self.gray_primary, 1.0, low_pass, -1.0, 128)

    def get_detailed_depth(self, blur_size, opacity):
        """Формирует детализированную карту глубин по формуле Overlay."""
        # Если 3D-данных изначально нет, микрорельеф игнорируется, возвращается плоская карта
        if not self.has_depth:
            return self.raw_depth

        hpf_image = self._apply_high_pass_filter(blur_size)
        if hpf_image.shape != self.raw_depth.shape:
            hpf_image = cv2.resize(
                hpf_image,
                (self.raw_depth.shape, self.raw_depth.shape),
                interpolation=cv2.INTER_LANCZOS4,
            )

        a_chan = self.raw_depth.astype(float) / 255.0
        b_chan = hpf_image.astype(float) / 255.0
        overlay = np.where(
            a_chan < 0.5, 2 * a_chan * b_chan, 1 - 2 * (1 - a_chan) * (1 - b_chan)
        )
        final_depth = (1.0 - opacity) * a_chan + opacity * overlay
        return np.clip(final_depth * 255.0, 0, 255).astype(np.uint8)
