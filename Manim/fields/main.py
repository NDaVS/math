from manim import *
import numpy as np

class PhasePortrait3D(ThreeDScene):
    def construct(self):
        # Определяем функцию для потока в 3D
        def func(pos):
            x, y, z = pos
            return np.array([-y, -x, -x*y])  # Определяем поток

        # Создаем линии потока
        stream_lines = StreamLines(func, stroke_width=2, max_anchors_per_line=20)

        # Добавляем линии потока на сцену
        self.add(stream_lines)

        # Настраиваем параметры анимации потоков
        flow_speed = 1.5
        total_time = stream_lines.virtual_time / flow_speed + 2  # Общее время анимации

        # Запускаем анимацию линий потока с зацикливанием
        stream_lines.start_animation(warm_up=False, flow_speed=flow_speed, cycle_animation=True)

        # Настройка камеры для 3D
        self.set_camera_orientation(phi=70 * DEGREES, theta=30 * DEGREES)

        # Настраиваем плавное вращение камеры с полным оборотом
        slow_factor = 3  # Уменьшает скорость вращения
        self.begin_ambient_camera_rotation(rate=(TAU / total_time) / slow_factor)

        # Ожидаем завершения текущей итерации анимации
        self.wait(total_time)

        # Завершаем анимацию
        self.stop_ambient_camera_rotation()

        # Ожидаем и повторяем (если нужно)
        self.wait(1)
        self.play(*[FadeOut(mob) for mob in self.mobjects])
