import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
import plotly.graph_objs as go
from typing import Literal
import json
from matplotlib.tri import Triangulation


class TerrainInterpolation:
    def __init__(self, lon, lat, elev, x_shape, y_shape):
        self.lon = np.asarray(lon).flatten()
        self.lat = np.asarray(lat).flatten()
        self.elev = np.asarray(elev).flatten()
        self.x_shape = x_shape
        self.y_shape = y_shape

    def _scaler(self):
        lat0 = np.mean(self.lat)
        lon0 = np.mean(self.lon)

        lat_scale = 111_320
        lon_scale = 111_320 * np.cos(np.radians(lat0))

        x = (self.lon - lon0) * lon_scale
        y = (self.lat - lat0) * lat_scale
        z = self.elev

        range_x = np.ptp(x)
        range_y = np.ptp(y)
        range_z = np.ptp(z)

        max_range = max(range_x, range_y, range_z)
        aspect_x = range_x / max_range
        aspect_y = range_y / max_range
        aspect_z = range_z / max_range

        return x, y, z, aspect_x, aspect_y, aspect_z

    def scatter_plot(self, is_scaled=0):
        if is_scaled:
            x, y, z, aspect_x, aspect_y, aspect_z = self._scaler()
        else:
            x, y, z = self.lon, self.lat, self.elev
            aspect_x = aspect_y = aspect_z = 1

        fig = go.Figure(data=[go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode='markers',
            marker=dict(
                size=5,
                color=z,
                colorscale='Viridis',
                colorbar=dict(title='Высота'),
                opacity=1
            ))])

        fig.update_layout(
            title='Интерактивный 3D-график высоты',
            scene=dict(
                xaxis_title='Долгота',
                yaxis_title='Широта',
                zaxis_title='Высота',
                aspectmode='manual',
                aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z)),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        fig.show()

    def bicubic_interpolation(self, is_scaled=0):
        if is_scaled:
            x, y, z, aspect_x, aspect_y, aspect_z = self._scaler()
        else:
            x, y, z = self.lon, self.lat, self.elev
            aspect_x = aspect_y = aspect_z = 1

        xi = np.linspace(min(x), max(x), 100)  # поменять плотность точек
        yi = np.linspace(min(y), max(y), 100)
        xx, yy = np.meshgrid(xi, yi)
        zz = griddata((x, y), z, (xx, yy), method='cubic')

        fig = go.Figure(data=[go.Surface(z=zz, x=xi, y=yi, colorscale='Plasma', opacity=1)])

        fig.update_layout(
            title='Bicubic Interpolation Surface',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
                aspectmode='manual',
                aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z)),
            height=800,
            width=800
        )

        fig.show()

    def linear_interpolation(self, is_scaled=0):
        if is_scaled:
            x, y, z, aspect_x, aspect_y, aspect_z = self._scaler()
        else:
            x, y, z = self.lon, self.lat, self.elev
            aspect_x = aspect_y = aspect_z = 1

        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xx, yy = np.meshgrid(xi, yi)
        zz = griddata((x, y), z, (xx, yy), method='linear')

        fig = go.Figure(data=[go.Surface(z=zz, x=xi, y=yi)])

        fig.update_layout(
            title='Linear Interpolation Surface',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
                aspectmode='manual',
                aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z)),
            height=800,
            width=800
        )

        fig.show()

    def nearest_interpolation(self, is_scaled=0):
        if is_scaled:
            x, y, z, aspect_x, aspect_y, aspect_z = self._scaler()
        else:
            x, y, z = self.lon, self.lat, self.elev
            aspect_x = aspect_y = aspect_z = 1

        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xx, yy = np.meshgrid(xi, yi)
        zz = griddata((x, y), z, (xx, yy), method='nearest')

        fig = go.Figure(data=[go.Surface(z=zz, x=xi, y=yi)])

        fig.update_layout(
            title='Nearest Interpolation Surface',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='z',
                aspectmode='manual',
                aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z)),
            height=800,
            width=800
        )

        fig.show()

    def delaunay_interpolation(self, is_scaled=0):
        if is_scaled:
            x, y, z, aspect_x, aspect_y, aspect_z = self._scaler()
        else:
            x, y, z = self.lon, self.lat, self.elev
            aspect_x = aspect_y = aspect_z = 1

        points2D = np.vstack((x, y)).T
        tri = Delaunay(points2D)

        fig = go.Figure(data=[go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=tri.simplices[:, 0],
            j=tri.simplices[:, 1],
            k=tri.simplices[:, 2],
            intensity=z,
            colorscale='Viridis',
            colorbar=dict(title='Высота (м)'),
            opacity=1,
            showscale=True
        )])

        fig.update_layout(
            title='3D-поверхность с пропорциональными осями (в метрах)',
            scene=dict(
                xaxis_title='x',
                yaxis_title='y',
                zaxis_title='elev',
                aspectmode='manual',
                aspectratio=dict(x=aspect_x, y=aspect_y, z=aspect_z)
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )

        fig.show()

    def plot_interpolation(self, is_scaled=0, plot_type: Literal['linear', 'nearest', 'cubic'] = 'linear'):
        if is_scaled:
            x, y, z, _, _, _ = self._scaler()
        else:
            x, y, z = self.lon, self.lat, self.elev
        xi = np.linspace(min(x), max(x), 1000)
        yi = np.linspace(min(y), max(y), 1000)
        xx, yy = np.meshgrid(xi, yi)
        zz = griddata((x, y), z, (xx, yy), method=plot_type)

        plt.figure(figsize=(10, 10))
        plt.title(plot_type)
        plt.imshow(zz, extent=(min(x), max(x), min(y), max(y)), origin='lower')
        plt.colorbar()
        plt.show()

    def terrain(self, is_scaled=0):
        meas = min(self.y_shape, self.x_shape) ** 2
        if meas < 500:
            self.bicubic_interpolation(is_scaled=is_scaled)
        else:
            self.delaunay_interpolation(is_scaled=is_scaled)


    def plot_interpolation_with_delaunay(self, is_scaled=0):
        if is_scaled:
            x, y, z, _, _, _ = self._scaler()
        else:
            x, y, z = self.lon, self.lat, self.elev

        # Создание триангуляции Делоне
        triang = Triangulation(x, y)

        plt.figure(figsize=(10, 10))
        # Отображение триангуляции
        plt.tricontourf(triang, z, levels=14, cmap='viridis')
        plt.colorbar()

        plt.show()
# X = np.arange(-15, 15, 1)
# Y = np.arange(-10, 10, 1)
# xx, yy = np.meshgrid(X, Y)
# Z = np.sin(xx) + np.cos(yy)

with open('data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
lats = [point['lat'] for point in data]
lons = [point['lon'] for point in data]
elevs = [point['elevation'] for point in data]
# print(len(set(lats)))
# print()

# plotter = TerrainInterpolation(xx, yy, Z, len(X), len(Y))
plotter = TerrainInterpolation(lons, lats, elevs, len(set(lons)), len(set(lats)))
# plotter.scatter_plot()
# plotter.bicubic_interpolation()
# plotter.linear_interpolation(is_scaled=0)
# plotter.nearest_interpolation(is_scaled=1)
# plotter.plot_interpolation(plot_type='linear', is_scaled=0)
# plotter.plot_interpolation(plot_type='cubic', is_scaled=1)
# plotter.plot_interpolation(plot_type='nearest', is_scaled=1)
plotter.plot_interpolation_with_delaunay(is_scaled=1)
# plotter.terrain(is_scaled=1)
# import time
#
# # Функция для измерения времени выполнения
# def measure_time(func, *args, **kwargs):
#     start_time = time.time()
#     func(*args, **kwargs)
#     end_time = time.time()
#     return end_time - start_time
#
# # Измерение времени выполнения каждой функции
# bicubic_time = measure_time(plotter.bicubic_interpolation)
# linear_time = measure_time(plotter.linear_interpolation, is_scaled=0)
# nearest_time = measure_time(plotter.nearest_interpolation, is_scaled=1)
# linear_plot_time = measure_time(plotter.plot_interpolation, plot_type='linear', is_scaled=0)
# cubic_plot_time = measure_time(plotter.plot_interpolation, plot_type='cubic', is_scaled=1)
# nearest_plot_time = measure_time(plotter.plot_interpolation, plot_type='nearest', is_scaled=1)
#
# # Вывод результатов
# print(f"Время bicubic_interpolation: {bicubic_time:.6f} секунд")
# print(f"Время linear_interpolation: {linear_time:.6f} секунд")
# print(f"Время nearest_interpolation: {nearest_time:.6f} секунд")
# print(f"Время plot_interpolation (linear): {linear_plot_time:.6f} секунд")
# print(f"Время plot_interpolation (cubic): {cubic_plot_time:.6f} секунд")
# print(f"Время plot_interpolation (nearest): {nearest_plot_time:.6f} секунд")
