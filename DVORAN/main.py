from PIL import Image
import math


def get_data():
    with open('data.txt') as f:
        data = f.readlines()
    print(len(data))

    # data = data[600: 800]

    for i in range(len(data)):
        data[i] = data[i].split()
        data[i].pop(-1)
        data[i] = [float(x) for x in data[i]]
    return data

def varu(input_data):
    data = input_data
    for row in range(0, len(data), 100):
        mid_row = data[row + 49]  # выбираем серединную строку
        for i in range(row, row + 100):
            for j in range(len(data[0])):
                if mid_row[j] == 0:
                    data[i][j] = 0
                else:
                    data[i][j] = data[i][j] / mid_row[j]
    return data


def smooth(input_data, r):
    data = input_data
    for i in range(len(data)):
        for j in range(len(data[0])):
            # тут мы берём точку. Теперь наша задача заключается в вычислении среднего значения по квадрату с стороной квадрата 2r + 1#
            # value = [[1 for j in range(2 * r + 1)] for i in range(2 * r + 1)]
            value = 0
            for y in range(- r, r + 1):
                for x in range(- r, + r + 1):
                    if i + y < 0 or j + x < 0 or i + y > len(data) - 1 or x + j > len(data[0]) - 1: value +=1 / ((2 * r + 1) ** 2)
                    else:value += data[i + y][j + x] / ((2 * r + 1) ** 2)
            data[i][j] = value
    return data


def linear_upscale(input_data, coef):
    data = input_data

    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] < 1000:
                data[i][j] *= j * coef * data[i][j]
            data[i][j] = int(255 * data[i][j] / max_value)
    return data


def log_filtration(input_data):
    data = input_data

    for i in range(len(data)):
        for j in range(len(data[0])):
            if data[i][j] > 0:
                data[i][j] = int(math.log(data[i][j] * 0.5, max_value).real * 255)
            else:
                data[i][j] = 0  # or any other appropriate value for non-positive numbers
    return data


def make_picture(data):
    img = Image.new('RGB', (8680, len(data)), 'black')

    for i in range(len(data)):
        for j in range(len(data[0])):
            value = int(data[i][j])
            img.putpixel((j, i), (value, value, value))
    img.show()
    img.save('image.png')


if __name__ == '__main__':
    data = get_data()
    max_value = max(max(row) for row in data)
    # data = varu(data)
    print("varu is done")
    data = linear_upscale(data, 0.00001)
    print("linear upscale is done")
    data = log_filtration(data)
    print("log filtration is done")
    make_picture(data)
