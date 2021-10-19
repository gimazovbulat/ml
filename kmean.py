import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
 Представление кластера
 points_x - координаты x
 points_y - координаты y
 centroids_x - координаты центроидов x
 centroids_y - координаты центроидов y
 """
class Cluster:
    def __init__(self, centroids_x, centroids_y):
        self.points_x = []
        self.points_y = []
        self.centroids_x = centroids_x
        self.centroids_y = centroids_y

    def add_point(self, x, y):
        self.points_x.append(x)
        self.points_y.append(y)

    def add_points(self, x, y):
        self.points_x.extend(x)
        self.points_y.extend(y)

    def mean(self):
        self.centroids_x = sum(self.points_x) / len(self.points_x)
        self.centroids_y = sum(self.points_y) / len(self.points_y)

    def clear_points(self):
        self.points_x = []
        self.points_y = []

"""
 Рисует картинку
 """
def show_picture(clusters):
    color = ['r', 'b', 'g', 'c', 'm', 'k', 'y', 'lime', 'teal', 'navy', 'plum', 'pink', 'cyan', 'slategray', 'peru', 'brown']
    i = 0
    for cl in clusters:
        plt.scatter(cl.centroids_x, cl.centroids_y, color=color[i], marker='x')
        plt.scatter(cl.points_x, cl.points_y, color=color[i])
        i += 1
    plt.show()


def map_to_array(clusters):
    x_c = []
    y_c = []
    for cl in clusters:
        x_c.append(cl.centroids_x)
        y_c.append(cl.centroids_y)
    return [x_c, y_c]

"""
 Расстояние от точки до точки
 """
def dist(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

"""
 Генерит рандомные точки в csv
 """
def random_points(n, filename):
    x = np.random.randint(0, 100, n)
    y = np.random.randint(0, 100, n)
    pd.DataFrame([x, y]).to_csv(filename)


"""
 Пишет точки в файл
 """
def generate_points_file(points_count, filename):
    x = np.random.randint(0, 100, points_count)
    y = np.random.randint(0, 100, points_count)
    with open(filename, 'w') as file:
        file.write('x,y\n')
        for item_x, item_y in zip(x, y):
            file.write(f'{item_x},{item_y}\n')

"""
 Читает csv
 """
def read_csv(filename):
    return pd.read_csv(filename)

"""
 Расставляет центроиды вокруг точек (дефолтное положение центроидов)
 """
def centroids(points, k):
    x_centr = points['x'].mean()
    y_centr = points['y'].mean()
    R = dist(x_centr, y_centr, points['x'][0], points['y'][0])
    for i in range(len(points)):
        R = max(R, dist(x_centr, y_centr, points['x'][i], points['y'][i]))
    x_c, y_c = [], []
    for i in range(k):
        x_c.append(x_centr + R * np.cos(2 * np.pi * i / k))
        y_c.append(y_centr + R * np.sin(2 * np.pi * i / k))
    return [x_c, y_c]


"""
 Находит ближайшие центроиды к точке
 """
def nearest_centroid(points, centroids):
    clusters = [Cluster(x_c, y_c) for x_c, y_c in zip(centroids[0], centroids[1])]
    indx = -1
    for x, y in zip(points['x'], points['y']):
        r = float('inf')
        for i, cl in enumerate(clusters):
            if r > dist(x, y, cl.centroids_x, cl.centroids_y):
                r = dist(x, y, cl.centroids_x, cl.centroids_y)
                indx = i
        if indx >= 0:
            clusters[indx].add_point(x, y)
    return clusters

"""
 Считает центры кластеров
 """
def recalculate_centroid(clusters):
    new_clusters = []
    for cl in clusters:
        new_cl = Cluster(cl.centroids_x, cl.centroids_y)
        new_cl.add_points(cl.points_x, cl.points_y)
        new_clusters.append(new_cl)
    for cl in new_clusters:
        if len(cl.points_x) != 0:
            cl.mean()
    return new_clusters

"""
 Сравнивает новые и старые кластеры
 """
def centroid_not_equals(old_cluster, new_cluster):
    if len(new_cluster) <= 0 or len(old_cluster) <= 0:
        return True
    for i in range(0, len(new_cluster) - 1):
        if old_cluster[i].centroids_x != new_cluster[i].centroids_x \
                and old_cluster[i].centroids_y != new_cluster[i].centroids_y:
            return True
    return False


def c_l(clusters):
    r = []
    for cl in clusters:
        for x, y in zip(cl.points_x, cl.points_y):
            r.append(dist(cl.centroids_x, cl.centroids_y, x, y))
    return sum(r)


def d_l(s0, s1, s2):
    if abs(s0 - s1) != 0:
        return abs(s1 - s2) / abs(s0 - s1)
    else:
        return float('inf')


"""
 Сам алгоритм k_means
 """
def k_means(points, k, is_show):
    cntds = centroids(points, k)
    clusters = nearest_centroid(points, cntds)
    if (is_show):
        show_picture(clusters)
    new_clusters = []
    old_clusters = []
    while centroid_not_equals(old_clusters, new_clusters):
        old_clusters = nearest_centroid(points, cntds)
        new_clusters = recalculate_centroid(old_clusters)
        cntds = map_to_array(new_clusters)
        if (is_show):
            show_picture(new_clusters)
    return new_clusters


if __name__ == "__main__":
    n = 100  # кол-во тчк
    k_max = 15  # максимальное кол-во кластеров
    k = 3  # кол-во кластеров
    c_ls = []
    d_ls = []
    k_s = []
    filename = 'dataset.csv'
    # generate_points_file(n, filename)
    points = read_csv(filename)
    for i in range(k, k_max):
        clusters = k_means(points, i, False)
        c_ls.append(c_l(clusters))
        k_s.append(i)

    dl = float('inf')
    k = 0
    for i in range(0, len(c_ls) - 2):
        d_ls.append(d_l(c_ls[i], c_ls[i+1], c_ls[i+2]))
        if dl > d_l(c_ls[i], c_ls[i+1], c_ls[i+2]):
            dl = d_l(c_ls[i], c_ls[i+1], c_ls[i+2])
            k = k_s[i+1]
    print(k)
    k_means(points, k, True)