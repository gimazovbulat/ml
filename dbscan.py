from enum import Enum
import pygame
import numpy as np
from matplotlib import cm

pygame.init()
screen = pygame.display.set_mode((700, 500))
pygame.display.set_caption('DBSCAN')
font = pygame.font.Font(None, 30)

exit = False
points = []
groups = []
singles = []
centres = []
max_dist = 100
min_neighbours_count = 3
points_count = 0


class Color(Enum):
    BLACK = 'black'
    RED = 'red'
    GREEN = 'green'
    YELLOW = 'yellow'
    WHITE = 'white'


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.color = Color.BLACK.value
        self.distances = {}

    def __eq__(self, other):
        if self.color == other.color and self.x == other.x and self.y == other.y:
            return True
        else:
            return False

    def __hash__(self):
        return hash((self.x, self.y, self.color))

    def __str__(self):
        return f'Point({self.x}, {self.y}, {Color(self.color)})'

    def __repr__(self):
        return f'Point({self.x}, {self.y}, {Color(self.color)})'

    def set_green(self):
        self.color = Color.GREEN.value

    def set_yellow(self):
        self.color = Color.YELLOW.value

    def set_red(self):
        self.color = Color.RED.value

    def draw(self, screen):
        pygame.draw.circle(screen, self.color, (self.x, self.y), 10)
"""
Считает дистанцию от себя до другой точки, кладем в мапу с расстояними до всех точек
"""
    def add_distance(self, point):
        dist = np.sqrt((self.x - point.x) ** 2 + (self.y - point.y) ** 2)
        self.distances[point] = dist
        self.distances = dict(sorted(self.distances.items(), key=lambda x: x[1]))

"""
Добавляем ближайшие точки в соседи
"""
    def get_neighbours(self, dist):
        neighbours = []
        for k, v in self.distances.items():
            if v <= dist:
                neighbours.append(k)

        return neighbours

"""
Сам метод кластеризации
"""
def clusterize():
    for point1 in points:
        for point2 in points:
            if point1 != point2 and point2 not in point1.distances:
                point1.add_distance(point2)

            if point1 != point2 and point1 not in point2.distances:
                point2.add_distance(point1)

"""
Если есть точки
"""
    while len(points) != 0:
        for point in points:
        """
         Если количество соседей точки меньше чем заданное количество соседей
         """
            if len(point.get_neighbours(max_dist)) < min_neighbours_count:
            """
             Тогда добавляем в singles
             """
                singles.append(point)
                points.remove(point)
            else:
            """
             Иначе красим точку в зеленый, добавляем в группу
             """
                points.remove(point)
                point.set_green()
                group = [point]
                """
                 Проходимся по соседям точки
                 """
                for n in point.get_neighbours(max_dist):
                """
                если точка в singles или количество соседей меньше минимального количества
                """
                    if n in singles or len(n.get_neighbours(max_dist)) < min_neighbours_count:
                    """
                     Если точка в singles, просто уберем ее
                     """
                        if n in singles:
                            singles.remove(n)
                        n.set_yellow()
                        group.append(n)
                        if n in points:
                            points.remove(n)
                     """
                      Если количество соседей больше или равно минимальному, то добавим в группу
                      """
                    if len(n.get_neighbours(max_dist)) >= min_neighbours_count:
                        n.set_green()
                        group.append(n)
                        if n in points:
                            points.remove(n)


                groups.append(group)

    for single in singles:
        single.set_red()

"""
 Проходимся по группам, если они пересекаются, тогда объединяем
 """
    for group1 in groups:
        for group2 in groups:
            if group1 != group2:
                c = list(set(group1) & set(group2))
                if len(c) > 0:
                    group1 += group2
                    groups.remove(group2)
"""
Проходимся по группам, добавляем в массив центров x, y, радиус
 """
    for group in groups:
        xs = [point.x for point in group]
        ys = [point.y for point in group]
        centre = [int(sum(xs) / len(group)), int(sum(ys) / len(group))]
        max_centre_dist = 0
        for point in group:
            dist = np.sqrt((point.x - centre[0]) ** 2 + (point.y - centre[1]) ** 2)
            if max_centre_dist < dist:
                max_centre_dist = dist
        centre.append(max_centre_dist + 5)
        centres.append(centre)

    colors = {
        0: 'purple',
        1: 'brown',
        2: 'violet',
        3: 'green',
        4: 'blue',
    }

    for idx, group in enumerate(groups):
        for point in group:
            point.color = colors[idx]

    print('Done!')


while not exit:
    pygame.time.delay(100)
    screen.fill((255, 255, 255))

    if len(points) == points_count:
        for point in points:
            point.draw(screen)
    else:
        for center in centres:
            pygame.draw.circle(screen, Color.BLACK.value, (center[0], center[1]), center[2])
            pygame.draw.circle(screen, Color.WHITE.value, (center[0], center[1]), center[2] - 2)
            pygame.draw.circle(screen, Color.BLACK.value, (center[0], center[1]), 1)

        for group in groups:
            for point in group:
                point.draw(screen)

        for point in singles:
            point.draw(screen)

    pygame.display.update()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit = True

        if event.type == pygame.MOUSEBUTTONDOWN:
            coors = event.pos
            point = Point(coors[0], coors[1])
            points.append(point)
            point.draw(screen)
            points_count += 1

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN and len(points) == points_count:
                clusterize()
                pygame.display.update()