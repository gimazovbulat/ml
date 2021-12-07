import math
import numpy as np
import pygame, math
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs

points, labels = make_blobs(n_samples=100, centers=2, center_box=((100, 100), (700, 500)), cluster_std=50)


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


"""
 Вычисляет к какому кластеру принадлежат точки
 """
def kNN(train_data, train_labels, test_data, k, num_of_classes):
"""
 Лейблы кластеров
 """
    predicted_labels = []
    """
     проходимся по точкам
     """
    for point in test_data:
    """
     Вычисляем дистанцию от точки до других точек
     """
        test_dist = [[dist(point, train_data[i]), train_labels[i]] for i in range(len(train_data))]
        stat = [0 for i in range(num_of_classes)]
        """
         Для каждого класса подсчитываем количество ближайших точек до текущей точки
         """
        for d in sorted(test_dist)[0:k]:
            stat[d[1]] += 1
         """
          Относим точку к одному из классов
          """
        predicted_labels.append(sorted(zip(stat, range(num_of_classes)), reverse=True)[0][1])
    return predicted_labels


def kfold(train_data, train_labels, num_folds, k, num_of_classes):
    subset_size = int(len(train_data) / num_folds)
    accuracies = []
    for i in range(num_folds):
        testing_this_round = train_data[i * subset_size:][:subset_size]
        training_this_round = train_data.tolist()[:i * subset_size] + train_data.tolist()[(i + 1) * subset_size:]
        test_labels_this_round = train_labels[i * subset_size:][:subset_size]
        train_labels_this_round = train_labels.tolist()[:i * subset_size]
        train_labels_this_round.extend(train_labels[(i + 1) * subset_size:])
        predicted = kNN(training_this_round, train_labels_this_round, testing_this_round, k, num_of_classes)
        accuracy = sum([1 if predicted[i] == test_labels_this_round[i] else 0]) / len(predicted)
        accuracies.append(accuracy)
    return sum(accuracies) / num_folds


def calculate_k(train_data, train_labels, num_of_classes):
    max_accuracy = 0
    k = 0
    for i in range(1, 6):
        accuracy = kfold(train_data, train_labels, 10, i, num_of_classes)
        if accuracy > max_accuracy:
            k = i
            max_accuracy = accuracy
    return k


K = calculate_k(points, labels, 2)

WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
FPS = 30
colors = [RED if label == 0 else BLACK for label in labels]
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill(WHITE)
pygame.display.set_caption("SVM")
clock = pygame.time.Clock()
drawn = []
drawn_colors = []
running = True
while running:
    clock.tick(FPS)
    mouse = pygame.mouse.get_pos()
    for point in zip(points, colors):
        pygame.draw.circle(screen, point[1], point[0], 5)
    for point in zip(drawn, drawn_colors):
        pygame.draw.circle(screen, point[1], point[0], 5)
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            drawn.append(mouse)
            predicted = kNN(points, labels, [mouse], K, 2)
            if predicted[0] == 0:
                color = RED
            else:
                color = BLACK
            drawn_colors.append(color)
        if event.type == pygame.QUIT:
            running = False
    pygame.display.flip()
pygame.quit()
