import numpy as np
import pygame
from sklearn import svm
from sklearn.datasets import make_blobs

points, labels = make_blobs(n_samples=40, centers=2, center_box=((100, 100), (700, 500)), cluster_std=50)
running = True
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
FPS = 30
clf = svm.SVC(kernel='linear', C=1000)
clf.fit(points, labels)
w = clf.coef_[0]
# ------------
colors = [RED if label == 0 else BLACK for label in labels]
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
screen.fill(WHITE)
pygame.display.set_caption("SVM")
clock = pygame.time.Clock()
drawn = []
drawn_colors = []
xx = np.linspace(0, 800)
w = clf.coef_[0]
a = -w[0] / w[1]
yy = a * xx - (clf.intercept_[0]) / w[1]
print('yy', yy)
while running:
    clock.tick(FPS)
    mouse = pygame.mouse.get_pos()
    for point in zip(xx, yy):
        pygame.draw.circle(screen, BLUE, point, 1)
    for point in zip(points, colors):
        pygame.draw.circle(screen, point[1], point[0], 5)
    for point in zip(drawn, drawn_colors):
        pygame.draw.circle(screen, point[1], point[0], 5)
    for point in clf.support_vectors_:
        center = int(point[0]), int(point[1])
        pygame.draw.circle(screen, GREEN, center, 10, 2)
    pygame.draw.line(screen, BLUE, (xx[0], yy[0]), (xx[-1], yy[-1]), 1)
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN:
            predicted_class = clf.predict([mouse])
            drawn.append(mouse)
            if predicted_class == 0:
                color = RED
            else:
                color = BLACK
            drawn_colors.append(color)
        if event.type == pygame.QUIT:
            running = False
            print('Stop')
    pygame.display.flip()
pygame.quit()
