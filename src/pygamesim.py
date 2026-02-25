import pygame
from sys import exit

pygame.init()
WIDTH = 1000
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Double Pendulum Simulation")
clock = pygame.time.Clock()

#variables
gravity = 0.5 

class Ball: 
    def __init__(self, x_pos, y_pos, radius, color, mass, x_speed, y_speed):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.radius = radius
        self.color = color
        self.mass = mass
        self.x_speed = x_speed
        self.y_speed = y_speed
        self.circle = ''

    def draw(self):
        self.circle = pygame.draw.circle(screen, self.color, (self.x_pos, self.y_pos), self.radius)

    def check_gravity(self):
        if self.y_pos < HEIGHT - self.radius:
            self.y_speed += gravity


        
ball1 = Ball(30, 30, 10, 'white', 100, 0, 0)
ball2 = Ball(100, 100, 10, 'white', 100, 0, 0)

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    ball1.draw()
    ball2.draw() 
    pygame.display.update()
    clock.tick(60)