import pygame
import sys
import math
import neat
import os

WIDTH, HEIGHT          = 1280, 720
CAR_SIZE_X, CAR_SIZE_Y = 40, 60
BORDER_COLOR           = (255, 255, 255)
SPEED                  = 8
GENERATION_TIME        = 40 * 60
MAP_FILE               = "map.png"
CURRENT_GENERATION     = 0

SPAWN_X     = 400
SPAWN_Y     = 625
SPAWN_ANGLE = 0


class Car:
    def __init__(self):
        self.sprite = pygame.image.load("car.png").convert_alpha()
        self.sprite = pygame.transform.scale(self.sprite, (CAR_SIZE_X, CAR_SIZE_Y))
        self.rotated_sprite = self.sprite
        self.position = [float(SPAWN_X), float(SPAWN_Y)]
        self.angle    = float(SPAWN_ANGLE)
        self.center   = [self.position[0] + CAR_SIZE_X / 2,
                         self.position[1] + CAR_SIZE_Y / 2]
        self.alive    = True
        self.distance = 0
        self.time     = 0
        self.radars   = [[(int(self.center[0]), int(self.center[1])), 1] for _ in range(5)]

    def draw(self, screen):
        screen.blit(self.rotated_sprite, self.position)
        for radar in self.radars:
            pygame.draw.line(screen, (0, 255, 0), self.center, radar[0], 1)
            pygame.draw.circle(screen, (0, 255, 0), radar[0], 5)

    def check_collision(self, game_map):
        self.alive = True
        for point in self.corners:
            x, y = int(point[0]), int(point[1])
            if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                self.alive = False
                return
            if game_map.get_at((x, y)) == BORDER_COLOR:
                self.alive = False
                return

    def check_radar(self, degree, game_map):
        length = 0
        x = int(self.center[0])
        y = int(self.center[1])
        while length < 300:
            x = int(self.center[0] + math.cos(math.radians(360 - (self.angle + degree))) * length)
            y = int(self.center[1] + math.sin(math.radians(360 - (self.angle + degree))) * length)
            if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                break
            if game_map.get_at((x, y)) == BORDER_COLOR:
                break
            length += 1
        dist = int(math.sqrt((x - self.center[0]) ** 2 + (y - self.center[1]) ** 2))
        self.radars.append([(x, y), dist])

    def update(self, game_map):
        self.rotated_sprite = pygame.transform.rotate(self.sprite, self.angle)
        self.position[0] += math.cos(math.radians(360 - self.angle)) * SPEED
        self.position[1] += math.sin(math.radians(360 - self.angle)) * SPEED
        self.distance += SPEED
        self.time     += 1
        self.center = [int(self.position[0]) + CAR_SIZE_X / 2,
                       int(self.position[1]) + CAR_SIZE_Y / 2]
        length = 0.5 * CAR_SIZE_X
        self.corners = [
            [self.center[0] + math.cos(math.radians(360 - (self.angle +  30))) * length,
             self.center[1] + math.sin(math.radians(360 - (self.angle +  30))) * length],
            [self.center[0] + math.cos(math.radians(360 - (self.angle -  30))) * length,
             self.center[1] + math.sin(math.radians(360 - (self.angle -  30))) * length],
            [self.center[0] + math.cos(math.radians(360 - (self.angle + 150))) * length,
             self.center[1] + math.sin(math.radians(360 - (self.angle + 150))) * length],
            [self.center[0] + math.cos(math.radians(360 - (self.angle - 150))) * length,
             self.center[1] + math.sin(math.radians(360 - (self.angle - 150))) * length],
        ]
        self.check_collision(game_map)
        self.radars.clear()
        for d in [-90, -45, 0, 45, 90]:
            self.check_radar(d, game_map)

    def get_data(self):
        data = [int(r[1] / 30) for r in self.radars]
        while len(data) < 5:
            data.append(0)
        return data[:5]

    def is_alive(self):
        return self.alive

    def get_reward(self):
        return self.distance / (CAR_SIZE_X / 2)


def run_simulation(genomes, config):
    global CURRENT_GENERATION
    CURRENT_GENERATION += 1
    nets, cars = [], []

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"Self Driving Car  Generation {CURRENT_GENERATION}")
    clock      = pygame.time.Clock()
    gen_font   = pygame.font.SysFont("Arial", 30)
    alive_font = pygame.font.SysFont("Arial", 20)
    game_map   = pygame.transform.scale(
                     pygame.image.load(MAP_FILE).convert(), (WIDTH, HEIGHT))

    for i, g in genomes:
        nets.append(neat.nn.FeedForwardNetwork.create(g, config))
        g.fitness = 0
        cars.append(Car())

    counter = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit(); sys.exit(0)

        for i, car in enumerate(cars):
            if not car.is_alive():
                continue
            output = nets[i].activate(car.get_data())
            if output[0] > output[1]:
                car.angle += 10
            else:
                car.angle -= 10

        still_alive = 0
        for i, car in enumerate(cars):
            if car.is_alive():
                still_alive += 1
                car.update(game_map)
                genomes[i][1].fitness += car.get_reward()

        if still_alive == 0:
            break
        counter += 1
        if counter == GENERATION_TIME:
            break

        screen.blit(game_map, (0, 0))
        for car in cars:
            if car.is_alive():
                car.draw(screen)

        t = gen_font.render(f"Generation: {CURRENT_GENERATION}", True, (0, 0, 255))
        screen.blit(t, t.get_rect(center=(900, 450)))
        t = alive_font.render(f"Still Alive: {still_alive}", True, (0, 0, 255))
        screen.blit(t, t.get_rect(center=(900, 490)))

        pygame.display.flip()
        clock.tick(60)


if __name__ == "__main__":
    for f in [MAP_FILE, "car.png", "config.txt"]:
        if not os.path.exists(f):
            print(f"ERROR: {f} not found!")
            sys.exit(1)
    config = neat.config.Config(
        neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
        "./config.txt")
    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.run(run_simulation, 1000)
