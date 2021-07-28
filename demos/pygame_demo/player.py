import pygame
from utils import *

class Player:
    """
    Class to keep track of a ball's location and vector.
    """
    def __init__(self, game, brain = None):
        self.game = game
        self.show = game.show
        self.brain = brain
        self.fitness = 0
        self.is_player = True
        self.eaten_food = False
        self.eaten_opponent = False
        self.x = None
        self.y = None
        self.score = 0
        self.color = convert_to_rgb(self.score)
        self.change_x = 0
        self.change_y = 0
        self.size = game.player_size
        self.game.find_valid_start_position(self)


    def update_size(self):
        """Update size of ball

        :returns: TODO

        """
        self.size = self.game.player_size + self.score

    def update_score_color(self):
        """Update color

        :f: TODO
        :returns: TODO

        """
        self.score += 1
        self.color = convert_to_rgb(self.score)

    def update_location(self, stop_at_wall = True):

        if stop_at_wall:
            if (((self.y >= self.game.height - self.size - self.game.edge) and
                (self.change_y == 1)) or
                ((self.y <= self.size + self.game.edge) and
                (self.change_y == -1))):
                    self.change_y = 0
            if (((self.x >= self.game.width - self.size - self.game.edge) and
                 (self.change_x == 1)) or
                ((self.x <= self.size + self.game.edge) and
                 (self.change_x == -1))):
                    self.change_x = 0

        self.x += self.change_x * self.game.speedup * 2
        self.y += self.change_y * self.game.speedup * 2

