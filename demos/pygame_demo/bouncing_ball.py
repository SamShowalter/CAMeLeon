import random
from utils import *


class Ball:
    """
    Class to keep track of a ball's location and vector.
    """
    def __init__(self,game):
        self.game = game
        # self.size = random.choice(self.game.ball_sizes)
        self.x = None
        self.y = None
        self.distance = None
        self.is_player = False
        self.change_x = random.randrange(game.ball_speed_min,game.ball_speed_max)
        self.change_y = random.randrange(game.ball_speed_min,game.ball_speed_max)
        self.score = random.choice(self.game.ball_scores)
        self.size = self.game.player_size + self.score
        self.color = convert_to_rgb(0,reverse = True)

        self.game.find_valid_start_position(self)

    def update_score_color(self):
        """Update color

        :f: TODO
        :returns: TODO

        """
        self.score += 1
        self.color = convert_to_rgb(0, reverse = True)

    def update_size(self):
        """Update size of ball

        :returns: TODO

        """
        self.size = self.game.player_size + self.score

    def change_direction(self):
        # Bounce the ball if needed
        if self.y > self.game.height - self.size - self.game.edge or self.y < self.size + self.game.edge:
                self.change_y *= -1
        if self.x > self.game.width - self.size - self.game.edge or self.x < self.size + self.game.edge:
                self.change_x *= -1

    def update_location(self):
        # Move the ball's center
        self.x += self.change_x * self.game.speedup
        self.y += self.change_y * self.game.speedup

        self.change_direction()








