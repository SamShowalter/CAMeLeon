import random


class Ball:
    """
    Class to keep track of a ball's location and vector.
    """
    def __init__(self,game):
        self.game = game
        self.size = random.randrange(game.ball_size_min, game.ball_size_max)
        self.x = None
        self.y = None
        self.distance = None
        self.change_x = random.randrange(game.ball_speed_min,game.ball_speed_max)
        self.change_y = random.randrange(game.ball_speed_min,game.ball_speed_max)

        self.game.find_valid_start_position(self)

    def change_direction(self):
        # Bounce the ball if needed
        if self.y > self.game.height - self.size - self.game.edge or self.y < self.size + self.game.edge:
                self.change_y *= -1
        if self.x > self.game.width - self.size - self.game.edge or self.x < self.size + self.game.edge:
                self.change_x *= -1

    def update_location(self):
        # Move the ball's center
        self.x += self.change_x
        self.y += self.change_y
 
        self.change_direction()
            

    



 
 
