import pygame

class Player:
	"""
	Class to keep track of a ball's location and vector.
	"""
	def __init__(self, game, brain = None):
		self.game = game
		self.show = game.show
		self.brain = brain
		self.fitness = 0
		self.x = None
		self.y = None
		self.change_x = 0
		self.change_y = 0
		self.size = game.player_size

		self.game.find_valid_start_position(self)

	def find_valid_start_position(self):

		while True:
			sample_x = random.randrange(self.game.edge + self.size, 
										self.game.width - self.game.edge - self.size)

			sample_y = random.randrange(self.game.edge + self.size, 
										self.game.height - self.game.edge - self.size)



	def update_location(self):
		self.x += self.change_x * self.game.speedup
		self.y += self.change_y * self.game.speedup

