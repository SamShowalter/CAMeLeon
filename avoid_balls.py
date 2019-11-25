import pygame, sys, random
from player import Player
from math import sqrt
from bouncing_ball import Ball
import numpy as np 
import time

class AvoidGame:

	def __init__(self, 
				player_size = 7, 
				ball_size_min = 6, ball_size_max = 12,
				ball_speed_min = -1, ball_speed_max = 2,
				show = True,
				width = 1000, height = 700, 
				num_players = 1, num_balls = 5,
				edge = 15, speedup = 2, human_play = False, 
				difficulty_update_rate_sec = 10,
				clock_rate = 300, base_reward = 0.01):

		#Game  Metadata Information
		self.show = show
		self.width = width
		self.height = height 
		self.edge = edge 
		self.speedup = speedup
		self.clock_rate = clock_rate
		self.player_size = player_size
		self.num_players = num_players
		self.human_play = human_play
		self.num_balls = num_balls
		self.ball_size_min = ball_size_min
		self.ball_size_max = ball_size_max
		self.ball_speed_min = ball_speed_min
		self.ball_speed_max = ball_speed_max
		self.difficulty_update_rate = difficulty_update_rate_sec
		self.base_reward = base_reward
		self.roster = []
		self.ball_list = []
		self.exit_message = {}
		self.active_game = None
		self.update_diff_clock = None

	def make_ball(self):
	    """
	    Function to make a new, random ball.
	    """
	    self.ball_list.append(Ball(self))


	def update_difficulty_monitor(self):

		if (time.time() - self.update_diff_clock) > self.difficulty_update_rate:
			self.update_difficulty()
			self.update_diff_clock = time.time()

	def update_difficulty(self):

		for i in range(len(self.ball_list) // 4):
			self.num_balls += 1
			self.make_ball()

	def make_player(self):
		if self.human_play:
				self.roster.append(Player(self))
		else:
			self.roster.append(Player(self, "ADD BRAIN"))

	def update_fitness(self, player):
		player.fitness += self.base_reward * self.num_balls

	def update_player_velocity(self, event, player):

		if self.human_play:

			if event.type == pygame.KEYUP:            
				if event.key in [pygame.K_LEFT, pygame.K_RIGHT]:          
				    player.change_x = 0
				if event.key in [pygame.K_UP, pygame.K_DOWN]:      
				    player.change_y = 0

			elif event.type == pygame.KEYDOWN:          
			    if event.key == pygame.K_LEFT:        # left arrow turns left
			        player.change_x = -1
			    if event.key == pygame.K_RIGHT:       # right arrow turns right
			        player.change_x = 1
			    if event.key == pygame.K_UP:          # up arrow goes up
			        player.change_y = -1
			    if event.key == pygame.K_DOWN:        # down arrow goes down
			        player.change_y = 1
		        

	def check_for_wall_collisions_human_play(self):

		self.exit_message['which_wall'] = ""

		player = self.roster[0]

		if player.x < self.edge + player.size:
			self.exit_message['which_wall'] = "left"
		elif player.x > self.width - self.edge - player.size:
			self.exit_message['which_wall'] = "right"
		elif player.y < self.edge + player.size:
			self.exit_message['which_wall'] = "top"
		elif player.y > self.height - self.edge - player.size:
			self.exit_message['which_wall'] = "bottom"

		if self.exit_message['which_wall'] != "":
			self.exit_message['reason'] = "wall"
			self.active_game = False
			self.game_stop = time.time()
			#self.running = False

	def check_for_ball_collisions_human_play(self):

		player = self.roster[0]

		for ball in self.ball_list:
			if AvoidGame.distance(ball.x, ball.y, player.x, player.y) < (ball.size + player.size):
				self.exit_message['reason'] = 'ball'
				self.active_game = False
				self.game_stop = time.time()
				#self.running = False
				return

	def evaluate_state(self):

		if self.human_play:
			self.check_for_wall_collisions_human_play()

			if self.exit_message.get('reason', None) != "wall":
				self.check_for_ball_collisions_human_play()

		for player in self.roster:
			self.network_input(player)


	    	
	def game_running_activities(self):

	    #Quit event etc
	    for event in pygame.event.get():
	        if event.type == pygame.QUIT:
	            pygame.quit()
	            sys.exit()
	            running= False

	        for player in self.roster:
	        	self.update_player_location(event, player)

	    
	def on_init(self):
		self.game_start = time.time()
		self.update_diff_clock = time.time()
		self.active_game = True

		for i in range(self.num_balls):
			self.make_ball()

		#Create players
		for i in range(self.num_players):
			self.make_player()


		#Graphics information
		if self.show:

			#Start up pygame
			pygame.init()
			pygame.display.set_caption("NEAT Avoidance Game")

			#Kick off font
			pygame.font.init()
			self.font = pygame.font.SysFont('Arial', 15)

			#Screen and clock
			self.clock = pygame.time.Clock()
			self.screen = pygame.display.set_mode((self.width,
													self.height))

			self.running = True

	def on_event(self, event):
		if event.type == pygame.QUIT:
		    self.running = False

		if not self.active_game \
			and event.type == pygame.KEYDOWN \
			and event.key == pygame.K_SPACE:

			self = AvoidGame(human_play = True)
			self.on_execute()

		for player in self.roster:
			self.update_player_velocity(event,player)

	def find_valid_start_position(self, obj):

		while True:
			bad = False
			obj.x = random.randrange(self.edge + obj.size, self.width - self.edge - obj.size)
			obj.y = random.randrange(self.edge + obj.size, self.height - self.edge - obj.size)

			for ball in self.ball_list:
			    if AvoidGame.distance(ball.x, ball.y, obj.x, obj.y) < (ball.size + obj.size):
			        bad = True

			for player in self.roster:
				if AvoidGame.distance(player.x, player.y, obj.x, obj.y) < (ball.size + obj.size):
					bad = True

			if not bad:
			    return

	def network_input(self, player):

		new_ball_list = []
		
		for ball in self.ball_list:
			placed = False
			ball.distance = AvoidGame.distance(ball.x, ball.y, player.x, player.y)
			if int(ball.distance) == 0:
				print("\n\n\n\nITS WORKING\n\n\n\n")
			for i in range(len(new_ball_list)):
				if ball.distance < new_ball_list[i].distance:
					new_ball_list.insert(i, ball)
					placed = True
					break
			if not placed:
				new_ball_list.append(ball)

		if len(new_ball_list) > len(self.ball_list):
			print(len(new_ball_list))
			print(len(self.ball_list))
			raise ValueError("Something is wrong")

		state_input = [player.x, player.y, player.change_x, player.change_y, player.size]

		top_5_closest_balls = new_ball_list[:5]

		for ball in top_5_closest_balls:

			state_input.append(ball.x)
			state_input.append(ball.y)
			state_input.append(ball.change_x)
			state_input.append(ball.change_y)
			state_input.append(ball.size)

		dist_l_wall = player.x - self.edge - self.player_size
		dist_r_wall = self.width - player.x - self.edge - self.player_size
		dist_t_wall = player.y - self.edge - self.player_size
		dist_b_wall = self.height - player.y - self.edge - self.player_size

		state_input.append(dist_l_wall)
		state_input.append(dist_r_wall)
		state_input.append(dist_t_wall)
		state_input.append(dist_b_wall)

		print(state_input)
		return state_input


	def on_loop(self):
		for ball in self.ball_list:
			ball.update_location()

		for player in self.roster:
			player.update_location()
			self.update_fitness(player)

		self.update_difficulty_monitor()

	def on_render(self):

		if self.active_game:
			player = self.roster[0] 
			x_pos = self.font.render("""X-Position: {:>5}""".format(player.x), False, (0, 0, 0))
			y_pos = self.font.render("""Y-Position: {:>5}""".format(player.y), False, (0, 0, 0))

			left_wall = self.font.render("""Dist-L-Wall: {:>5}"""\
									.format(round(np.log((player.x - self.edge - self.player_size)/10),3)), 
											False, (0, 0, 0))
			right_wall = self.font.render("""Dist-R-Wall: {:>5}"""\
									.format(round(np.log((self.width - player.x - self.edge - self.player_size) / 10),3)), 
											False, (0, 0, 0))
			top_wall = self.font.render("""Dist-T-Wall: {:>5}"""\
									.format(round(np.log((player.y - self.edge - self.player_size)/10),3)), 
											False, (0, 0, 0))
			bottom_wall = self.font.render("""Dist-B-Wall: {:>5}"""\
									.format(round(np.log((self.height - player.y - self.edge - self.player_size)/10),3)), 
											False, (0, 0, 0))

			seconds_alive = self.font.render("""Seconds Alive: {:>8}"""\
									.format(int(time.time() - self.game_start)), 
											False, (0, 0, 0))

			fitness = self.font.render("""Fitness: {:>5}"""\
									.format(round(player.fitness,3)), 
											False, (0, 0, 0))

			#Game edges
			h_edge = pygame.Surface((self.edge, self.height), 0)
			h_edge.fill((0,0,0))
			v_edge = pygame.Surface((self.width, self.edge), 0)
			v_edge.fill((0,0,0))	    #print(rect.x, rect.y)
			self.screen.fill((255, 255, 255))

			#Draw players
			for player in self.roster:
				pygame.draw.circle(self.screen, (0,0,255), [player.x, player.y], player.size)

			for ball in self.ball_list:
				pygame.draw.circle(self.screen, (255,0,0), [ball.x, ball.y], ball.size)

			self.screen.blit(v_edge, (0, self.height-self.edge))
			self.screen.blit(h_edge, (self.width-self.edge, 0))
			self.screen.blit(v_edge, (0, 0))
			self.screen.blit(h_edge, (0, 0))

			self.screen.blit(x_pos,(self.width - self.edge - 100, self.edge + 10))
			self.screen.blit(y_pos,(self.width - self.edge - 100, self.edge + 30))

			self.screen.blit(left_wall,(self.edge + 5, self.edge + 10))
			self.screen.blit(right_wall,(self.edge + 5, self.edge + 30))
			self.screen.blit(top_wall,(self.edge + 5, self.edge + 50))
			self.screen.blit(bottom_wall,(self.edge + 5, self.edge + 70))

			self.screen.blit(seconds_alive,(self.edge + 5, self.edge + 110))
			self.screen.blit(fitness,(self.edge + 5, self.edge + 130))

		else:
			exit_font_title = pygame.font.SysFont('freesansbold.ttf',200)
			exit_font_reason = pygame.font.SysFont('freesansbold.ttf',80)
			exit_font_score = pygame.font.SysFont('freesansbold.ttf',30)
			exit_reason = None
			exit_title = exit_font_title.render("""Game Over""", False, (0, 0, 0))
			elapsed_time_score = exit_font_score.render("""Total Time Score: {:>5}"""\
													.format(int(self.game_stop - self.game_start)),
													 False, (0, 0, 0))

			if self.exit_message['reason'] == 'wall':
				exit_reason = exit_font_reason.render("""You ran into the {} wall"""\
												.format(self.exit_message['which_wall']),
													False, (0, 0, 0))
				self.screen.blit(exit_reason,(self.edge + 120, self.edge + 400))
				self.screen.blit(elapsed_time_score,(self.edge + 120, self.edge + 500))

			elif self.exit_message['reason'] == 'ball':
				exit_reason = exit_font_reason.render("""You were hit by a predator ball""",
													False, (0, 0, 0))
				self.screen.blit(exit_reason,(self.edge + 80, self.edge + 400))
				self.screen.blit(elapsed_time_score,(self.edge + 80, self.edge + 500))
			
			self.screen.blit(exit_title,(self.edge + 100, self.edge + 70))
			

			

		pygame.display.update()
		self.clock.tick(self.clock_rate)

	def on_cleanup(self):
	    pygame.quit()

	def on_execute(self):
		self.on_init()
		self.update_difficulty_monitor()
		while(self.running):

			for event in pygame.event.get():
				self.on_event(event)
	        
			if self.active_game:
				self.on_loop()

				if self.show:
					self.on_render()

				self.evaluate_state()

			elif self.show:
				self.on_render()

		self.on_cleanup()

	@staticmethod
	def distance(x1,y1, x2, y2):
		return sqrt((x1 - x2)**2 + (y1 - y2)**2)
 
if __name__ == "__main__" :
    Game = AvoidGame(human_play = True, clock_rate = 300, speedup = 2)
    Game.on_execute()