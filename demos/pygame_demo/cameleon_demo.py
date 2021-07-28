import pygame, sys, random
from player import Player
from math import sqrt
from bouncing_ball import Ball
from food import Food
import numpy as np
import time

class AvoidGame:

    def __init__(self,
                #Size of the player (larger  = harder to avoid others)
                player_size = 8,

                #Range of sizes for the predator balls
                ball_scores = [0,1,2,4,8],

                #Range of speeds that the predator balls can have
                ball_speed_min = -1, ball_speed_max = 2,

                #Show the playing screen and/or the stats
                show = True, show_stats = False,

                #Game baord size
                width = 800, height = 800,

                # Number of players v. number of predator balls
                num_players = 1, num_balls = 5,

                # Size of the edge, the speedup rate, and if human play is utilized
                edge = 8, speedup = 2, human_play = False,

                # How often difficulty is updated
                difficulty_update_rate_sec = 10,

                # Food available
                food_num = 3,

                # Clock rate, base time reward, win_reward
                clock_rate = 100, base_reward = 0.001, eat_reward = 10, win_reward = 500):

        #Game  Metadata Information
        self.show = show
        self.eat_reward = eat_reward
        self.show_stats = show_stats
        self.width = width + edge
        self.height = height + edge
        self.edge = edge
        self.speedup = speedup
        self.clock_rate = clock_rate
        self.player_size = player_size
        self.num_players = num_players
        self.human_play = human_play
        self.num_balls = num_balls
        self.food_num = food_num
        self.ball_scores = ball_scores
        self.ball_speed_min = ball_speed_min
        self.ball_speed_max = ball_speed_max
        self.difficulty_update_rate = difficulty_update_rate_sec
        self.base_reward = base_reward
        self.win_reward = win_reward

        # Roster of predator balls and agent
        self.roster = []

        # Food list
        self.food_list = []

        # List of predator balls
        self.ball_list = []

        #Exist message and potentially active game
        self.exit_message = {}
        self.active_game = None
        self.update_diff_clock = None

    def make_ball(self):
        """
        Function to make a new, random ball.
        """
        self.ball_list.append(Ball(self))

    def make_food(self):
        """
        Makes food for agent or opponent to eat
        """
        self.food_list.append(Food(self))

    def determine_food_size(self):
        return self.player_size

    def update_difficulty_monitor(self):
        """
        check current update for difficulty
        """
        if (time.time() - self.update_diff_clock) > self.difficulty_update_rate:
            self.update_difficulty()
            self.update_diff_clock = time.time()

    def update_difficulty(self):
        """
        Update difficulty of the game
        """
        for i in range(len(self.ball_list) // 4):
            self.num_balls += 1
            self.make_ball()

    def make_player(self):
        """
        Add a player to the roster
        """
        if self.human_play:
                self.roster.append(Player(self))
        else:
            self.roster.append(Player(self, "ADD BRAIN"))

    def update_fitness(self, player):
        """
        Update the fitness of the player
        """
        #Reward for eating opponent
        if player.eaten_opponent != False:
            player.fitness += (player.eaten_opponent+1) * self.eat_reward
            player.eaten_opponent = False

        #Reward added for food
        elif player.eaten_food:
            player.fitness += self.eat_reward/self.clock_cycles
            player.eaten_food = False

        # Negative incentive, makes agent want to move
        player.fitness -= self.base_reward

    def update_player_velocity(self, event, player):
        """
        Update the player velocity based on human play
        """
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


    def check_for_wall_collisions_human_play(self, effect = "stop"):
        """
        Check for a wall collision, which would stop the agent or kill it.

        Parameters:
            effect: string, options of ["stop", "kill"]

        """

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
            if (effect == "kill"):
                self.exit_message['reason'] = "wall"
                self.active_game = False
                self.game_stop = time.time()
                #self.running = False
            elif (effect == "stop"):
                self.exit_message['which_wall'] = ""

                # Do something to stop motion

    def check_for_food_collisions(self):
        """Check to see if there are any food collisions

        :returns: TODO

        """
        player = self.roster[0]

        checks = self.ball_list
        checks = [player] + checks

        for item in checks:
            to_remove = []
            for i,food in enumerate(self.food_list):
                if (AvoidGame.distance(item.x, item.y,food.x,food.y) < (item.size + food.size)):
                    to_remove.append(i)
                    item.score += 1
                    item.update_score_color()
                    item.eaten_food = True
                    item.update_size()
                    if item.is_player:
                        self.update_fitness(item)

            self.food_list = [self.food_list[i] for i in range(len(self.food_list)) if i not in to_remove]

        self.regenerate_food()

    def regenerate_food(self):
        """Regenerate food if needed
        :returns: TODO

        """
        l = len(self.food_list)

        for i in range(self.food_num -l):
            self.food_list.append(Food(self))


    def check_for_ball_collisions_human_play(self):
        """
        Check for ball collisions during human play
        """

        player = self.roster[0]

        to_remove = set()
        for i,ball in enumerate(self.ball_list):
            if AvoidGame.distance(ball.x, ball.y, player.x, player.y) < (ball.size + player.size):
                if (ball.score < player.score) and (i not in to_remove):
                    #Remove eaten ball
                    to_remove.add(i)
                    player.eaten_opponent = ball.score
                    player.score += 1
                    player.update_score_color()
                    player.update_size()
                    self.update_fitness(player)

                else:
                    self.exit_message['reason'] = 'ball'
                    self.active_game = False
                    self.game_stop = time.time()
                    #self.running = False
                    return

        self.ball_list = [self.ball_list[i] for i in range(len(self.ball_list)) if i not in to_remove]

    def check_win_conditions(self):
        """Check win conditions (has agent eaten all others)
        :returns: TODO

        """
        win = len(self.ball_list) == 0
        if win:
            self.roster[0].fitness += self.win_reward
            self.exit_message['reason'] = 'victory'
            return True

        return False

    def evaluate_state(self):
        """
        Evaluate the state of the game and prepare the input for the agent
        """

        #Human play game
        if self.human_play:

            self.check_for_ball_collisions_human_play()

            self.check_for_food_collisions()

            for player in self.roster:
                self.network_input(player)

            # Game has been won
            # self.check_for_wall_collisions_human_play()
            win = self.check_win_conditions()
            if win:
                self.active_game = False
                self.game_stop = time.time()




    def game_running_activities(self):

        """
        If any of the pygame events are quit, end the game
        """

        #Quit event etc
        for event in pygame.event.get():
            # Quit the game if the game ends
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                running= False

            for player in self.roster:
                # Update all player locations
                self.update_player_location(event, player)


    def on_init(self):
        """
        At the start of the game
        """
        self.clock_cycles = 1
        self.game_start = time.time()
        self.update_diff_clock = time.time()
        self.active_game = True

        # Make foods
        for i in range(self.food_num):
            self.make_food()

        # Make balls
        for i in range(self.num_balls):
            self.make_ball()

        #Create players
        for i in range(self.num_players):
            self.make_player()


        #Graphics information
        if self.show:

            #Start up pygame
            pygame.init()
            pygame.display.set_caption("CAML Sandbox Game")

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

            self = AvoidGame(human_play = self.human_play, speedup = self.speedup, show_stats = self.show_stats,
                             clock_rate = self.clock_rate)
            self.on_execute()

        for player in self.roster:
            self.update_player_velocity(event,player)

    def find_valid_start_position(self, obj):

        while True:
            bad = False
            obj.x = random.choice(np.arange(self.edge + obj.size, self.width - self.edge - obj.size,self.speedup))
            obj.y = random.choice(np.arange(self.edge + obj.size, self.height - self.edge - obj.size,self.speedup))

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
            # print(len(new_ball_list))
            # print(len(self.ball_list))
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

        # print(state_input)
        return state_input


    def on_loop(self):
        self.clock_cycles += 0.01
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
                                    .format((player.x - self.edge - self.player_size)),
                                            False, (0, 0, 0))
            right_wall = self.font.render("""Dist-R-Wall: {:>5}"""\
                                    .format((self.width - player.x - self.edge - self.player_size)),
                                            False, (0, 0, 0))
            top_wall = self.font.render("""Dist-T-Wall: {:>5}"""\
                                    .format((player.y - self.edge - self.player_size)),
                                            False, (0, 0, 0))
            bottom_wall = self.font.render("""Dist-B-Wall: {:>5}"""\
                                    .format((self.height - player.y - self.edge - self.player_size)),
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
            v_edge.fill((0,0,0))        #print(rect.x, rect.y)
            self.screen.fill((255, 255, 255))

            #Draw players
            for player in self.roster:
                pygame.draw.circle(self.screen, player.color, [player.x, player.y], player.size)

            for ball in self.ball_list:
                pygame.draw.circle(self.screen, ball.color, [ball.x, ball.y], ball.size)

            for food in self.food_list:
                pygame.draw.circle(self.screen, food.color, [food.x, food.y], food.size)


            self.screen.blit(v_edge, (0, self.height-self.edge))
            self.screen.blit(h_edge, (self.width-self.edge, 0))
            self.screen.blit(v_edge, (0, 0))
            self.screen.blit(h_edge, (0, 0))


            if self.show_stats:

                self.screen.blit(x_pos,(self.width - self.edge - 130, self.edge + 10))
                self.screen.blit(y_pos,(self.width - self.edge - 130, self.edge + 30))

                self.screen.blit(left_wall,(self.edge + 5, self.edge + 10))
                self.screen.blit(right_wall,(self.edge + 5, self.edge + 30))
                self.screen.blit(top_wall,(self.edge + 5, self.edge + 50))
                self.screen.blit(bottom_wall,(self.edge + 5, self.edge + 70))

                self.screen.blit(seconds_alive,(self.edge + 5, self.edge + 110))
                self.screen.blit(fitness,(self.edge + 5, self.edge + 130))

            # string_image = pygame.image.tostring(self.screen, 'RGB')
            # temp_surf = pygame.image.fromstring(string_image,(self.width, self.height),'RGB' )
            # tmp_arr = pygame.surfarray.array3d(temp_surf)
            # print(tmp_arr.shape)
            # sys.exit(1)

        else:
            exit_font_title = pygame.font.SysFont('freesansbold.ttf',140)
            exit_font_reason = pygame.font.SysFont('freesansbold.ttf',50)
            exit_font_score = pygame.font.SysFont('freesansbold.ttf',25)
            exit_reason = None
            fitness_score = exit_font_score.render("""Total Fitness: {:>5}"""\
                                                    .format(int(self.roster[0].fitness)),
                                                     False, (0, 0, 0))

            if self.exit_message['reason'] == 'victory':
                exit_title = exit_font_title.render("""You Win!""", False, (0, 0, 0))
                self.screen.blit(exit_title,(self.edge + 180, self.edge + 200))
                self.screen.blit(fitness_score,(self.edge + 110, self.edge + 450))

            else:

                exit_title = exit_font_title.render("""Game Over""", False, (0, 0, 0))
                if self.exit_message['reason'] == 'wall':
                    exit_reason = exit_font_reason.render("""You ran into the {} wall"""\
                                                    .format(self.exit_message['which_wall']),
                                                        False, (0, 0, 0))
                    self.screen.blit(exit_reason,(self.edge + 110, self.edge + 350))
                    self.screen.blit(fitness_score,(self.edge + 110, self.edge + 450))

                elif self.exit_message['reason'] == 'ball':
                    exit_reason = exit_font_reason.render("""You were hit by a predator ball""",
                                                        False, (0, 0, 0))
                    self.screen.blit(exit_reason,(self.edge + 110, self.edge + 350))
                    self.screen.blit(fitness_score,(self.edge + 110, self.edge + 450))

                self.screen.blit(exit_title,(self.edge + 100, self.edge + 200))


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
    Game = AvoidGame(human_play = True, clock_rate = 60,
                     show_stats = True,
                     speedup = 4)
    Game.on_execute()
