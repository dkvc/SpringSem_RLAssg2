import random

import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk


UNIT = 40   # pixels per cell (width and height)
MAZE_H = 10  # height of the entire grid in cells
MAZE_W = 10  # width of the entire grid in cells
origin = np.array([UNIT/2, UNIT/2])


class Maze(tk.Tk, object):
    def __init__(self, agentXY, goalXY, walls=[],pits=[]):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.wallblocks = []
        self.pitblocks=[]
        self.UNIT = 40   # pixels per cell (width and height)
        self.MAZE_H = 10  # height of the entire grid in cells
        self.MAZE_W = 10  # width of the entire grid in cells
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_W * UNIT))
        self.build_shape_maze(agentXY, goalXY, walls, pits)
        #self.build_maze()

    def build_shape_maze(self,agentXY,goalXY, walls,pits):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)


        for x,y in walls:
            self.add_wall(x,y)
        for x,y in pits:
            self.add_pit(x,y)
        self.add_goal(goalXY[0],goalXY[1])
        self.add_agent(agentXY[0],agentXY[1])
        self.canvas.pack()

    '''Add a solid wall block at coordinate for centre of bloc'''
    def add_wall(self, x, y):
        wall_center = origin + np.array([UNIT * x, UNIT*y])
        self.wallblocks.append(self.canvas.create_rectangle(
            wall_center[0] - 15, wall_center[1] - 15,
            wall_center[0] + 15, wall_center[1] + 15,
            fill='black'))

    '''Add a solid pit block at coordinate for centre of bloc'''
    def add_pit(self, x, y):
        pit_center = origin + np.array([UNIT * x, UNIT*y])
        self.pitblocks.append(self.canvas.create_rectangle(
            pit_center[0] - 15, pit_center[1] - 15,
            pit_center[0] + 15, pit_center[1] + 15,
            fill='blue'))

    '''Add a solid goal for goal at coordinate for centre of bloc'''
    def add_goal(self, x=4, y=4):
        goal_center = origin + np.array([UNIT * x, UNIT*y])

        self.goal = self.canvas.create_oval(
            goal_center[0] - 15, goal_center[1] - 15,
            goal_center[0] + 15, goal_center[1] + 15,
            fill='yellow')

    '''Add a solid wall red block for agent at coordinate for centre of bloc'''
    def add_agent(self, x=0, y=0):
        agent_center = origin + np.array([UNIT * x, UNIT*y])

        self.agent = self.canvas.create_rectangle(
            agent_center[0] - 15, agent_center[1] - 15,
            agent_center[0] + 15, agent_center[1] + 15,
            fill='red')
        
    def is_collision(self, bbox1, bbox2):
        # Check if two bounding boxes collide
        return not (bbox2[0] > bbox1[2] or bbox2[2] < bbox1[0] or bbox2[1] > bbox1[3] or bbox2[3] < bbox1[1])
        
    def update_dynamic_elements(self):
        # Move a randomly selected obstacle to a neighboring grid cell if possible
        for obstacle_to_move in self.pitblocks:
            #obstacle_to_move = random.choice(self.pitblocks)
            current_x, current_y = self.canvas.coords(obstacle_to_move)[:2]

            # Calculate grid indices for current position
            current_grid_x = int(current_x / UNIT)
            current_grid_y = int(current_y / UNIT)

            # Define possible neighboring grid cells
            # move offset [-2, -1, 0, 1, 2] in x and y direction
            move_offset = random.choice([-2, -1, 0, 1, 2])
            neighboring_grids = [(current_grid_x - move_offset, current_grid_y),  # left
                                (current_grid_x + move_offset, current_grid_y),  # right
                                (current_grid_x, current_grid_y - move_offset),  # up
                                (current_grid_x, current_grid_y + move_offset),  # down
                                (current_grid_x + move_offset, current_grid_y + move_offset),  # down-right
                                (current_grid_x - move_offset, current_grid_y - move_offset),  # up-left
                                (current_grid_x - move_offset, current_grid_y + move_offset),  # down-left
                                (current_grid_x + move_offset, current_grid_y - move_offset)  # up-right
                                ]

            # Filter valid neighboring grid cells that stay within the maze
            valid_grids = [(x, y) for x, y in neighboring_grids
                        if 0 <= x < MAZE_W and 0 <= y < MAZE_H]

            if valid_grids:
                # Randomly select a neighboring grid cell to move to
                new_grid_x, new_grid_y = random.choice(valid_grids)

                # Calculate new position within the selected grid cell
                new_x = new_grid_x * UNIT + UNIT / 2
                new_y = new_grid_y * UNIT + UNIT / 2

                # Check for collision with the circle (goal) and walls
                circle_bbox = self.canvas.coords(self.goal)
                wall_collisions = any(self.is_collision((new_x - 15, new_y - 15, new_x + 15, new_y + 15), self.canvas.coords(wall)) for wall in self.wallblocks)
                obstacle_collisions = any(self.is_collision((new_x - 15, new_y - 15, new_x + 15, new_y + 15), self.canvas.coords(obstacle)) for obstacle in self.pitblocks if obstacle != obstacle_to_move)
            
                if self.is_collision((new_x - 15, new_y - 15, new_x + 15, new_y + 15), circle_bbox) or wall_collisions or obstacle_collisions:
                    print("Collision detected. Obstacle cannot be moved.")
                else:
                    # Move the obstacle to the new position
                    self.canvas.coords(obstacle_to_move, new_x - 15, new_y - 15, new_x + 15, new_y + 15)


    def reset(self, value = 1, resetAgent=True, episodeVal=0):
        if episodeVal and episodeVal % 5 == 0:
            self.update_dynamic_elements()
        self.update()
        time.sleep(0.2)
        if(value == 0):
            return self.canvas.coords(self.agent)
        else:
            #Reset Agent
            if(resetAgent):
                self.canvas.delete(self.agent)
                self.agent = self.canvas.create_rectangle(origin[0] - 15, origin[1] - 15,
                origin[0] + 15, origin[1] + 15,
                fill='red')

            return self.canvas.coords(self.agent)

    '''computeReward - definition of reward function'''
    def computeReward(self, currstate, action, nextstate):
            reverse=False
            if nextstate == self.canvas.coords(self.goal):
                reward = 1
                done = True
                nextstate = 'terminal'
            #elif nextstate in [self.canvas.coords(self.pit1), self.canvas.coords(self.pit2)]:
            elif nextstate in [self.canvas.coords(w) for w in self.wallblocks]:
                reward = -0.3
                done = False
                nextstate = currstate
                reverse=True
                #print("Wall penalty:{}".format(reward))
            elif nextstate in [self.canvas.coords(w) for w in self.pitblocks]:
                reward = -10
                done = True
                nextstate = 'terminal'
                reverse=False
                #print("Wall penalty:{}".format(reward))
            else:
                reward = -0.1
                done = False
            return reward,done, reverse

    '''step - definition of one-step dynamics function'''
    def step(self, action):
        s = self.canvas.coords(self.agent)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.agent, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.agent)  # next state
        #print("s_.coords:{}({})".format(self.canvas.coords(self.agent),type(self.canvas.coords(self.agent))))
        #print("s_:{}({})".format(s_, type(s_)))

        # call the reward function
        reward, done, reverse = self.computeReward(s, action, s_)
        if(reverse):
            self.canvas.move(self.agent, -base_action[0], -base_action[1])  # move agent back
            s_ = self.canvas.coords(self.agent)  

        return s_, reward, done

    def render(self, sim_speed=.01):
        time.sleep(sim_speed)
        self.update()


def update():
    for t in range(10):
        print("The value of t is", t)
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break


if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
