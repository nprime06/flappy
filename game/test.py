#!/usr/bin/env python3
import numpy as np
import gymnasium as gym
import flappy_bird_gymnasium  # registers FlappyBird-* env ids with gymnasium
import random
import time
import matplotlib.pyplot as plt

def main():
    env_id = "FlappyBird-v0"
    interactive = True
    p = 0.2  # when interactive=False: probability of taking action=1 (flap)
    seed = 0  # RNG seed for reproducible random policy
    fps = 30

    
    render_mode = "human" # headless (none), human, rgb_array
    env = gym.make(env_id, screen_size=(576, 1024), render_mode=render_mode, disable_env_checker=True, use_lidar=False)
    rng = random.Random(seed)

    obs, info = env.reset()
    # env.render() # render the initial state

    pygame = None
    clock = None
    if interactive:
        # Only initialize pygame when we actually want keyboard input / event processing.
        import pygame as _pygame

        pygame = _pygame
        pygame.init()
        clock = pygame.time.Clock()
    running = True
    terminated = False
    truncated = False

    obs_list = []

    step = 0
    while running:
        step += 1
        action = 0  # 0 = do nothing, 1 = flap

        if interactive:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False

                    # flap
                    elif event.key in (pygame.K_SPACE, pygame.K_UP):
                        action = 1

                    # restart after death
                    elif event.key == pygame.K_r and (terminated or truncated):
                        obs, info = env.reset()
                        terminated = False
                        truncated = False
        else:
            # non-interactive + headless: no pygame, just pick actions randomly
            if not (terminated or truncated):
                action = 1 if (rng.random() < p) else 0

        if not running:
            break

        # game step
        if not (terminated or truncated):
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.append(obs)
        else:
            # headless non-interactive: stop once the episode ends
            if not interactive:
                break

        # fps cap (pygame clock only when interactive; otherwise sleep)
        if clock is not None:
            clock.tick(fps)

    env.close()
    if pygame is not None:
        try:
            pygame.quit()
        except Exception:
            pass
    
    obs_list = np.array(obs_list)   

    steps = np.arange(len(obs_list))
    nearest_pipe_x = obs_list[:, 0]
    nearest_gap_top_y = obs_list[:, 1]
    nearest_gap_bottom_y = obs_list[:, 2]
    bird_y = obs_list[:, 9]
    bird_vel_y = obs_list[:, 10]

    gap_middle_y = (nearest_gap_top_y + nearest_gap_bottom_y) / 2

    dy = (gap_middle_y - bird_y - 0.01) / 0.11
    dx = (nearest_pipe_x - 0.1) / 0.3


    plt.plot(steps, dx, label="dx")
    plt.plot(steps, bird_vel_y, label="bird_vel_y")
    plt.plot(steps, dy, label="dy")
    plt.legend()
    plt.show()
# 0: nearest pipe x ([-0.2, 0.4])
# 1: nearest gap top y
# 2: nearest gap bottom y
# 9: bird y ([0, 1])
# 10: bird vel_y ([-0.9, 1])
# 11: bird rotation (not needed)



if __name__ == "__main__":
    main()
