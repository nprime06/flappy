#!/usr/bin/env python3
import gymnasium as gym
import flappy_bird_gymnasium  # registers FlappyBird-* env ids with gymnasium
import random
import time


def main():
    env_id = "FlappyBird-v0" # FlappyBird-v0 or FlappyBird-rgb-v0
    interactive = False
    p = 0.2  # when interactive=False: probability of taking action=1 (flap)
    seed = 0  # RNG seed for reproducible random policy
    fps = 30

    # Headless: render_mode=None means no window / no video subsystem.
    # If you want visuals, switch to render_mode="human" or "rgb_array".
    render_mode = None
    env = gym.make(env_id, render_mode=render_mode, disable_env_checker=True, use_lidar=False)
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

    step = 0
    while running:
        step += 1
        print(f"Step {step}")
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


if __name__ == "__main__":
    main()
