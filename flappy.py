import random
import sys

import numpy as np
import tensorflow as tf

import pygame
from pygame.locals import *

from itertools import cycle

FPS = 60
SCREENWIDTH = 288.0
SCREENHEIGHT = 512.0

PIPEGAPSIZE = 100
BASEY = SCREENHEIGHT * 0.79

IMAGES, SOUNDS, HITMASKS = {}, {}, {}

load_saved_pool = 1
save_current_pool = 1
current_pool = []
fitness = []
total_models = 50

next_pipe_x = -1
next_pipe_y = -1
generation = 1


def save_pool():
    for xi in range(total_models):
        current_pool[xi].save_weights("current_model_pool/model_new" + str(xi) + ".keras")
    print("Saved current pool!")


def model_crossover(model_idx1, model_idx2):
    global current_pool
    weights1 = current_pool[model_idx1].get_weights()
    weights2 = current_pool[model_idx2].get_weights()
    weightsnew1 = weights1
    weightsnew2 = weights2
    weightsnew1[0] = weights2[0]
    weightsnew2[0] = weights1[0]
    return np.asarray([weightsnew1, weightsnew2])


def model_mutate(weights):
    for xi in range(len(weights)):
        for yi in range(len(weights[xi])):
            if random.uniform(0, 1) > 0.85:
                change = random.uniform(-0.5, 0.5)
                weights[xi][yi] += change
    return weights


def predict_action(height, dist, pipe_height, model_num):
    global current_pool
    height = min(SCREENHEIGHT, height) / SCREENHEIGHT - 0.5
    dist = dist / 450 - 0.5
    pipe_height = min(SCREENHEIGHT, pipe_height) / SCREENHEIGHT - 0.5
    neural_input = np.asarray([height, dist, pipe_height])
    neural_input = np.atleast_2d(neural_input)
    output_prob = current_pool[model_num](neural_input, 1)[0]
    if output_prob[0] <= 0.5:
        return 1
    return 2


for i in range(total_models):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(7, input_dim=3))
    model.add(tf.keras.layers.Activation("sigmoid"))
    model.add(tf.keras.layers.Dense(1))
    model.add(tf.keras.layers.Activation("sigmoid"))

    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.01,
        decay_steps=10000,
        decay_rate=1e-6)
    sgd = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
    model.compile(loss="mse", optimizer=sgd, metrics=["accuracy"])
    current_pool.append(model)
    fitness.append(-100)

if load_saved_pool:
    for i in range(total_models):
        current_pool[i].load_weights("Current_Model_Pool/model_new" + str(i) + ".keras")

for i in range(total_models):
    print(current_pool[i].get_weights())

PLAYERS_LIST = (
    (
        'assets/sprites/redbird-upflap.png',
        'assets/sprites/redbird-midflap.png',
        'assets/sprites/redbird-downflap.png',
    ),
    (
        'assets/sprites/bluebird-upflap.png',
        'assets/sprites/bluebird-midflap.png',
        'assets/sprites/bluebird-downflap.png',
    ),
    (
        'assets/sprites/yellowbird-upflap.png',
        'assets/sprites/yellowbird-midflap.png',
        'assets/sprites/yellowbird-downflap.png',
    ),
)

BACKGROUNDS_LIST = (
    'assets/sprites/background-day.png',
    'assets/sprites/background-night.png',
)

PIPES_LIST = (
    'assets/sprites/pipe-green.png',
    'assets/sprites/pipe-red.png',
)


def main():
    global SCREEN, FPSCLOCK
    pygame.init()
    FPSCLOCK = pygame.time.Clock()
    SCREEN = pygame.display.set_mode((int(SCREENWIDTH), int(SCREENHEIGHT)))
    pygame.display.set_caption('Flappy Bird')

    IMAGES['numbers'] = (
        pygame.image.load('assets/sprites/0.png').convert_alpha(),
        pygame.image.load('assets/sprites/1.png').convert_alpha(),
        pygame.image.load('assets/sprites/2.png').convert_alpha(),
        pygame.image.load('assets/sprites/3.png').convert_alpha(),
        pygame.image.load('assets/sprites/4.png').convert_alpha(),
        pygame.image.load('assets/sprites/5.png').convert_alpha(),
        pygame.image.load('assets/sprites/6.png').convert_alpha(),
        pygame.image.load('assets/sprites/7.png').convert_alpha(),
        pygame.image.load('assets/sprites/8.png').convert_alpha(),
        pygame.image.load('assets/sprites/9.png').convert_alpha()
    )

    IMAGES['gameover'] = pygame.image.load('assets/sprites/gameover.png').convert_alpha()
    IMAGES['message'] = pygame.image.load('assets/sprites/message.png').convert_alpha()
    IMAGES['base'] = pygame.image.load('assets/sprites/base.png').convert_alpha()
    IMAGES['generation'] = pygame.image.load('assets/sprites/generation.png').convert_alpha()
    IMAGES['species'] = pygame.image.load('assets/sprites/species.png').convert_alpha()

    if 'win' in sys.platform:
        sound_ext = '.wav'
    else:
        sound_ext = '.ogg'

    SOUNDS['die'] = pygame.mixer.Sound('assets/audio/die' + sound_ext)
    SOUNDS['hit'] = pygame.mixer.Sound('assets/audio/hit' + sound_ext)
    SOUNDS['point'] = pygame.mixer.Sound('assets/audio/point' + sound_ext)
    SOUNDS['swoosh'] = pygame.mixer.Sound('assets/audio/swoosh' + sound_ext)
    SOUNDS['wing'] = pygame.mixer.Sound('assets/audio/wing' + sound_ext)

    while True:
        rand_bg = random.randint(0, len(BACKGROUNDS_LIST) - 1)
        IMAGES['background'] = pygame.image.load(BACKGROUNDS_LIST[rand_bg]).convert()

        rand_player = random.randint(0, len(PLAYERS_LIST) - 1)
        IMAGES['player'] = (
            pygame.image.load(PLAYERS_LIST[rand_player][0]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[rand_player][1]).convert_alpha(),
            pygame.image.load(PLAYERS_LIST[rand_player][2]).convert_alpha(),
        )

        pipe_index = random.randint(0, len(PIPES_LIST) - 1)
        IMAGES['pipe'] = (
            pygame.transform.rotate(pygame.image.load(PIPES_LIST[pipe_index]).convert_alpha(), 180),
            pygame.image.load(PIPES_LIST[pipe_index]).convert_alpha(),
        )

        HITMASKS['pipe'] = (
            hitmask(IMAGES['pipe'][0]),
            hitmask(IMAGES['pipe'][1]),
        )
        HITMASKS['player'] = (
            hitmask(IMAGES['player'][0]),
            hitmask(IMAGES['player'][1]),
            hitmask(IMAGES['player'][2]),
        )

        move_info = welcome()
        global fitness
        for idx in range(total_models):
            fitness[idx] = 0
        crash_info = game(move_info)
        game_over(crash_info)


def welcome():
    return {
        'playery': int((SCREENHEIGHT - IMAGES['player'][0].get_height()) / 2),
        'basex': 0,
        'player_num_gen': cycle([0, 1, 2, 1]),
    }


def game(move_info):
    global fitness
    score = player_num = loop_iter = 0
    player_num_gen = move_info['player_num_gen']
    players_x = []
    players_y = []
    for i in range(total_models):
        playerx, playery = int(SCREENWIDTH * 0.2), move_info['playery']
        players_x.append(playerx)
        players_y.append(playery)
    basex = move_info['basex']
    base_shift = IMAGES['base'].get_width() - IMAGES['background'].get_width()

    pipe1 = random_pipe()
    pipe2 = random_pipe()

    upper_pipes = [
        {'x': SCREENWIDTH + 200, 'y': pipe1[0]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': pipe2[0]['y']},
    ]
    lower_pipes = [
        {'x': SCREENWIDTH + 200, 'y': pipe1[1]['y']},
        {'x': SCREENWIDTH + 200 + (SCREENWIDTH / 2), 'y': pipe2[1]['y']},
    ]

    global next_pipe_x
    global next_pipe_y

    next_pipe_x = lower_pipes[0]['x']
    next_pipe_y = (lower_pipes[0]['y'] + (upper_pipes[0]['y'] + IMAGES['pipe'][0].get_height())) / 2

    pipe_vel_x = -4
    players_vel_y = []
    player_max_vel_y = 10
    players_acc_y = []
    player_flap_acc = -9
    players_flapped = []
    players_state = []

    for i in range(total_models):
        players_vel_y.append(-9)
        players_acc_y.append(1)
        players_flapped.append(False)
        players_state.append(True)

    alive_players = total_models

    while True:
        for i in range(total_models):
            if players_y[i] < 0 and players_state[i] is True:
                alive_players -= 1
                players_state[i] = False
        if alive_players == 0:
            return {
                'y': 0,
                'ground_crash': True,
                'basex': basex,
                'upper_pipes': upper_pipes,
                'lower_pipes': lower_pipes,
                'score': score,
                'player_vel_y': 0,
            }
        for i in range(total_models):
            if players_state[i]:
                fitness[i] += 1
        next_pipe_x += pipe_vel_x
        for i in range(total_models):
            if players_state[i]:
                if predict_action(players_y[i], next_pipe_x, next_pipe_y, i) == 1:
                    if players_y[i] > -2 * IMAGES['player'][0].get_height():
                        players_vel_y[i] = player_flap_acc
                        players_flapped[i] = True
                        SOUNDS['wing'].play()
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                sys.exit()

        crash_test = check_crash({'x': players_x, 'y': players_y, 'index': player_num}, upper_pipes, lower_pipes)

        for i in range(total_models):
            if players_state[i] is True and crash_test[i] is True:
                alive_players -= 1
                players_state[i] = False
        if alive_players == 0:
            return {
                'y': playery,
                'ground_crash': crash_test[1],
                'basex': basex,
                'upper_pipes': upper_pipes,
                'lower_pipes': lower_pipes,
                'score': score,
                'player_vel_y': 0,
            }
        hit = False
        for i in range(total_models):
            if players_state[i] is True:
                pipe_i = 0
                hit = False
                player_mid_pos = players_x[i]
                for pipe in upper_pipes:
                    pipe_mid_pos = pipe['x'] + IMAGES['pipe'][0].get_width()
                    if pipe_mid_pos <= player_mid_pos < pipe_mid_pos + 4:
                        next_pipe_x = lower_pipes[pipe_i + 1]['x']
                        next_pipe_y = (lower_pipes[pipe_i + 1]['y'] +
                                       (upper_pipes[pipe_i + 1]['y'] + IMAGES['pipe'][pipe_i + 1].get_height())) / 2
                        hit = True
                        fitness[i] += 25
                    pipe_i += 1
        if hit:
            score += 1
            SOUNDS['point'].play()

        if (loop_iter + 1) % 3 == 0:
            player_num = next(player_num_gen)
        loop_iter = (loop_iter + 1) % 30
        basex = -((-basex + 100) % base_shift)

        for i in range(total_models):
            if players_state[i] is True:
                if players_vel_y[i] < player_max_vel_y and not players_flapped[i]:
                    players_vel_y[i] += players_acc_y[i]
                if players_flapped[i]:
                    players_flapped[i] = False
                player_height = IMAGES['player'][player_num].get_height()
                players_y[i] += min(players_vel_y[i], BASEY - players_y[i] - player_height)

        for upper_pipe, lower_pipe in zip(upper_pipes, lower_pipes):
            upper_pipe['x'] += pipe_vel_x
            lower_pipe['x'] += pipe_vel_x

        if 0 < upper_pipes[0]['x'] < 5:
            pipe = random_pipe()
            upper_pipes.append(pipe[0])
            lower_pipes.append(pipe[1])

        if upper_pipes[0]['x'] < -IMAGES['pipe'][0].get_width():
            upper_pipes.pop(0)
            lower_pipes.pop(0)

        SCREEN.blit(IMAGES['background'], (0, 0))
        for upper_pipe, lower_pipe in zip(upper_pipes, lower_pipes):
            SCREEN.blit(IMAGES['pipe'][0], (upper_pipe['x'], upper_pipe['y']))
            SCREEN.blit(IMAGES['pipe'][1], (lower_pipe['x'], lower_pipe['y']))
        SCREEN.blit(IMAGES['base'], (basex, BASEY))

        show(score, 'score')

        alive = total_models
        for i in range(total_models):
            if players_state[i] is True:
                SCREEN.blit(IMAGES['player'][player_num], (players_x[i], players_y[i]))
            else:
                alive -= 1

        show(alive, 'alive')
        SCREEN.blit(IMAGES["species"], (10, 420))

        show(generation, 'generation')
        SCREEN.blit(IMAGES["generation"], (185, 420))

        pygame.display.update()
        FPSCLOCK.tick(FPS)


def game_over(crash_info):
    global current_pool
    global fitness
    global generation
    weights = []
    total_fitness = 0

    for select in range(total_models):
        total_fitness += fitness[select]
    for select in range(total_models):
        fitness[select] /= total_fitness
        if select > 0:
            fitness[select] += fitness[select - 1]
    for select in range(int(total_models / 2)):
        parent1 = random.uniform(0, 1)
        parent2 = random.uniform(0, 1)
        i1 = -1
        i2 = -1
        for i in range(total_models):
            if fitness[i] >= parent1:
                i1 = i
                break
        for i in range(total_models):
            if fitness[i] >= parent2:
                i2 = i
                break
        new_weights = model_crossover(i1, i2)
        updated_weights1 = model_mutate(new_weights[0])
        updated_weights2 = model_mutate(new_weights[1])
        weights.append(updated_weights1)
        weights.append(updated_weights2)
    for select in range(len(weights)):
        fitness[select] = -100
        current_pool[select].set_weights(weights[select])
    if save_current_pool == 1:
        save_pool()
    generation = generation + 1


def random_pipe():
    gap_y = random.randrange(0, int(BASEY * 0.6 - PIPEGAPSIZE))
    gap_y += int(BASEY * 0.2)
    pipe_height = IMAGES['pipe'][0].get_height()
    pipe_x = SCREENWIDTH + 10

    return [
        {'x': pipe_x, 'y': gap_y - pipe_height},
        {'x': pipe_x, 'y': gap_y + PIPEGAPSIZE},
    ]


def show(num, entity):
    digits = [int(x) for x in list(str(num))]
    total_width = 0

    for digit in digits:
        total_width += IMAGES['numbers'][digit].get_width()

    if entity == 'score':
        xoffset = (SCREENWIDTH - total_width) / 2
        height = SCREENHEIGHT * 0.1
    elif entity == 'alive':
        xoffset = SCREENWIDTH * 0.1
        height = SCREENHEIGHT * 0.9
    elif entity == 'generation':
        xoffset = SCREENWIDTH * 0.8
        height = SCREENHEIGHT * 0.9
    else:
        return

    for digit in digits:
        SCREEN.blit(IMAGES['numbers'][digit], (xoffset, height))
        xoffset += IMAGES['numbers'][digit].get_width()


def check_crash(players, upper_pipes, lower_pipes):
    statuses = []
    for i in range(total_models):
        statuses.append(False)

    for i in range(total_models):
        statuses[i] = False
        player = players['index']
        players['w'] = IMAGES['player'][0].get_width()
        players['h'] = IMAGES['player'][0].get_height()

        if players['y'][i] + players['h'] >= BASEY - 1:
            statuses[i] = True
        player_rect = pygame.Rect(players['x'][i], players['y'][i], players['w'], players['h'])
        pipe_width = IMAGES['pipe'][0].get_width()
        pipe_height = IMAGES['pipe'][0].get_height()

        for upper_pipe, lower_pipe in zip(upper_pipes, lower_pipes):
            upper_pipe_rect = pygame.Rect(upper_pipe['x'], upper_pipe['y'], pipe_width, pipe_height)
            lower_pipe_rect = pygame.Rect(lower_pipe['x'], lower_pipe['y'], pipe_width, pipe_height)

            player_hitmask = HITMASKS['player'][player]
            upper_pipe_hitmask = HITMASKS['pipe'][0]
            lower_pipe_hitmask = HITMASKS['pipe'][1]

            upper_collision = pixel_collision(player_rect, upper_pipe_rect, player_hitmask, upper_pipe_hitmask)
            lower_collision = pixel_collision(player_rect, lower_pipe_rect, player_hitmask, lower_pipe_hitmask)

            if upper_collision or lower_collision:
                statuses[i] = True
    return statuses


def pixel_collision(rect1, rect2, hitmask1, hitmask2):
    rect = rect1.clip(rect2)

    if rect.width == 0 or rect.height == 0:
        return False

    x1, y1 = rect.x - rect1.x, rect.y - rect1.y
    x2, y2 = rect.x - rect2.x, rect.y - rect2.y

    for x in range(rect.width):
        for y in range(rect.height):
            if hitmask1[x1 + x][y1 + y] and hitmask2[x2 + x][y2 + y]:
                return True
    return False


def hitmask(image):
    mask = []
    for x in range(image.get_width()):
        mask.append([])
        for y in range(image.get_height()):
            mask[x].append(bool(image.get_at((x, y))[3]))
    return mask


if __name__ == '__main__':
    main()
