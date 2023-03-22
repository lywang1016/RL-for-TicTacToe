import pygame

pieces_images = {
    'b_piece': pygame.image.load("img/black.jpg"),
    'r_piece': pygame.image.load("img/white.png")
}

round_imgs = {
    'r_move': pygame.image.load("img/white.png"),
    'b_move': pygame.image.load("img/black.jpg")
}

button_imgs = {
    'reset': pygame.image.load("img/reset.jpg")
}

piece_values = {
    'b_piece': -1,
    'r_piece': 1
}

values_piece = {
    -1: 'b_piece',
    1: 'r_piece'
}

posi_idx_map = {
    (0, 0): 1,
    (0, 1): 2,
    (0, 2): 3,
    (1, 0): 4,
    (1, 1): 5,
    (1, 2): 6,
    (2, 0): 7,
    (2, 1): 8,
    (2, 2): 9
}

idx_rotate_180 = {
    1: (2, 2),
    2: (2, 1),
    3: (2, 0),
    4: (1, 2),
    5: (1, 1),
    6: (1, 0),
    7: (0, 2),
    8: (0, 1),
    9: (0, 0)
}

idx_rotate_lr = {
    1: (0, 2),
    2: (0, 1),
    3: (0, 0),
    4: (1, 2),
    5: (1, 1),
    6: (1, 0),
    7: (2, 2),
    8: (2, 1),
    9: (2, 0)
}

idx_rotate_180lr = {
    1: (2, 0),
    2: (2, 1),
    3: (2, 2),
    4: (1, 0),
    5: (1, 1),
    6: (1, 2),
    7: (0, 0),
    8: (0, 1),
    9: (0, 2)
}