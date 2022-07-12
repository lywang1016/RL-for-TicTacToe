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

action2posi = {
    0: (0,0),
    1: (0,1),
    2: (0,2),
    3: (1,0),
    4: (1,1),
    5: (1,2),
    6: (2,0),
    7: (2,1),
    8: (2,2)
}