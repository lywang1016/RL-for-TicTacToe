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