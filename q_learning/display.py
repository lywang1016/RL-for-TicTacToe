import sys
import math
import pygame
from heapq import heapify, heappop, heappush
from constant import pieces_images, values_piece, round_imgs, button_imgs

class GUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((560, 400))
        self.background_img = pygame.image.load("img/board.png")
        self.side_img = pygame.image.load("img/bg.jpg")
        self.pixel_dif = 120
        pixel_init_row = 30
        pixel_init_col = 30
        self.piece_size = 100
        self.positions = []
        for i in range(3):
            temp = []
            for j in range(3):
                temp.append((pixel_init_col+j*self.pixel_dif, pixel_init_row+i*self.pixel_dif))
            self.positions.append(temp)
    
    def update(self, board, round):
        self.screen.blit(self.side_img, (0, 0))
        self.screen.blit(self.side_img, (400, 0))
        self.screen.blit(self.background_img, (20, 20))
        self.screen.blit(button_imgs['reset'], (400, 180))

        if round == 'r':
            self.screen.blit(round_imgs['r_move'], (425, 40))
        else:
            self.screen.blit(round_imgs['b_move'], (425, 260))     

        for i in range(3):
            for j in range(3):
                if board[i][j] != 0:
                    self.screen.blit(pieces_images[values_piece[board[i][j]]], self.positions[i][j])

        pygame.display.update()
    
    def check_event(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()  
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                mouse_x = pos[0]-math.ceil(self.pixel_dif/2)
                mouse_y = pos[1]-math.ceil(self.pixel_dif/2)
                if mouse_x < 320 and mouse_y < 320:  
                    queue = []
                    heapify(queue)
                    for i in range(3):
                        for j in range(3):
                            center_x = self.positions[i][j][0]
                            center_y = self.positions[i][j][1]
                            dis = (center_x - mouse_x)**2 + (center_y - mouse_y)**2
                            heappush(queue, (dis, (i, j)))
                    dis, posi = heappop(queue)
                    return 'grid', posi
                elif mouse_x < 490 and mouse_x > 340 and mouse_y < 160 and mouse_y > 120:
                    return 'reset', (-1, -1)
                else:
                    print(mouse_x)
                    print(mouse_y)
                    return 'none', (-1, -1)
        return 'none', (-1, -1)