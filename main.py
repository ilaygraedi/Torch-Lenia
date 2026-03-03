import pygame
import engine
import numpy as np
import torch

def main():
    width = 800
    height = 800
    inner_radius = np.array([10, 10, 10])
    outer_radius = np.array([60, 60, 60])
    top = np.array([0.15, 0.15, 0.15])
    flex = np.array([0.015, 0.015, 0.015])
    dt = 0.1
    weights = [
    [1.0, 0.3, -0.3], 
    [-0.3, 1.0, 0.3], 
    [0.3, -0.3, 1.0]  
    ]
    cell_size = 60
    lenia1 = engine.Lenia(width,height,inner_radius,outer_radius,top,flex,dt,weights)
    
    pygame.init()
    screen = pygame.display.set_mode((width ,height))
    running = True
    clock = pygame.time.Clock()
    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        if pygame.mouse.get_pressed()[0]:
                mouse_pos = pygame.mouse.get_pos()
                x = mouse_pos[1]
                y =mouse_pos[0]
                x = max(cell_size, min(x, height - cell_size))
                y = max(cell_size, min(y, width - cell_size))
                x_start, x_end = x - cell_size, x + cell_size
                y_start, y_end = y - cell_size, y + cell_size
                random = torch.rand(lenia1.channels,cell_size*2,cell_size*2,device=lenia1.device) * 0.5
                lenia1.board[:,x_start :x_end, y_start:y_end] = random
        
        color_mat=(lenia1.board.permute(2,1,0)*255).cpu().numpy().astype(np.uint8)
        surf = pygame.surfarray.make_surface(color_mat)
        lenia1.update_step()
        
        screen.fill((0,0,0))
        screen.blit(surf, (0,0))
        pygame.display.update()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()