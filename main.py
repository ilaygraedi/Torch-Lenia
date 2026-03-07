import pygame
import engine
import numpy as np
import torch

def main():
    setting = {"width" : 800,
               "height" : 800,
               "inner_radius" : np.array([10, 10, 10]),
               "outer_radius" : np.array([15, 15, 15]),
               "top" : np.array([0.15, 0.15, 0.15]),
               "flex" : np.array([0.015, 0.015, 0.015]),
               "dt" : 0.1,
               "weights":[
                [1.0, 0.4, -0.4], 
                [-0.4, 1.0, 0.3], 
                [0.4, -0.4, 1.0]  
                ],
                "cell":torch.rand(3,300,300),
                "slope" : -4}
     

    lenia1 = engine.Lenia(setting)
    
    pygame.init()
    screen = pygame.display.set_mode((setting["width"] ,setting["height"]))
    running = True
    clock = pygame.time.Clock()
    while running:
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
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
