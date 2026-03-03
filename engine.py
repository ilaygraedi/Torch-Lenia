import numpy as np
import torch

class Lenia:
    
    def __init__(self,width,height,inner_radius,outer_radius,top,flex,dt,weights):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.width = width
        self.height = height
        self.channels = len(top)
        self.inner_radius = torch.tensor(inner_radius, device=self.device).reshape(self.channels,1,1)
        self.outer_radius = torch.tensor(outer_radius, device=self.device).reshape(self.channels,1,1)
        self.top = torch.tensor(top, device=self.device).reshape(self.channels,1,1)
        self.flex = torch.tensor(flex, device=self.device).reshape(self.channels,1,1)
        self.dt = dt
        self.board = self.create_board()
        kernel = torch.fft.ifftshift(self.create_kernel())
        self.kernel = torch.fft.fft2(kernel)
        self.weights = torch.tensor(weights, device=self.device)
    
    #---יצירת הלוח---
    def create_board(self):
        return torch.zeros((self.channels, self.height, self.width),device=self.device)
    
    #---יצירת ליבה(טבעת) ---
    def create_kernel(self):
        x = torch.arange(-self.width//2, self.width//2,device=self.device)
        y = torch.arange(-self.height//2, self.height//2,device=self.device)
        cor_x, cor_y = torch.meshgrid(x,y,indexing="xy")
        distance = cor_x**2 + cor_y**2
        kernel = ((distance > self.inner_radius**2) & (distance < self.outer_radius**2)).float()
        return kernel
    
    #---חישוב ערך גדילה---
    def calculate_growth(self,mat):#top = mean, felx =standart deviation
        growth = torch.exp(- ((mat - self.top)**2) / (self.flex**2 * 2)) #gaussian
        growth = growth * 2 - 1
        return growth

    #---עדכון הגדילה/דעיכה---
    def update_step(self):
        mat = torch.fft.ifft2(torch.fft.fft2(self.board) * self.kernel).real
        mat = torch.tensordot(self.weights.to(mat.dtype) ,mat, dims=1)
        self.board = torch.clamp(self.calculate_growth(mat)* self.dt + self.board, 0, 1)
        return self.board
    

