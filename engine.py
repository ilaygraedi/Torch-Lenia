import torch

class Lenia:
    
    def __init__(self, settings):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.width = settings["width"]
        self.height = settings["height"]
        self.channels = len(settings["top"])
        self.inner_radius = torch.tensor(settings["inner_radius"], device=self.device).reshape(self.channels,1,1)
        self.outer_radius = torch.tensor(settings["outer_radius"], device=self.device).reshape(self.channels,1,1)
        self.top = torch.tensor(settings["top"], device=self.device).reshape(self.channels,1,1)
        self.flex = torch.tensor(settings["flex"], device=self.device).reshape(self.channels,1,1)
        self.dt = settings["dt"]
        self.slope = settings["slope"]
        self.board = self.create_board()
        kernel = torch.fft.ifftshift(self.create_kernel())
        self.kernel = torch.fft.fft2(kernel)
        self.weights = torch.tensor(settings["weights"], device=self.device)
        self.cell_injection(self.width//2, self.height//2, settings["cell"])
        
    #---יצירת הלוח---
    def create_board(self):
        return torch.zeros((self.channels, self.height, self.width),device=self.device)
    
    #---יצירת ליבה(טבעת) ---
    def create_kernel(self):
        x = torch.arange(-self.width//2, self.width//2,device=self.device)
        y = torch.arange(-self.height//2, self.height//2,device=self.device)
        cor_x, cor_y = torch.meshgrid(x,y,indexing="xy")
        distance = torch.sqrt(cor_x**2 + cor_y**2)
        mid = (self.outer_radius + self.inner_radius)/2
        mid_width = mid - self.inner_radius
        dis_mid = (distance - mid)/mid_width
        normal_dis = ( torch.exp(self.slope * dis_mid ** 2 ))
        kernel = torch.where((dis_mid<1) & (dis_mid>-1), normal_dis, 0)
        return kernel/kernel.sum(dim = (1,2), keepdim = True)
    
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
        self.board = torch.where(self.board< 10**self.slope ,0,self.board)
        return self.board
    
    def cell_injection(self,cor_x,cor_y,mat):
        mid_x = mat.shape[2]//2
        mid_y = mat.shape[1]//2
        start_y = cor_y - mid_y
        start_x = cor_x - mid_x
        self.board[:,start_y:start_y + mat.shape[1], start_x:start_x + mat.shape[2]] = mat
