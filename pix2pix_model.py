import torch
from torch import nn


###################################################################################################
# -------------------------------------    Encoder Block   ----------------------------------------
###################################################################################################
class encoder_block(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, apply_norm:bool=True, apply_bias:bool=False):
        super().__init__()

        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=apply_bias),
            nn.InstanceNorm2d(out_channels) if apply_norm else nn.Identity(),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
    
    def forward(self, x:torch.tensor) -> torch.tensor:
        return self.down(x)
    

###################################################################################################
# -------------------------------------    Decoder Block   ----------------------------------------
###################################################################################################
class decoder_block(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, apply_dropout:bool=True):
        super().__init__()

        self.up = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, stride=2, kernel_size=4, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5) if apply_dropout else nn.Identity()
        )

    def forward(self, x:torch.tensor, skip_input):
        x = self.up(x)
        x = torch.cat((x, skip_input), dim=1)
        return x
    

###################################################################################################
# ------------------------------------    Generator Model   ---------------------------------------
###################################################################################################
class Generator(nn.Module):
    def __init__(self, in_channels:int=3, out_channels:int=3, num_filter:int=32):
        super().__init__()

        F = num_filter

        self.E1 = encoder_block(in_channels, F, apply_norm=False, apply_bias=True) # (3, 256, 256) -> (32, 128, 128) 
        self.E2 = encoder_block(F, F*2) # (32, 128, 128) -> (64, 64, 64)
        self.E3 = encoder_block(F*2, F*4) # (64, 64, 64) -> (128, 32, 32)
        self.E4 = encoder_block(F*4, F*8) # (128, 32, 32) -> (256, 16, 16)
        self.E5 = encoder_block(F*8, F*16) # (256, 16, 16) -> (512, 8, 8)

        self.bn = nn.Sequential(
            nn.Conv2d(in_channels=F*16, out_channels=F*16, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        ) # (512, 8, 8) -> (512, 4, 4)

        self.D1 = decoder_block(F*16, F*16) # (512, 4, 4) -> (512, 8, 8) + (512, 8, 8) -> (1024, 8, 8)
        self.D2 = decoder_block(F*32, F*8) # (1024, 8, 8) -> (256, 16, 16) + (256, 16, 16) -> (512, 16, 16)
        self.D3 = decoder_block(F*16, F*4, apply_dropout=False) # (512, 16, 16) -> (128, 32, 32) + (128, 32, 32) -> (256, 32, 32)
        self.D4 = decoder_block(F*8, F*2, apply_dropout=False) # (256, 32, 32) -> (64, 64, 64) + (64, 64, 64) -> (128, 64, 64)
        self.D5 = decoder_block(F*4, F, apply_dropout=False) # (128, 64, 64) -> (32, 128, 128) + (32, 128, 128) -> (64, 128, 128)

        self.output = nn.Sequential(
            nn.ConvTranspose2d(in_channels=F*2, out_channels=out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # 👉👉👉 image should be in range (-1, 1)
        ) # (64, 128, 128) -> (3, 256, 256)

    def forward(self, x:torch.tensor) -> torch.tensor:
        e1 = self.E1(x); e2 = self.E2(e1); e3 = self.E3(e2); e4 = self.E4(e3); e5 = self.E5(e4)

        bn = self.bn(e5)

        d1 = self.D1(bn, e5); d2 = self.D2(d1, e4); d3 = self.D3(d2, e3); d4 = self.D4(d3, e2); d5 = self.D5(d4, e1)

        output = self.output(d5)
        return output    


###################################################################################################
# ----------------------------------    Discriminator Model   -------------------------------------
###################################################################################################
class Discriminator(nn.Module):
    def __init__(self, in_channels:int=3, num_filter:int=32):
        super().__init__()
        F = num_filter

        self.C32 = encoder_block(in_channels=in_channels*2, out_channels=F, 
                                 apply_norm=False, apply_bias=True) # (6, 256, 256) -> (32, 128, 128)
        self.C64 = encoder_block(in_channels=F, out_channels=F*2) # (32, 128, 128) -> (64, 64, 64)
        self.C128 = encoder_block(in_channels=F*2, out_channels=F*4) # (64, 64, 64) -> (128, 32, 32)
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=F*4, out_channels=1, stride=1, padding=1, kernel_size=4, bias=True),
            nn.Sigmoid()
        )
    
    def forward(self, x:torch.tensor, y:torch.tensor) -> torch.tensor:
        x = torch.cat([x,y], dim=1)
        x = self.C32(x); x = self.C64(x); x = self.C128(x)
        return self.output(x)