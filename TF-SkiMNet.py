import torch
import torch.nn as nn
import math
from torch import Tensor
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
import os

class LayerNorm(nn.LayerNorm):

    def __init__(self, seq_last: bool, **kwargs) -> None:
        """
        Arg s:
            seq_last (bool): whether the sequence dim is the last dim
        """
        super().__init__(**kwargs)
        self.seq_last = seq_last

    def forward(self, input: Tensor) -> Tensor:
        if self.seq_last:
            input = input.transpose(-1, 1)  # [B, H, Seq] -> [B, Seq, H], or [B,H,w,h] -> [B,h,w,H]
        o = super().forward(input)
        if self.seq_last:
            o = o.transpose(-1, 1)
        return o
    
class Causal_Conv1d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,groups):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(in_channels,out_channels,kernel_size=kernel_size,padding=kernel_size-1,groups=groups)
    def forward(self,x):
        x = self.conv(x)
        x = x[:,:,:-(self.kernel_size-1)]
        return x
    
class TFSkiMNet(nn.Module):
    def __init__(
            self,
            blocks: int,
            in_channel: int,
            out_channel: int,
            hidden_size: int,
            encoder_kernel_size: int,
            dim_t_local: int,
            kernel_t_loacl:int,
            global_feature_type: str,
            with_h_net: bool,
            with_c_net: bool,
            with_global_t: bool,
            with_loacl_t: bool
    ):
        super().__init__()
        assert global_feature_type in ['c','h','ch','X','0']
        self.blocks = blocks
        self.hidden_size = hidden_size
        self.encoder_kernel_size = encoder_kernel_size
        self.dim_t_loacl = dim_t_local
        self.kernel_t_loacl = kernel_t_loacl
        self.global_feature_type = global_feature_type

        self.with_h_net = with_h_net
        self.with_c_net = with_c_net
        self.with_global_t = with_global_t
        self.with_loacl_t = with_loacl_t

        self.encoder = nn.Conv1d(in_channel,hidden_size,encoder_kernel_size,1,padding=encoder_kernel_size-1)

        self.f_local_modules = nn.ModuleList([])
        self.f_modules = nn.ModuleList([])
        self.local_t_moduels = nn.ModuleList([])
        self.c_net = nn.ModuleList([])
        self.h_net = nn.ModuleList([])
        for i in range(self.blocks):

            self.f_local_modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_size,hidden_size,(1,5),stride=(1,1),padding=(0,2)),
                    nn.SiLU()
                )
            )

            self.f_modules.append(
                nn.ModuleDict(
                    {
                        'ln': LayerNorm(normalized_shape=self.hidden_size,seq_last=False),
                        'lstm': nn.LSTM(batch_first=True,bidirectional=True,input_size=self.hidden_size,hidden_size=self.hidden_size),
                        'proj': nn.Linear(self.hidden_size*2,self.hidden_size)
                    }
                )
            )
            
            if self.with_loacl_t:
                self.local_t_moduels.append(
                    nn.Sequential(
                        LayerNorm(normalized_shape=self.hidden_size,seq_last=True),
                        Causal_Conv1d(dim_t_local,dim_t_local,kernel_size=self.kernel_t_loacl,groups=2),
                        nn.SiLU(),
                    )
                )

            if i < self.blocks-1:
                if self.with_c_net:
                    self.c_net.append(
                        nn.ModuleDict(
                            {
                                'lstm': nn.LSTM(batch_first=True,bidirectional=False,input_size=2*self.hidden_size,hidden_size=2*self.hidden_size),
                                'ln': LayerNorm(normalized_shape=self.hidden_size*2,seq_last=False),
                            }
                        )
                    )

                if self.with_h_net:
                    self.h_net.append(
                        nn.ModuleDict(
                            {
                                'lstm': nn.LSTM(batch_first=True,bidirectional=False,input_size=2*self.hidden_size,hidden_size=2*self.hidden_size),
                                'ln': LayerNorm(normalized_shape=self.hidden_size*2,seq_last=False),
                            }
                        )
                    )


        self.decoder = nn.Conv1d(hidden_size,out_channel,encoder_kernel_size,1,padding=encoder_kernel_size-1)
    
    def f_moudle_local_forward(self,X,i):
        '''
            X: [B,F,T,H]
        '''
        B,F,T,H = X.shape
        back_up = X

        X = X.permute(0,3,2,1) #B,C,T,F
        X = self.f_local_modules[i](X)
        X = X.permute(0,3,2,1)

        X = X + back_up
        return X
    
    def f_moudle_global_forward(self,X,i,hc):
        '''
            X: [B,F,T,H]
        '''

        B,F,T,H = X.shape

        if hc is None:
            h = torch.zeros(2,B*T,self.hidden_size).to(X.device)
            c = torch.zeros(2,B*T,self.hidden_size).to(X.device)
        else:
            h,c = hc

        back_up = X

        X = X.permute(0,2,1,3) #[B,T,F,H]
        X = X.reshape(B*T,F,H)
        X = self.f_modules[i]['ln'](X)
        X,(h,c) = self.f_modules[i]['lstm'](X,(h,c)) # h:B,T,C
        X = self.f_modules[i]['proj'](X)
        X = X.reshape(B,T,F,H).permute(0,2,1,3) #[B,F,T,H]

        X = X + back_up

        return X,(h,c)
    
    def local_t_forward(self,X,i):
        '''
            X: [B,F,T,H]
        '''

        B,F,T,H = X.shape
        back_up = X

        X = X.reshape(B*F,T,H).transpose(1,2) #[B*F,H,T]
        X = self.local_t_moduels[i](X)
        X = X.reshape(B,F,H,T).transpose(2,3) #[B,F,T,H]

        X = back_up + X

        return X
    
    def c_net_forward(self,input,i):
        B,T,D = input.shape
        output,_ = self.c_net[i]['lstm'](input)
        output = self.c_net[i]['ln'](output)
        output = output + input
        return output
    
    def h_net_forward(self,input,i):
        B,T,D = input.shape
        output,_ = self.h_net[i]['lstm'](input)
        output = self.h_net[i]['ln'](output)
        output = output + input
        return output
    
    def forward(self,X):
        '''
         X:[B,F,T,H]
        '''
        B,F,T,H = X.shape
        X = self.encoder(X.reshape(B*F,T,H).permute(0, 2, 1)).permute(0,2,1)[:,:-(self.encoder_kernel_size-1),:]
        H = X.shape[2]
        X = X.reshape(B,F,T,H)

        hc = None
        for i in range(self.blocks):
            # F Module
            X = self.f_moudle_local_forward(X,i)
            X,(h,c) = self.f_moudle_global_forward(X,i,hc)
            # Trans h and c
            h = h.transpose(0,1).reshape(B,T,2,H).reshape(B,T,2*H)
            c = c.transpose(0,1).reshape(B,T,2,H).reshape(B,T,2*H)
            # Local T
            if self.with_loacl_t:
                X = self.local_t_forward(X,i)
            # Get Global State
            if self.with_global_t:
                if i < self.blocks-1:
                    if self.with_h_net:
                        h = self.h_net_forward(h,i) # B,T,D
                    if self.with_c_net:
                        c = self.c_net_forward(c,i) # B,T,D
                    h = h.reshape(B*T,2,-1).permute(1,0,2).contiguous()
                    c = c.reshape(B*T,2,-1).permute(1,0,2).contiguous()
                    hc = (h,c)
            else:
                hc = None

        X = self.decoder(X.reshape(B*F,T,H).permute(0, 2, 1)).permute(0,2,1)[:,:-(self.encoder_kernel_size-1),:]
        H = X.shape[2]
        X = X.reshape(B,F,T,H)

        return X
    
if __name__ == '__main__':
    from thop import profile,clever_format
    model = TFSkiMNet(
        blocks=6,
        in_channel=2,
        out_channel=2,
        hidden_size=32,
        encoder_kernel_size=5,
        dim_t_local=32,
        kernel_t_loacl=5,
        global_feature_type='ch',
        with_h_net=True,
        with_c_net=True,
        with_global_t=True,
        with_loacl_t=True
    ).cuda(9)

    X = torch.rand(1,161,101,2).cuda(9)

    flops, params = profile(model,inputs=(X,))
    flops, params = clever_format([flops, params], "%.3f")
    print(flops)
    print(params)