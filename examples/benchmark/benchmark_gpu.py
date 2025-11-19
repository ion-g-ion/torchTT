import torchtt as tntt 
import torch as tn 
import torch.nn as nn
import sys 

dev = 'cuda' if tn.cuda.is_available() else None 
device_name = tn.cuda.get_device_name() if tn.cuda.is_available() else "CPU num_threads %d"%(tn.get_num_threads())
try:
    additional_info = sys.argv[sys.argv.index("--info")+1] 
except:
    additional_info = ""
    
def full2tt(n, d, r, dtype=tn.float64):
    """ Benchmark full to tt for a `[2]*d` tensor with maximal rank `r`."""
    x = tntt.randn([n]*d, [1]+[r]*(d-1)+[1], dtype=dtype).full().to(dev)
    with tntt.Timer("full2tt(%d,%d,%d,%s)"%(n,d,r,str(dtype))) as t:
        x_tt = tntt.TT(x, rmax=r)
        if dev != 'cpu': tn.cuda.synchronize(dev)
    return t.interval

def round(n, d, r1, r2, dtype=tn.float64):
    x = tntt.randn([n]*d, [1]+[r1]*(d-1)+[1], dtype=dtype).to(dev)
    x = x/x.norm()
    y = tntt.randn([n]*d, [1]+[r2]*(d-1)+[1], dtype=dtype).to(dev)
    y = y/y.norm() * 1e-6
    x += y
    with tntt.Timer("round(%d,%d,%d,%d,%s)"%(n,d,r1,r2,str(dtype))) as t:
        x = x.round(1e-3)
        if dev != 'cpu': tn.cuda.synchronize(dev)
    return t.interval
    
def tt_layers(batch_size, dtype=tn.float32):
    """ Benchmark the TT layers. """
    class BasicTT(nn.Module):
        def __init__(self):
            super().__init__()
            self.ttl1 = tntt.nn.LinearLayerTT([16,16,16,16], [8,8,8,8], [1,3,3,3,1], dtype=dtype)
            self.ttl2 = tntt.nn.LinearLayerTT([8,8,8,8], [4,4,4,4], [1,2,2,2,1], dtype=dtype)
            self.ttl3 = tntt.nn.LinearLayerTT([4,4,4,4], [2,4,2,4], [1,2,2,2,1], dtype=dtype)
            self.linear = nn.Linear(64, 10, dtype = dtype)

        def forward(self, x):
            x = self.ttl1(x)
            x = tn.relu(x)
            x = self.ttl2(x)
            x = tn.relu(x)
            x = self.ttl3(x)
            x = tn.relu(x)
            x = tn.reshape(x,[-1,64])
            return self.linear(x)
    
    model = BasicTT().to(dev)
    
    input_batch = tn.rand((10,16,16,16,16), dtype=dtype).to(dev)
    y = model(input_batch)
    input_batch = tn.rand((batch_size,16,16,16,16), dtype=dtype).to(dev)

    with tntt.Timer("tt_layers(%d,%s)"%(batch_size, str(dtype))) as t:
        y = model(input_batch)
        if dev != 'cpu': tn.cuda.synchronize(dev)
        

    return t.interval
    
    
if __name__=="__main__":
    print()
    print("Device name:", device_name)
    print("Additional info:", additional_info)
    
    N = 8
    
    times = []

    # decomposition
    v = min([full2tt(2,10,100, tn.float64) for _ in range(N)])
    times.append(v)
    v = min([full2tt(2,10,100, tn.complex128) for _ in range(N)])
    times.append(v)
    v = min([full2tt(100,3,100, tn.float64) for _ in range(N)])
    times.append(v)
    v = min([round(50, 4, 10, 1000, tn.float64) for _ in range(N)])
    times.append(v)
    v = min([round(50, 4, 100, 100, tn.float64) for _ in range(N)])
    times.append(v)

    # TT layers
    v = min([tt_layers(1000, tn.float32) for _ in range(N)])
    times.append(v)
    v = min([tt_layers(1000, tn.float64) for _ in range(N)])
    times.append(v)

    
    
    print(times)
    
    
    