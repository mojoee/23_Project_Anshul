import numpy as np

def data_loader(path=r"data\\PCYL.dat"):

    f=open(path, "rb")
    data=np.genfromtxt(f,  
                    skip_header=1,
                     skip_footer=1,
                     names=True,
                     dtype=None,
                     delimiter=' ')
    f.close()
    print(data)

if __name__=="__main__":
    data_loader()