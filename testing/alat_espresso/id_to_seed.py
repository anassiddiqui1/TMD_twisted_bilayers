import sys
from glob import glob
idnum= int(sys.argv[1])
files = sorted(glob(f'*.pwi'))
#files = ['bilayer_md.pwi','monolayer1.pwi','monolayer2.pwi']
print(files[idnum][:-4])
