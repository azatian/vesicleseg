# %%
import subprocess

#This part is for making all the directory under cutouts/
for i in range(415):
    cmd = "mkdir " + "cutouts/dcvsoma" + str(i+1)
    _temp = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    output = _temp.communicate()[0]
# %%
#This part is for making the subdirectories under cutouts/dcvsynx

for i in range(415):
    cmd = "mkdir " + "cutouts/dcvsoma" + str(i+1) + "/img" + " cutouts/dcvsoma" + str(i+1) + "/cellseg" 
    _temp = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    output = _temp.communicate()[0]

# %%
for i in range(415):
    cmd = "python3 wk.py dcvsoma" + str(i+1)
    _temp = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    output = _temp.communicate()[0]

# %%
