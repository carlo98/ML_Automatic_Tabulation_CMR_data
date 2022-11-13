#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np

# 0 dead, 1 ice, 2 normal


# In[ ]:


id_instance = 1
while id_instance <= 25:
    n = np.random.randint(6, 10)
    num_blocks = np.random.randint(1, 4 if n>10 else 2)
    grid = [[0 for c in range(n)] for r in range(n)]  # initializing grid of dead cells
    
    # Start positions for agent, blocks and targets
    pos_agent = (np.random.randint(1, n-1), np.random.randint(1, n-1))
    grid[pos_agent[0]][pos_agent[1]] = 2  # normal
    pos_blocks, pos_targets = [], []
    for id_block in range(num_blocks):
        pos_block = (np.random.randint(2, n-2), np.random.randint(2, n-2))  # avoid blocks near borders
        while pos_block == pos_agent or pos_block in pos_blocks:  # Avoid sovrappositions
            pos_block = (np.random.randint(2, n-2), np.random.randint(2, n-2))  # avoid blocks near borders
        pos_blocks.append(pos_block)
        pos_target = (np.random.randint(1, n-1), np.random.randint(1, n-1))
        while pos_target == pos_block:  # Avoid easy win
            pos_target = (np.random.randint(1, n-1), np.random.randint(1, n-1))
        pos_targets.append(pos_target)
        grid[pos_target[0]][pos_target[1]] = 2  # normal
        grid[pos_block[0]][pos_block[1]] = 2  # normal
        
    # solve with a relaxed version of the model
    filename = "bombastic/Bombastic_Bombastic"+str(id_instance)+".param"
    with open(filename, "w") as f:
        f.write("language ESSENCE' 1.0\n")
        f.write("letting avatarInitRow be "+str(pos_agent[0]+1)+"\n")
        f.write("letting avatarInitCol be "+str(pos_agent[1]+1)+"\n")
        f.write("letting numBlocks be "+str(num_blocks)+"\n")
        f.write("letting blocksInitRow be "+str([x[0]+1 for x in pos_blocks])+"\n")
        f.write("letting blocksInitCol be "+str([x[1]+1 for x in pos_blocks])+"\n")
        f.write("letting blocksGoalRow be "+str([x[0]+1 for x in pos_targets])+"\n")
        f.write("letting blocksGoalCol be "+str([x[1]+1 for x in pos_targets])+"\n")
        f.write("letting gridInit be ["+str(list(grid[0]))+",\n")
        for i in range(1, len(grid)-1):
            f.write("                     "+str(list(grid[i]))+",\n")
        f.write("                     "+str(list(grid[-1]))+"]\n")
        
        # Service parameters, to be removed in final param file
        f.write("letting upper_bound_changes be "+str(min(int(0.33*(n-1)*(n-1)), 30))+"\n")
        f.write("letting upper_bound_steps be "+str(int(0.33*(n-1)*(n-1))))
    
    os.system("savilerow-main/savilerow -timelimit 600 -solver-options -cpulimit_3600 bombastic/Bombastic_relaxed.eprime %s -run-solver -out-solution bombastic/s0" % (filename))
    
    flag = False
    if not os.path.exists(filename+".info"):
        continue

    with open(filename+".info", "r") as f:
        for line in f.readlines():
            if "SolverSolutionsFound:0" in line or "SolverTimeOut:1" in line:
                flag = True
                break
    if flag:
        continue
        
    id_instance += 1
    flag_grid = False
    with open("bombastic/s0", "r") as f:
        lines = f.readlines()
    for line in lines:
        if "step" in line:
            steps = int(line.split(" ")[-1])
        if flag_grid:
            if "]]" in line:
                flag_grid = False
            grid.append([int(x) for x in line if x!="[" and x != "," and x!="]" and x!=" " and x!="\n"])
        if "gridCurrent" in line:
            grid = [[int(x) for x in line.split("be ")[-1] if x!="[" and x != "," and x!=" " and x!="]" and x!="\n"]]
            flag_grid = True

    with open(filename, "r") as f:
        # Removing Service parameters
        old_file = f.readlines()[:-2]
    os.system("rm "+filename)
    with open(filename, "w") as f:
        for line in old_file:
            if "gridInit" in line:  # gridInit is the last parameter
                break
            f.write(line)
        f.write("letting steps be "+str(steps)+"\n")
        f.write("letting gridInit be ["+str(list(grid[0]))+",\n")
        for i in range(1, len(grid)-1):
            f.write("                     "+str(list(grid[i]))+",\n")
        f.write("                     "+str(list(grid[-1]))+"]\n")

os.system("rm bombastic/s0")
os.system("rm bombastic/.MINION*")
os.system("rm bombastic/*.infor")
os.system("rm bombastic/*.info")
os.system("rm bombastic/*.minion")

