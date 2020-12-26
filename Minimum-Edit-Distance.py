'''
Calculate The Minimum Edit Distance Between 2 Strings
1. Set the cost of insert, delete and replace edits
2. Define a function which takes in 2 inputs - the source and the target
3. Initialise a matrix with the source as the rows and the target as the columns; include #
4. Populate the first row and column of the matrix
5. Calculate the possible edit distances for each entries
6. Populate each entry with the minimum possible edit distance
'''

import numpy as np

# step 1: set the cost of insert, delete and replace edits
INSERT, DELETE, REPLACE = 1,1,2

# step 2: define a function which takes in 2 inputs - the source and the target

def min_edit_d(source,target):
    # step 3: initialise a matrix with the source as the rows and the target as the columns; include #
    rows = len(source)+1
    cols = len(target)+1
    table = np.zeros((rows,cols))
    # step 4: populate the first row and column of the matrix
    for x in range(1,rows):
        table[x][0] = table[x-1][0]+DELETE
    for x in range(1,cols):
        table[0][x] = table[0][x-1]+INSERT
    # step 5: calculate the possible edit distances for each entries
    # step 6: populate each entry with the minimum possible edit distance
    for i in range(1,rows):
        for j in range(1,cols):
            table[i][j] = min([table[i-1][j] + DELETE,
                              table[i][j-1] + INSERT,
                              table[i-1][j-1] + [REPLACE if source[i-1] != target[j-1] else 0][0]])
    print(table)
    return table[-1][-1]
