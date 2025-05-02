
visited = [["x" for _ in range(80)] for _ in range(22)]
#print(visited)

for r, row in enumerate(visited):
    line = f'{r}:'
    for c, cell in enumerate(row):
        #print(f'{r}:{cell}')
        line += f" {cell}"
    print(line)