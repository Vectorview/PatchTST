
runs = {}
key = ""
with open("result.txt", "r") as f:
    for line in f:
        if len(line[0]) == 0:
            pass
        elif line[0] == 'm':
            runs[key] = float(line[4:14])
        else:
            key = line


for k, v in sorted(runs.items(), key=lambda x: -x[1]):
    print(k, v)
