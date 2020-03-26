import numpy as np

def load_bc():
    x = []
    y = []
    with open("breast-cancer.txt", mode="r") as fd:
        for line in fd:
            data = line.split("\t")
            data = [(i, -1)[i=="?"] for i in data]
            data = [int(i) for i in data]
            y.append(data[0])
            x.append(data[1:])

    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    print(x)
    print(y)

    np.save("x-bc.npy", x)
    np.save("y-bc.npy", y)

    return x, y

if __name__ == "__main__":
    load_bc()

