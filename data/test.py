from numpy import load

def main() -> None:
    data = load('processed.npz')
    lst = data.files
    for item in lst:
        print(item)
        print(data[item])


if __name__ == "__main__":
    main()
