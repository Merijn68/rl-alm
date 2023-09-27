import seaborn as sns
import numpy as np


def main():
    print(sns.__version__)
    c = [1, 2, 3]
    sns.barplot(x=np.arange(0, len(c)), y=c)

    print("Done")


if __name__ == "__main__":
    main()
