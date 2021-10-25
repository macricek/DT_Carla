from carlaDP import Environment
import genetic



def test_genetic():
    genetic.createGenetic()


def main():
    env = Environment(True)
    env.__del__()

if __name__ == '__main__':
    test_genetic()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
