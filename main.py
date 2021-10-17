from carlaDP import Environment
import genetic_learning



def test_genetic():
    mw = 0
    aw = 0
    for i in range(0, 10):
        my, mf = genetic_learning.myGenetic()
        auto, af = genetic_learning.genetic()
        if mf > af:
            mw += 1
        else:
            aw += 1
    print("MW: {mw}, AW: {aw}".format(mw=mw, aw=aw))


def main():
    env = Environment(False)
    env.__del__()

if __name__ == '__main__':
    main()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
