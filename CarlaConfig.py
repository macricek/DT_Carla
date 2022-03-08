import carla
import configparser
import queue

class CarlaConfig:
    client: carla.Client

    def __init__(self, client):
        self.client = client
        self.parser = configparser.ConfigParser()
        self.path = "config.ini"
        self.sync = client.get_world().get_settings().synchronous_mode
        self.apply()

    def apply(self):
        ''''APPLY CONFIG SETTINGS TO SIMULATOR'''
        self.parser.read(self.path)
        self.client.set_timeout(30)
        weather = self.parser.get("CARLA", "weather")
        map = self.parser.get("CARLA", "map")
        fixed = self.parser.get("CARLA", "ts")
        sync = self.parser.get("CARLA", "sync")
        world = self.client.get_world()
        curMap = world.get_map().name

        if map not in curMap:
            world = self.client.load_world(map)
        elif len(world.get_actors()) > 0:
            print("Re-load world!")
            world = self.client.reload_world()
        world.set_weather(eval(weather))

        if len(fixed) > 0:
            settings = world.get_settings()
            settings.fixed_delta_seconds = eval(fixed)
            world.apply_settings(settings)

        self.turnOnSync() if eval(sync) else self.turnOffSync()
        self.client.set_timeout(5)

    def readSection(self, section):
        return dict(self.parser.items(section))

    def loadAndIncrementNE(self) -> dict:
        expDict = self.readSection("NE")
        old = int(self.parser.get("NE", "rev"))
        self.parser.set("NE", "rev", str(old + 1))
        self.rewrite()
        return expDict

    def rewrite(self):
        with open(self.path, 'w') as configfile:
            self.parser.write(configfile)

    def turnOffSync(self):
        world = self.client.get_world()
        if self.sync:
            settings = self.client.get_world().get_settings()
            settings.synchronous_mode = False
            self.sync = False
            world.apply_settings(settings)
            print("Sync mode turned off!")

    def turnOnSync(self):
        world = self.client.get_world()
        if not self.sync:
            settings = self.client.get_world().get_settings()
            settings.synchronous_mode = True
            self.sync = True
            world.apply_settings(settings)
            print("Sync mode turned on!")

    @staticmethod
    def loadPath() -> queue.Queue:
        path = queue.Queue()
        x = [-8.6, -121.8, -363.4, -463.6, -490]
        y = [107.5, 395.2, 406, 333.6, 174]
        for i in range(len(x)):
            loc = carla.Location()
            loc.x = x[i]
            loc.y = y[i]
            loc.z = 0.267
            path.put(loc)
        return path


if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    conf = CarlaConfig(client)
    s = conf.readSection("NE")
    print(s)
    from NeuroEvolution import NeuroEvolution
    testNe = NeuroEvolution(s)
