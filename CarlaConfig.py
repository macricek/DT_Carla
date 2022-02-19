import carla
import configparser


class CarlaConfig:
    client: carla.Client

    def __init__(self, client):
        self.client = client
        self.parser = configparser.ConfigParser()
        self.path = "config.txt"
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
        world.set_weather(eval(weather))

        if len(fixed) > 0:
            settings = world.get_settings()
            settings.fixed_delta_seconds = eval(fixed)
            world.apply_settings(settings)

        self.turnOnSync() if eval(sync) else self.turnOffSync()
        self.client.set_timeout(5)

    def readSection(self, section):
        return dict(self.parser.items(section))

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


if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    conf = CarlaConfig(client)
    s = conf.readSection("Sensors")
    print(s)
    print(s.get("radarsensor"))
    print(bool(s.get("radarsensor")))

