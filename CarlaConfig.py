import carla
import configparser


class CarlaConfig:
    client: carla.Client

    def __init__(self, client):
        self.client = client
        self.parser = configparser.ConfigParser()
        self.path = "config.txt"
        self.apply()

    def apply(self):
        ''''APPLY CONFIG SETTINGS TO SIMULATOR'''
        self.parser.read(self.path)
        self.client.set_timeout(30)
        weather = self.parser.get("CARLA", "weather")
        map = self.parser.get("CARLA", "map")
        fixed = self.parser.get("CARLA", "ts")

        world = self.client.get_world()
        curMap = world.get_map().name

        if map not in curMap:
            world = self.client.load_world(map)
        world.set_weather(eval(weather))

        if len(fixed) > 0:
            settings = world.get_settings()
            settings.fixed_delta_seconds = eval(fixed)
            world.apply_settings(settings)
        self.client.set_timeout(5)

    def readSection(self, section):
        return dict(self.parser.items(section))


if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    carlaConfig = CarlaConfig(client)
    print(carlaConfig.readSection('Camera'))
