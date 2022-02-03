import carla
import configparser


class CarlaConfig:
    client: carla.Client

    def __init__(self, client):
        self.client = client
        self.parser = configparser.ConfigParser()
        self.path = "config.txt"

    def apply(self):
        self.parser.read(self.path)
        weather = self.parser.get("CARLA", "weather")
        map = self.parser.get("CARLA", "map")
        world = self.client.load_world(map)
        world.set_weather(eval(weather))


if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(10)
    carlaConfig = CarlaConfig(client)
    carlaConfig.apply()
