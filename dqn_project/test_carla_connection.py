import carla

try:
    client = carla.Client("localhost", 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    print("✅ Connected to CARLA:", world.get_map().name)
except Exception as e:
    print("❌ Connection failed:", e)
