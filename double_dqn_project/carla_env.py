import carla
import random
import time
import numpy as np

class CarlaEnv:
    def __init__(self):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)

        self.world = self.client.load_world("Town03")
        self.blueprint_library = self.world.get_blueprint_library()

        self.vehicle_bp = self.blueprint_library.filter("model3")[0]

        self.spawn_point = random.choice(self.world.get_map().get_spawn_points())

        self.collision_bp = self.blueprint_library.find("sensor.other.collision")

        self.vehicle = None
        self.collision_sensor = None
        self.collision_flag = False

        self.action_space = 3
        self.state_size = 4

    def safe_destroy(self, actor):
        """Safely destroy actors without causing errors."""
        if actor is not None:
            try:
                if actor.is_alive:
                    actor.destroy()
            except:
                pass

    def _attach_collision_sensor(self):
        self.safe_destroy(self.collision_sensor)

        self.collision_sensor = self.world.try_spawn_actor(
            self.collision_bp,
            carla.Transform(),
            attach_to=self.vehicle
        )

        @self.collision_sensor.listen
        def on_collision(event):
            self.collision_flag = True

    def reset(self):
        # Clean old actors
        self.safe_destroy(self.vehicle)
        self.safe_destroy(self.collision_sensor)

        # Spawn vehicle
        self.vehicle = None
        while self.vehicle is None:
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, self.spawn_point)
            if self.vehicle is None:
                time.sleep(0.1)

        # Reset velocity — CARLA 0.9.14 correct API
        self.vehicle.set_target_velocity(carla.Vector3D(0, 0, 0))
        self.vehicle.set_target_angular_velocity(carla.Vector3D(0, 0, 0))

        # Collision sensor
        self.collision_flag = False
        self._attach_collision_sensor()

        time.sleep(0.3)
        return self.get_state()

    def get_state(self):
        vel = self.vehicle.get_velocity()
        speed = np.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        loc = self.vehicle.get_transform().location
        return np.array([speed, loc.x, loc.y, loc.z], dtype=np.float32)

    def step(self, action):
        control = carla.VehicleControl()

        if action == 0:
            control.throttle = 0.4
            control.steer = -0.3
        elif action == 1:
            control.throttle = 0.5
            control.steer = 0.0
        elif action == 2:
            control.throttle = 0.4
            control.steer = 0.3

        self.vehicle.apply_control(control)

        time.sleep(0.05)
        next_state = self.get_state()

        speed = next_state[0]
        reward = speed * 0.1

        if self.collision_flag:
            reward -= 200
            done = True
        else:
            done = False

        return next_state, reward, done
