# carla_env.py - STABILIZED VERSION
import carla
import random
import time
import numpy as np
import math

class CarlaEnv:
    """
    Stabilized CARLA environment with normalized state and balanced rewards.
    State: [normalized_speed, normalized_distance_to_center, collision_flag, normalized_distance_to_target]
    """
    def __init__(self, town="Town03"):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)

        self.world = self.client.load_world(town)
        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter("model3")[0]

        self.spawn_point = random.choice(self.map.get_spawn_points())

        self.collision_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = None
        self.collision_flag = False

        self.vehicle = None
        self.action_space = 3
        self.state_size = 4

        # BALANCED reward coefficients (similar scales)
        self.reward_speed = 1.0           # Encourage forward motion
        self.reward_alive = 0.1           # Small bonus for staying alive
        self.penalty_lane = 2.0           # Penalize lane deviation
        self.penalty_collision = 10.0     # Moderate collision penalty
        self.penalty_offmap = 10.0        # Moderate off-map penalty
        self.bonus_progress = 0.5         # Reward progress toward target
        
        # Normalization constants
        self.max_speed = 15.0  # m/s (~54 km/h)
        self.max_lane_deviation = 3.0  # meters
        self.max_target_distance = 50.0  # meters

        # Track previous distance for progress reward
        self.prev_target_distance = None

        settings = self.world.get_settings()
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

    def safe_destroy(self, actor):
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
        if self.collision_sensor is not None:
            def on_collision(event):
                self.collision_flag = True
            self.collision_sensor.listen(on_collision)

    def reset(self):
        self.safe_destroy(self.vehicle)
        self.safe_destroy(self.collision_sensor)
        self.collision_flag = False
        self.prev_target_distance = None

        self.vehicle = None
        attempts = 0
        while self.vehicle is None and attempts < 50:
            attempts += 1
            self.spawn_point = random.choice(self.map.get_spawn_points())
            spawn = carla.Transform(self.spawn_point.location, self.spawn_point.rotation)
            spawn.location.z += 0.3
            self.vehicle = self.world.try_spawn_actor(self.vehicle_bp, spawn)
            if self.vehicle is None:
                time.sleep(0.05)

        if self.vehicle is None:
            raise RuntimeError("Failed to spawn vehicle in CARLA")

        self.vehicle.set_target_velocity(carla.Vector3D(0,0,0))
        self.vehicle.set_target_angular_velocity(carla.Vector3D(0,0,0))

        self.collision_flag = False
        self._attach_collision_sensor()

        # Set target waypoint ahead
        wp = self.map.get_waypoint(self.vehicle.get_transform().location)
        next_wps = wp.next(30.0)  # Look further ahead
        if next_wps:
            self._target_location = next_wps[0].transform.location
        else:
            self._target_location = wp.transform.location

        time.sleep(0.2)
        state = self.get_state()
        self.prev_target_distance = state[3] * self.max_target_distance  # Denormalize for tracking
        return state

    def get_state(self):
        """Returns normalized state for stable learning."""
        vel = self.vehicle.get_velocity()
        speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        wp = self.map.get_waypoint(self.vehicle.get_transform().location, project_to_road=True)
        lane_center = wp.transform.location
        loc = self.vehicle.get_transform().location

        dist_center = math.hypot(loc.x - lane_center.x, loc.y - lane_center.y)
        target = self._target_location
        dist_to_target = math.hypot(loc.x - target.x, loc.y - target.y)

        # NORMALIZE all values to [0, 1] or similar range
        norm_speed = np.clip(speed / self.max_speed, 0, 1)
        norm_dist_center = np.clip(dist_center / self.max_lane_deviation, 0, 1)
        collision = 1.0 if self.collision_flag else 0.0
        norm_dist_target = np.clip(dist_to_target / self.max_target_distance, 0, 1)

        state = np.array([norm_speed, norm_dist_center, collision, norm_dist_target], dtype=np.float32)
        return state

    def step(self, action):
        control = carla.VehicleControl()

        # Smoother control actions
        if action == 0:  # Turn left
            control.throttle = 0.5
            control.steer = -0.3
        elif action == 1:  # Straight
            control.throttle = 0.6
            control.steer = 0.0
        elif action == 2:  # Turn right
            control.throttle = 0.5
            control.steer = 0.3
        else:
            control.throttle = 0.0
            control.steer = 0.0

        self.vehicle.apply_control(control)
        self.world.tick()

        next_state = self.get_state()
        
        # Denormalize for reward calculation
        speed = next_state[0] * self.max_speed
        dist_center = next_state[1] * self.max_lane_deviation
        collision = bool(next_state[2])
        dist_to_target = next_state[3] * self.max_target_distance

        # BALANCED reward function (all components on similar scales)
        reward = 0.0
        
        # 1. Speed reward (0 to 1.0)
        reward += self.reward_speed * next_state[0]
        
        # 2. Staying alive bonus
        reward += self.reward_alive
        
        # 3. Lane keeping penalty (0 to -2.0)
        reward -= self.penalty_lane * next_state[1]
        
        # 4. Progress reward (approaching target)
        if self.prev_target_distance is not None:
            progress = self.prev_target_distance - dist_to_target
            reward += self.bonus_progress * np.clip(progress, -1, 1)
        self.prev_target_distance = dist_to_target

        done = False
        
        # 5. Collision penalty
        if collision:
            reward -= self.penalty_collision
            done = True

        # 6. Off-map penalty
        loc = self.vehicle.get_transform().location
        if abs(loc.x) > 500 or abs(loc.y) > 500:
            reward -= self.penalty_offmap
            done = True
        
        # 7. Check if reached target
        if dist_to_target < 2.0:
            reward += 5.0  # Bonus for reaching target
            done = True

        return next_state, float(reward), done

    def close(self):
        self.safe_destroy(self.collision_sensor)
        self.safe_destroy(self.vehicle)