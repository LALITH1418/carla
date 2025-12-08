# carla_env.py - STABILIZED VERSION
import carla
import random
import time
import numpy as np
import math
from collections import deque
import cv2


class CarlaEnv:
    """
    Stabilized CARLA environment with normalized state and balanced rewards.
   """
    def __init__(self, town="Town03", frame_stack_size=4):
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(20.0)

        self.world = self.client.load_world(town)
        self.spectator = self.world.get_spectator()
        self.vehicle = None
        
        self.max_steps = 1000
        self.step_count = 0

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        self.map = self.world.get_map()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle_bp = self.blueprint_library.filter("model3")[0]

        self.spawn_point = random.choice(self.map.get_spawn_points())

        #colllision sensor setup
        self.collision_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = None
        self.collision_flag = False

        #camera setup
        self.camera_bp = self.blueprint_library.find("sensor.camera.rgb")
        # Image resolution (tweak as needed)
        self.cam_width = 84
        self.cam_height = 84
        self.camera_bp.set_attribute("image_size_x", str(self.cam_width))
        self.camera_bp.set_attribute("image_size_y", str(self.cam_height))
        self.camera_bp.set_attribute("fov", "90")

        self.camera = None
        self.latest_image = None

        #stacking frames 
        self.frame_stack_size = frame_stack_size
        self.frame_stack = deque(maxlen=self.frame_stack_size)
        self.action_size = 4  # Left, Straight, Right, Brake

        self.image_shape = (self.cam_height, self.cam_width, 3)
        self.fram_stack_shape = (self.frame_stack_size,
                                 self.cam_height,
                                 self.cam_width,
                                 3)
                            
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
                print("Collision detected!")
                time.sleep(0.1)  # Debounce
            self.collision_sensor.listen(on_collision)

    def _attach_camera(self):
        self.safe_destroy(self.camera)
        self.latest_image = None

        # Position camera on hood / windshield
        camera_transform = carla.Transform(
            carla.Location(x=1.5, z=1.7),  # in front and above
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0)
        )

        self.camera = self.world.spawn_actor(
            self.camera_bp,
            camera_transform,
            attach_to=self.vehicle
        )

        # Callback to convert CARLA image to numpy array
        def _on_image(image):
            # BGRA 8-bit
            array = np.frombuffer(image.raw_data, dtype=np.uint8)
            array = array.reshape((image.height, image.width, 4))
            rgb = array[:, :, :3][:, :, ::-1]  # Convert BG R -> R G B (optional)
            
            rgb = cv2.resize(rgb, (self.cam_height,self.cam_width), interpolation=cv2.INTER_AREA)
            self.latest_image = rgb

        self.camera.listen(_on_image)

    def _update_spectator(self):
         # spectator behind and above the car
        transform = self.vehicle.get_transform()
        location  = transform.location
        rotation  = transform.rotation
        forward_vec = transform.get_forward_vector()

        dist_back = 8.0  
        height    = 4.0   
        pitch     = -10.0 


        cam_location = location - forward_vec * dist_back
        cam_location.z += height

        cam_rotation = carla.Rotation(
            pitch=pitch,
            yaw=rotation.yaw,
            roll=0.0
        )

        self.spectator.set_transform(carla.Transform(cam_location, cam_rotation))
    
    def _get_vec_state(self):
        """Returns normalized vector state of shape (4,)."""
        vel = self.vehicle.get_velocity()
        speed = math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

        wp = self.map.get_waypoint(
            self.vehicle.get_transform().location,
            project_to_road=True
        )
        lane_center = wp.transform.location
        loc = self.vehicle.get_transform().location

        dist_center = math.hypot(loc.x - lane_center.x, loc.y - lane_center.y)
        target = self._target_location
        dist_to_target = math.hypot(loc.x - target.x, loc.y - target.y)

        # NORMALIZE all values to [0, 1]
        norm_speed = np.clip(speed / self.max_speed, 0, 1)
        norm_dist_center = np.clip(dist_center / self.max_lane_deviation, 0, 1)
        collision = 1.0 if self.collision_flag else 0.0
        norm_dist_target = np.clip(dist_to_target / self.max_target_distance, 0, 1)

        state = np.array(
            [norm_speed, norm_dist_center, collision, norm_dist_target],
            dtype=np.float32
        )
        return state, dist_to_target

    def _ensure_frame_stack_filled(self):
        """
        Ensures frame_stack has frame_stack_size frames.
        If latest_image is None, use black frames.
        """
        if self.latest_image is None:
            # Create a black frame if nothing has arrived yet
            frame = np.zeros(self.image_shape, dtype=np.uint8)
        else:
            frame = self.latest_image

        # If stack empty, fill with the same frame
        if len(self.frame_stack) == 0:
            for _ in range(self.frame_stack_size):
                self.frame_stack.append(frame.copy())
        else:
            # Just append the latest frame
            self.frame_stack.append(frame.copy())
   
    def _get_observation(self):
        """
        Returns:
        frames: np.array, shape (T, C, H, W), float32 in [0, 1]
        dist_to_target: float
        """
        _, dist_to_target = self._get_vec_state()
        
        self._ensure_frame_stack_filled()

        # Stack frames into float32 [0,1]
        frames = np.stack(self.frame_stack, axis=0).astype(np.float32) / 255.0
        frames = np.transpose(frames,(0,3,1,2))
        return frames, dist_to_target
    
    def reset(self):
        self.safe_destroy(self.vehicle)
        self.safe_destroy(self.collision_sensor)
        self.safe_destroy(self.camera)

        self.collision_flag = False
        self.prev_target_distance = None
        self.step_count = 0
        self.latest_image = None
        self.frame_stack.clear()

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
        self._attach_camera()

        for _ in range(5):
            self.world.tick()

        self._update_spectator()

        # Set target waypoint ahead
        wp = self.map.get_waypoint(self.vehicle.get_transform().location)
        next_wps = wp.next(30.0)  # Look further ahead
        if next_wps:
            self._target_location = next_wps[0].transform.location
        else:
            self._target_location = wp.transform.location

        frames, dist_to_target = self._get_observation()
        self.prev_target_distance = dist_to_target  # Denormalize for tracking
        
        return frames


    # def get_state(self):
    #     """Returns normalized state for stable learning."""
    #     vel = self.vehicle.get_velocity()
    #     speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

    #     wp = self.map.get_waypoint(self.vehicle.get_transform().location, project_to_road=True)
    #     lane_center = wp.transform.location
    #     loc = self.vehicle.get_transform().location

    #     dist_center = math.hypot(loc.x - lane_center.x, loc.y - lane_center.y)
    #     target = self._target_location
    #     dist_to_target = math.hypot(loc.x - target.x, loc.y - target.y)

    #     # NORMALIZE all values to [0, 1] or similar range
    #     norm_speed = np.clip(speed / self.max_speed, 0, 1)
    #     norm_dist_center = np.clip(dist_center / self.max_lane_deviation, 0, 1)
    #     collision = 1.0 if self.collision_flag else 0.0
    #     norm_dist_target = np.clip(dist_to_target / self.max_target_distance, 0, 1)

    #     state = np.array([norm_speed, norm_dist_center, collision, norm_dist_target], dtype=np.float32)
    #     return state

    def step(self, action):
        self.step_count += 1
        control = carla.VehicleControl()

        # Smoother control actions
        if action == 0:  # Turn left
            control.throttle = 0.5
            control.steer = -0.2
        elif action == 1:  # Straight
            control.throttle = 0.2
            control.steer = 0.0
        elif action == 2:  # Turn right
            control.throttle = 0.5
            control.steer = 0.2
        elif action == 3:  # break
            control.throttle = 0.0
            control.brake = 0.2
        else:
            control.throttle = 0.0
            control.steer = 0.0

        self.vehicle.apply_control(control)
        self.world.tick()

        #spectator update
        self._update_spectator()
        vec_state, dist_to_target = self._get_vec_state()
        norm_speed, norm_dist_center, collision_flag, _ = vec_state
        collision = bool(collision_flag)

        #frames, _ = self._get_observation()

        # BALANCED reward function (all components on similar scales)
        reward = 0.0
        
        # 1. Speed reward (0 to 1.0)
        reward += self.reward_speed * norm_speed
        
        # 2. Staying alive bonus
        reward += self.reward_alive
        
        # 3. Lane keeping penalty (0 to -2.0)
        reward -= self.penalty_lane * norm_dist_center
        
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
        
        if self.step_count >= self.max_steps:
            done = True
        self._ensure_frame_stack_filled()
        frames = np.stack(self.frame_stack, axis=0).astype(np.float32) / 255.0
        frames = np.transpose(frames, (0, 3, 1, 2))

        return frames, float(reward), done

    def close(self):
        if self.camera is not None:
            try:
                self.camera.stop()
            except Exception:
                pass
                
        self.safe_destroy(self.collision_sensor)
        self.safe_destroy(self.camera)
        self.safe_destroy(self.vehicle)