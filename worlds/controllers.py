import carla
import pygame


from pygame.constants import K_BACKSPACE, K_TAB, K_c, KMOD_SHIFT, K_BACKQUOTE, K_0, K_9, K_r, KMOD_CTRL, K_p, K_MINUS, \
    K_EQUALS, K_UP, K_w, K_LEFT, K_a, K_RIGHT, K_d, K_DOWN, K_s, K_SPACE, K_ESCAPE, K_q


# -------------------------------------------------------------------------------------------------
# -- Keyboard Controller
# -------------------------------------------------------------------------------------------------

class KeyboardController(object):
    """KeyboardController: control the player's vehicle with keyboard."""

    def __init__(self, world):
        # assert isinstance(world.player, carla.Vehicle)

        self._control = carla.VehicleControl()
        self._steer_cache = 0.0

    def parse_events(self, client, world, clock, training=False):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

                elif event.key == K_BACKSPACE:
                    world.restart()

                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()

                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)

                elif event.key == K_c:
                    world.next_weather()

                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()

                elif K_0 < event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)

                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()

                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if world.recording_enabled:
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")

                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False

                    # work around to fix camera at start of replaying
                    currentIndex = world.camera_manager.index
                    world.destroy_sensors()

                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(currentIndex)

                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % world.recording_start)

                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % world.recording_start)

                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1

        # apply control to vehicle
        if not training:
            self._parse_vehicle_keys(keys=pygame.key.get_pressed(), milliseconds=clock.get_time())
            self._control.reverse = self._control.gear < 0
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds * 2

        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment

        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0

        self._steer_cache = min(0.8, max(-0.8, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)
