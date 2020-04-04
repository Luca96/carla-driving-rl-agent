import carla
import numpy as np
import pygame

from worlds.utils import Attachment, cc


# -------------------------------------------------------------------------------------------------
# -- SensorsManager
# -------------------------------------------------------------------------------------------------

class SensorsManager(object):
    """Abstracts the management of different sensors."""
    pass


# -------------------------------------------------------------------------------------------------
# -- CameraManager
# -------------------------------------------------------------------------------------------------

class CameraManager(object):
    def __init__(self, parent_actor, hud, callback=None, gamma_correction=2.2):
        self.sensor = None
        self.surface = None
        self.parent = parent_actor
        self.world = self.parent.get_world()
        self.hud = hud
        self.recording = False
        self.on_image_callback = callback

        bound_y = 0.5 + self.parent.bounding_box.extent.y

        self._camera_transforms = [
            (carla.Transform(carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=1.6, z=1.7)), Attachment.Rigid),
            (carla.Transform(carla.Location(x=5.5, y=1.5, z=1.5)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), Attachment.SpringArm),
            (carla.Transform(carla.Location(x=-1, y=-bound_y, z=0.5)), Attachment.Rigid)]

        self.transform_index = 1

        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB', {}],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)', {}],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)', {}],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
             'Camera Semantic Segmentation (CityScapes Palette)', {}]]

        bp_library = self.world.get_blueprint_library()

        for item in self.sensors:
            bp = bp_library.find(item[0])

            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))

                if bp.has_attribute('gamma'):
                    bp.set_attribute('gamma', str(gamma_correction))

                for attr_name, attr_value in item[3].items():
                    bp.set_attribute(attr_name, attr_value)

            item.append(bp)

        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def on_image(self, image):
        # Image from camera rgb/depth/semantic seg.
        image.convert(self.sensors[self.index][1])
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]

        if self.on_image_callback is not None:
            self.on_image_callback(array.copy())

        self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

    def set_sensor(self, index, notify=True, force_respawn=False):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else \
            (force_respawn or (self.sensors[index][2] != self.sensors[self.index][2]))

        if needs_respawn:
            self._destroy_sensor()
            self.sensor = self.world.spawn_actor(self.sensors[index][-1],
                                                 self._camera_transforms[self.transform_index][0],
                                                 attach_to=self.parent,
                                                 attachment_type=self._camera_transforms[self.transform_index][1])
            self.sensor.listen(self.on_image)
            print('camera-transform', self._camera_transforms[self.transform_index][0], self._camera_transforms[self.transform_index][1])

        if notify:
            self.hud.notification(self.sensors[index][2])  # sensor display name

        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    def _destroy_sensor(self):
        """Destroys the current equipped sensor."""
        if self.sensor is not None:
            self.sensor.stop()
            self.sensor.destroy()
            self.sensor = None
            self.surface = None

    def destroy(self):
        self._destroy_sensor()
        self.parent = None
        self.world = None
        self.hud = None
        self.recording = None
        self.on_image_callback = None
        self._camera_transforms = None
        self.transform_index = None
        self.sensors = None
        self.index = None
