###########################################################################
# Example Rigid Ball Simulation
#
# Simple demo of a ball hitting a surface and rolling.
#
###########################################################################

import math
import os

import numpy as np
from pxr import Usd, UsdGeom

import warp as wp
import warp.examples
import warp.sim
import warp.sim.render


class Example:
    def __init__(self, stage_path="ball_simulation.usd"):
        builder = wp.sim.ModelBuilder()

        self.sim_time = 0.0
        fps = 60
        self.frame_dt = 1.0 / fps

        self.sim_substeps = 10
        self.sim_dt = self.frame_dt / self.sim_substeps

        self.num_bodies = 1
        self.scale = 0.8
        self.ke = 1.0e5
        self.kd = 250.0
        self.kf = 500.0

        # spheres
        for i in range(self.num_bodies):
            b = builder.add_body(origin=wp.transform((i, 2.5, 2.0), wp.quat_identity()))

            builder.add_shape_sphere(
                pos=wp.vec3(0.0, 0.0, 0.0), 
                radius=0.25 * self.scale, 
                body=b, 
                ke=self.ke, kd=self.kd, kf=self.kf
            )
        
        quat = wp.quat_from_axis_angle(wp.vec3(0, 0, 1), np.pi / 4.0)

        # Add static collider to offset ball's falling direction
        builder.add_shape_box(
            -1,
            pos=(0, 1.8, 2.0),
            rot=quat,
            hx=0.75,
            hy=0.1,
            hz=0.2
        )

        # Ground
        builder.add_shape_box(
            -1,
            pos=(0, 0, 2.0),
            rot=wp.quat_identity(),
            hx=10.0,
            hy=0.1,
            hz=0.2
        )

        # finalize model
        self.model = builder.finalize()
        self.model.ground = False

        self.integrator = wp.sim.SemiImplicitIntegrator()

        if stage_path:
            self.renderer = wp.sim.render.SimRenderer(self.model, stage_path, scaling=0.5)
        else:
            self.renderer = None

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        wp.sim.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, None, self.state_0)

        self.use_cuda_graph = wp.get_device().is_cuda
        if self.use_cuda_graph:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph

    def load_mesh(self, filename, path):
        asset_stage = Usd.Stage.Open(filename)
        mesh_geom = UsdGeom.Mesh(asset_stage.GetPrimAtPath(path))

        points = np.array(mesh_geom.GetPointsAttr().Get())
        indices = np.array(mesh_geom.GetFaceVertexIndicesAttr().Get()).flatten()

        return wp.sim.Mesh(points, indices)

    def simulate(self):
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            wp.sim.collide(self.model, self.state_0)
            self.integrator.simulate(self.model, self.state_0, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        with wp.ScopedTimer("step", active=True):
            if self.use_cuda_graph:
                wp.capture_launch(self.graph)
            else:
                self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=True):
            self.renderer.begin_frame(self.sim_time)
            self.renderer.render(self.state_0)
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage_path",
        type=lambda x: None if x == "None" else str(x),
        default="ball_simulation.usd",
        help="Path to the output USD file.",
    )
    parser.add_argument("--num_frames", type=int, default=300, help="Total number of frames.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path)

        for _ in range(args.num_frames):
            example.step()
            example.render()

        if example.renderer:
            example.renderer.save()