"""Resolve known robot geometry by its native ID, including unnamed gaps."""

from .ik_v3 import MultiStartPandaIK


class RobotModelNames:
    def __init__(self, model, prefixes):
        self.original = model
        # LIBERO's geom_names tuple omits unnamed geometries. Its list index is
        # therefore not a MuJoCo geometry ID in the kitchen scenes.
        names = [None]*int(model._model.ngeom)
        for name in model.geom_names:
            if name and name.startswith(prefixes):
                names[int(model.geom_name2id(name))] = name
        self.geom_names = tuple(names)

    def __getattr__(self, name):
        return getattr(self.original, name)


class RobotSimNames:
    def __init__(self, sim, robot):
        self.original = sim
        self.model = RobotModelNames(sim.model, (robot.robot_model.naming_prefix, robot.gripper.naming_prefix))

    def __getattr__(self, name):
        return getattr(self.original, name)


class ArticulatedPandaIK(MultiStartPandaIK):
    def __init__(self, sim, robot):
        super().__init__(RobotSimNames(sim, robot), robot)
