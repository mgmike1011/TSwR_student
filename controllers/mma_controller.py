import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel

class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3

        self.Tp = Tp

        self.Model_1 = ManiuplatorModel(Tp)
        self.Model_1.m3 = 0.1
        self.Model_1.r3 = 0.05

        self.Model_2 = ManiuplatorModel(Tp)
        self.Model_2.m3 = 0.01
        self.Model_2.r3 = 0.01

        self.Model_3 = ManiuplatorModel(Tp)
        self.Model_3.m3 = 1.0
        self.Model_3.r3 = 0.3

        self.models = [self.Model_1, self.Model_2, self.Model_3]
        self.i = 0

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        pass

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        e = q_r - q
        e_dot = q_r_dot - q_dot
        Kd = np.array([[25, 0], [0, 15]])
        Kp = np.array([[25, 0], [0, 60]])
        v = q_r_ddot + Kd @ e_dot + Kp @ e
        # v = q_r_ddot # TODO: add feedback
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        return u
