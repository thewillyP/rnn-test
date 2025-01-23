import unittest

import torch

from recurrent.datarecords import Input2Output1
from recurrent.mylearning import UORO, doRnnReadout, doRnnStep
from recurrent.myrecords import RnnGodState
from recurrent.objectalgebra.interpreter import BaseRnnInterpreter
from recurrent.parameters import RnnConfig, RnnParameter, UORO_Param
from recurrent.mytypes import *
from recurrent.util import *


class Test_UORO(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        alpha = 1.0

        n_h = 2
        n_in = 2
        n_out = 2

        _w_rec = torch.eye(n_h)
        _w_in = torch.eye(n_in)
        _b_rec = torch.zeros(n_h, 1)
        w_rec = torch.cat((_w_rec, _w_in, _b_rec), dim=1)

        _w_out = torch.eye(n_out)
        _b_out = torch.zeros(n_out, 1)
        w_out = torch.cat((_w_out, _b_out), dim=1)

        w_rec = torch.flatten(w_rec)
        w_out = torch.flatten(w_out)

        a_init = torch.ones(n_h)

        parameter = RnnParameter(
            w_rec=PARAMETER(w_rec),
            w_out=PARAMETER(w_out),
        )
        rnnConfig = RnnConfig(
            n_h=n_h, n_in=n_in, n_out=n_out, alpha=alpha, activationFn=lambda x: x
        )

        cls.initEnv = RnnGodState[RnnParameter, None, None](
            # activation=ACTIVATION(torch.randn(batch_size, n_h)),
            activation=ACTIVATION(a_init),
            influenceTensor=torch.empty(0),
            ohoInfluenceTensor=torch.empty(0),
            parameter=parameter,
            hyperparameter=torch.empty(0),
            metaHyperparameter=torch.empty(0),
            rnnConfig=rnnConfig,
            rnnConfig_bilevel=torch.empty(0),
            rfloConfig=torch.empty(0),
            rfloConfig_bilevel=torch.empty(0),
            uoro=UORO_Param[RnnParameter](
                A=torch.ones(n_h),
                B=fmapPytree(torch.ones_like, parameter),
            ),
        )

        x = 2 * torch.ones(n_in)
        label = 2 * torch.ones(n_out)
        cls.x_input = Input2Output1(x, label)

        type DL = BaseRnnInterpreter[RnnParameter, None, None]
        type ENV = RnnGodState[RnnParameter, None, None]
        cls.dialect = BaseRnnInterpreter[RnnParameter, None, None]()

        cls.uoro = UORO[
            DL,
            Input2Output1,
            ENV,
            ACTIVATION,
            RnnParameter,
            torch.Tensor,
            PREDICTION,
        ]()

        cls.rnnLearner = cls.uoro.onlineLearning(
            doRnnStep(),
            doRnnReadout(),
            lambda a, b: LOSS(torch.functional.F.cross_entropy(a, b)),
        )

    def test_update_learning_vars(self):
        v = torch.distributions.uniform.Uniform(-1, 1).sample(
            (self.initEnv.rnnConfig.n_h,)
        )
        (a_j, p_papw), env = self.uoro.gradientFlow(v, doRnnStep()).func(
            self.dialect, self.x_input, self.initEnv
        )

        flats_, _ = pytree.tree_flatten(p_papw)
        p_papw_pred = torch.cat(flats_)

        correct_a_J_ = torch.eye(2)
        correct_a_J = correct_a_J_ @ self.initEnv.activation
        correct_papw_ = v @ torch.tensor(
            [[1, 1, 2, 2, 1], [1, 1, 2, 2, 1]], dtype=torch.float32
        )
        correct_papw = Gradient[RnnParameter](
            RnnParameter(
                w_rec=PARAMETER(torch.flatten(correct_papw_)),
                w_out=PARAMETER(torch.zeros_like(self.initEnv.parameter.w_out)),
            )
        )
        flats, _ = pytree.tree_flatten(correct_papw)
        correct_papw = torch.cat(flats)

        torch.allclose(a_j, correct_a_J)
        print(p_papw_pred)
        print(correct_papw)
        torch.allclose(p_papw_pred, correct_papw)
