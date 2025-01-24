import unittest
from unittest.mock import MagicMock

import torch

from recurrent.datarecords import Input2Output1
from recurrent.mylearning import UORO, doRnnReadout, doRnnStep
from recurrent.myrecords import RnnGodState
from recurrent.objectalgebra.interpreter import BaseRnnInterpreter
from recurrent.parameters import RnnConfig, RnnParameter, UORO_Param
from recurrent.mytypes import *
from recurrent.util import *
from recurrent.monad import pure


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
                B=Gradient[RnnParameter](
                    RnnParameter(
                        w_rec=PARAMETER(torch.ones_like(parameter.w_rec)),
                        w_out=PARAMETER(torch.zeros_like(parameter.w_out)),
                    )
                ),
            ),
        )

        x = 2 * torch.ones(n_in)
        label = 2 * torch.ones(n_out)
        cls.x_input = Input2Output1(x, label)

        type DL = BaseRnnInterpreter[RnnParameter, None, None]
        type ENV = RnnGodState[RnnParameter, None, None]
        cls.dialect = BaseRnnInterpreter[RnnParameter, None, None]()

        distribution = MagicMock()
        distribution.sample.return_value = torch.tensor([1, -1], dtype=torch.float32)

        cls.uoro = UORO[
            DL,
            Input2Output1,
            ENV,
            ACTIVATION,
            RnnParameter,
            torch.Tensor,
            PREDICTION,
        ](distribution)

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
            [[1, 1, 2, 2, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 2, 2, 1]],
            dtype=torch.float32,
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
        torch.allclose(p_papw_pred, correct_papw)

    def test_get_influence_estimate(self):

        _, env = self.rnnLearner.rnnWithGradient.func(
            self.dialect, self.x_input, self.initEnv
        )
        A_pred = env.uoro.A
        B_pred = env.uoro.B
        flats_, _ = pytree.tree_flatten(B_pred)
        B_pred = torch.cat(flats_)

        p0 = torch.sqrt(torch.sqrt(torch.tensor(5.0)))
        p1 = torch.sqrt(torch.sqrt(torch.tensor(11.0)))
        M_proj = torch.tensor([[1, 1, 2, 2, 1], [-1, -1, -2, -2, -1]])
        A_correct = p0 * torch.tensor([1, 1]) + p1 * torch.tensor([1, -1])
        B_correct = (1 / p0) * torch.ones((2, 5)) + (1 / p1) * M_proj
        B_correct = Gradient[RnnParameter](
            RnnParameter(
                w_rec=PARAMETER(torch.flatten(B_correct)),
                w_out=PARAMETER(torch.zeros_like(self.initEnv.parameter.w_out)),
            )
        )
        flats_, _ = pytree.tree_flatten(B_correct)
        B_correct = torch.cat(flats_)

        torch.allclose(A_pred, A_correct)
        torch.allclose(B_pred, B_correct)

    def test_get_rec_grads(self):

        A, B = self.initEnv.uoro.A, self.initEnv.uoro.B
        error = pure(Gradient(torch.ones(2) * 0.5))
        rec_grads, _ = self.uoro.creditAssign(A, B, error).func(
            self.dialect, self.x_input, self.initEnv
        )
        flats_, _ = pytree.tree_flatten(rec_grads)
        rec_grads = torch.cat(flats_)

        correct_rec_grads = torch.ones((2, 5))
        correct_rec_grads = Gradient[RnnParameter](
            RnnParameter(
                w_rec=PARAMETER(torch.flatten(correct_rec_grads)),
                w_out=PARAMETER(torch.zeros_like(self.initEnv.parameter.w_out)),
            )
        )
        flats_, _ = pytree.tree_flatten(correct_rec_grads)
        correct_rec_grads = torch.cat(flats_)

        torch.allclose(rec_grads, correct_rec_grads)
