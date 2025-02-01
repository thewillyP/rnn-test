import unittest
from unittest.mock import MagicMock


from recurrent.datarecords import InputOutput
from recurrent.mylearning import UORO, RnnLibrary, doRnnReadout, doRnnStep
from recurrent.myrecords import RnnGodState
from recurrent.objectalgebra.interpreter import BaseRnnInterpreter
from recurrent.parameters import IsVector, Logs, RnnConfig, RnnParameter, UORO_Param
from recurrent.mytypes import *
from recurrent.util import *
from recurrent.monad import PX, pure
import jax.numpy as jnp
import jax
import optax


class Test_UORO(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        alpha = 1.0

        n_h = 2
        n_in = 2
        n_out = 2

        _w_rec = jnp.eye(n_h)
        _w_in = jnp.eye(n_in)
        _b_rec = jnp.zeros((n_h, 1))
        w_rec = jnp.concat((_w_rec, _w_in, _b_rec), axis=1)

        _w_out = jnp.eye(n_out)
        _b_out = jnp.zeros((n_out, 1))
        w_out = jnp.concat((_w_out, _b_out), axis=1)

        a_init = jnp.ones(n_h)

        parameter = RnnParameter(
            w_rec=w_rec,
            w_out=w_out,
        )
        rnnConfig = RnnConfig(n_h=n_h, n_in=n_in, n_out=n_out, alpha=alpha, activationFn=lambda x: x)

        cls.key = PRNG(jax.random.key(42))

        cls.initEnv = RnnGodState[RnnParameter, None, None](
            # activation=ACTIVATION(torch.randn(batch_size, n_h)),
            activation=ACTIVATION(a_init),
            influenceTensor=jnp.empty(0),
            ohoInfluenceTensor=jnp.empty(0),
            parameter=parameter,
            hyperparameter=jnp.empty(0),
            metaHyperparameter=jnp.empty(0),
            rnnConfig=rnnConfig,
            rnnConfig_bilevel=jnp.empty(0),
            uoro=UORO_Param(
                A=jnp.ones(n_h),
                B=toVector(
                    endowVector(
                        RnnParameter(
                            w_rec=jnp.ones_like(parameter.w_rec),
                            w_out=jnp.zeros_like(parameter.w_out),
                        )
                    )
                ),
            ),
            prng=cls.key,
        )

        x = 2 * jnp.ones(n_in)
        label = 2 * jnp.ones(n_out)
        cls.x_input = InputOutput(x, label)

        type DL = BaseRnnInterpreter[RnnParameter, None, None]
        type ENV = RnnGodState[RnnParameter, None, None]
        cls.dialect = BaseRnnInterpreter[RnnParameter, None, None]()

        distribution = MagicMock()
        distribution.return_value = jnp.array([1.0, -1.0])

        cls.uoro = UORO[
            InputOutput,
            ENV,
            ACTIVATION,
            RnnParameter,
            jax.Array,
            PREDICTION,
        ](distribution)

        cls.rnnLearner: RnnLibrary[DL, InputOutput, ENV, PREDICTION, RnnParameter]
        cls.rnnLearner = cls.uoro.createLearner(
            doRnnStep(),
            doRnnReadout(),
            lambda a, b: LOSS(optax.safe_softmax_cross_entropy(a, b)),
        )

    def test_update_learning_vars(self):
        v = jax.random.uniform(self.key, (self.initEnv.rnnConfig.n_h,), minval=-1, maxval=1)

        model = eqx.filter_jit(self.uoro.getProjectedGradients(v, doRnnStep()).func)
        safe_model = model.lower(self.dialect, self.x_input, self.initEnv).compile()
        (a_j, p_papw), env = safe_model(self.dialect, self.x_input, self.initEnv)

        correct_a_J_ = jnp.eye(2)
        correct_a_J = correct_a_J_ @ self.initEnv.activation
        correct_papw_ = v @ jnp.array(
            [[1, 1, 2, 2, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 1, 2, 2, 1]],
            dtype=jnp.float32,
        )
        correct_papw = toVector(
            endowVector(
                RnnParameter(
                    w_rec=jnp.ravel(correct_papw_),
                    w_out=jnp.zeros_like(self.initEnv.parameter.w_out),
                )
            )
        )

        jnp.allclose(a_j, correct_a_J)
        jnp.allclose(p_papw, correct_papw)

    def test_get_influence_estimate(self):
        model = eqx.filter_jit(self.rnnLearner.rnnWithGradient.func)
        safe_model = model.lower(self.dialect, self.x_input, self.initEnv).compile()
        _, env = safe_model(self.dialect, self.x_input, self.initEnv)
        A_pred = env.uoro.A
        B_pred = env.uoro.B

        p0 = jnp.sqrt(jnp.sqrt(jnp.array(5.0)))
        p1 = jnp.sqrt(jnp.sqrt(jnp.array(11.0)))
        M_proj = jnp.array([[1, 1, 2, 2, 1], [-1, -1, -2, -2, -1]])
        A_correct = p0 * jnp.array([1, 1]) + p1 * jnp.array([1, -1])
        B_correct = (1 / p0) * jnp.ones((2, 5)) + (1 / p1) * M_proj
        B_correct = toVector(
            endowVector(
                RnnParameter(
                    w_rec=jnp.ravel(B_correct),
                    w_out=jnp.zeros_like(self.initEnv.parameter.w_out),
                )
            )
        )

        jnp.allclose(A_pred, A_correct)
        jnp.allclose(B_pred, B_correct)

    def test_get_rec_grads(self):
        A, B = self.initEnv.uoro.A, self.initEnv.uoro.B
        error = pure(Gradient(jnp.ones(2) * 0.5), PX())
        model = eqx.filter_jit(self.uoro.propagateRecurrentError(A, B, error).func)
        safe_model = model.lower(self.dialect, self.x_input, self.initEnv).compile()
        rec_grads, _ = safe_model(self.dialect, self.x_input, self.initEnv)

        correct_rec_grads = jnp.ones((2, 5))
        correct_rec_grads = toVector(
            endowVector(
                RnnParameter(
                    w_rec=jnp.ravel(correct_rec_grads),
                    w_out=jnp.zeros_like(self.initEnv.parameter.w_out),
                )
            )
        )

        jnp.allclose(rec_grads.value, correct_rec_grads)
