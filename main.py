# %%
import time
from recurrent.datarecords import Input2Output1
from recurrent.mylearning import RnnLibrary, createRnnLibrary, offlineLearning
from recurrent.myrecords import RnnFutureFaceState
import torch
from recurrent.mytypes import *

from recurrent.objectalgebra.base_interpreter import RnnLearnableInterpreter
from recurrent.objectalgebra.tensor_intepreter import Input2Output1Interpreter

from matplotlib import pyplot as plt
from recurrent.parameters import RnnParameter, SgdParameter
from recurrent.util import rnnSplitParameters, tree_stack

"""
Todo
1) implement vanilla rnn training loop
2) implement feedforward to show how easy it is
3) implement oho to show how easy it is
"""

# n_h, n_in = config.n_h, config.n_in
# w_rec = torch.reshape(w_rec, (n_h, n_h + n_in + 1))


def rnnStep(x: torch.Tensor, a: ACTIVATION, param: RnnParameter) -> ACTIVATION:
    a_ = param.w_rec @ torch.cat((x, a, torch.tensor([1.0])))
    return ACTIVATION((1 - param.alpha) * a + param.alpha * param.activationFn(a_))


# def rnnStep(
#     config: WithRnnConfig, x: torch.Tensor, a: torch.Tensor, w_rec: torch.Tensor
# ) -> torch.Tensor:
#     n_h, n_in = config.n_h, config.n_in
#     w_rec = torch.reshape(w_rec, (n_h, n_h + n_in + 1))
#     input_combined = torch.cat((x, torch.tensor([1.0], device=x.device)), dim=-1)
#     input_combined = torch.atleast_2d(input_combined)
#     a_combined = torch.atleast_2d(a)
#     rnn = torch.nn.RNN(
#         input_size=n_in + 1,  # Including the bias
#         hidden_size=n_h,
#         nonlinearity="tanh",  # Or any other activation function you prefer
#         bias=False,
#     )
#     rnn.weight_hh_l0.data = w_rec[:, :n_h]
#     rnn.weight_ih_l0.data = w_rec[:, n_h : n_h + n_in + 1]
#     _, hidden = rnn(input_combined, a_combined)  # Adding batch dimension

#     return (1 - config.alpha) * a + config.alpha * hidden[0]

# w_out = torch.reshape(w_out, (config.n_out, config.n_h + 1))


def rnnReadout(_: torch.Tensor, a: ACTIVATION, param: RnnParameter) -> PREDICTION:
    return PREDICTION(param.w_out @ torch.cat((a, torch.tensor([1.0]))))


def trainStep(unbatched_time_series: Input2Output1):

    n_h = 30
    n_in = 2
    n_out = 1
    learning_rate = 0.1
    alpha = 1.0

    dialect = RnnLearnableInterpreter[RnnFutureFaceState[RnnParameter, SgdParameter]]()
    dataDialect = Input2Output1Interpreter()

    parameters = torch.randn(n_h * (n_h + n_in + 1) + n_out * (n_h + 1))
    w_rec, w_out = rnnSplitParameters(n_h, n_in, n_out, parameters)
    parameter = RnnParameter(
        w_rec=PARAMETER(w_rec),
        w_out=PARAMETER(w_out),
        n_h=n_h,
        n_in=n_in,
        n_out=n_out,
        alpha=alpha,
        activationFn=torch.tanh,
    )
    sgd = SgdParameter(learning_rate=LEARNING_RATE(torch.tensor(learning_rate)))

    initEnv = RnnFutureFaceState[RnnParameter, SgdParameter](
        # activation=ACTIVATION(torch.randn(batch_size, n_h)),
        activation=ACTIVATION(torch.randn(n_h)),
        parameter=parameter,
        hyperparameter=sgd,
    )

    # y = PREDICTION(torch.tensor(0.0))
    rnnLibrary: RnnLibrary[
        RnnFutureFaceState[RnnParameter, SgdParameter], PREDICTION, Input2Output1
    ] = createRnnLibrary(
        dialect,
        dataDialect,
        lambda x, a, w: a,
        lambda _, a, w: PREDICTION(a),
        # rnnStep,
        # rnnReadout,
    )

    rnnLearner = offlineLearning(
        dialect,
        dataDialect,
        lambda a, b: LOSS(torch.functional.F.mse_loss(a, b)),
        rnnLibrary,
    )

    predictions, _ = rnnLearner.rnn.run(unbatched_time_series, initEnv)
    return predictions


# %%

length = 100000
random_data = torch.randn(length, 3)
dataclass_list = map(lambda row: Input2Output1(x=row[0:2], y=row[2]), random_data)

import cProfile, pstats

data = tree_stack(dataclass_list)

profiler = cProfile.Profile()
profiler.enable()
predictions = trainStep(data)
profiler.disable()
stats = pstats.Stats(profiler).sort_stats("cumtime")
stats.print_stats()
stats.dump_stats("profile_results.prof")

# predictions = list(predictions)
# indices = torch.arange(len(predictions))

# # Plot the data
# plt.figure(figsize=(8, 5))
# plt.plot(indices, predictions, marker="o", label="Data")
# plt.title("Plot of List Data with Indices")
# plt.xlabel("Indices")
# plt.ylabel("Values")
# plt.grid(True)
# plt.legend()
# plt.show()

# %%


# Define the simple RNN model
class SimpleRNN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        # Define the RNN layer
        self.rnn = torch.nn.RNN(input_size, hidden_size)
        # Define a fully connected layer to map RNN output to final prediction
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(1, self.hidden_size)
        # Get RNN output (output, hidden_state)
        out, _ = self.rnn(x, h0)
        # Pass the RNN output through the fully connected layer for each time step
        out = self.fc(out)  # Apply to all time steps
        return out


random_data = torch.randn(length, 2)

# Initialize the model
input_size = 2  # Corresponding to x1, x2, and y
hidden_size = 30  # Arbitrary size for hidden state
output_size = 1  # We are predicting a single output value

model = SimpleRNN(input_size, hidden_size, output_size)

# Generate predictions for the entire sequence
with torch.no_grad():  # No need to compute gradients
    start_time_vmap = time.time()
    predictions = model(random_data)
    vmap_time = time.time() - start_time_vmap
    print(f"vmap execution time: {vmap_time:.6f} seconds")

print(predictions.shape)
# %%
