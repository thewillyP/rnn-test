from donotation import do

from typing import Callable, Generator, Generic, Never, TypeVar

import torch
from torch import Tensor

S = TypeVar("S")
A = TypeVar("A")
B = TypeVar("B")
E = TypeVar("E")


class Unit:
    __slots__ = ()


class Proxy(Generic[A]):
    __slots__ = ()


# class StateMonad(Generic[S, A]):
#     def __init__(self, func: Callable[[S], tuple[A, S]]):
#         self.func = func

#     def __iter__(self) -> Generator[None, None, A]: ...

#     def flat_map(self, func: Callable[[A], "StateMonad[S, B]"]):
#         def next(state) -> tuple[B, S]:
#             value, n_state = self.func(state)
#             return func(value).func(n_state)

#         return StateMonad[S, B](next)


class Fold(Generic[E, S, A]):

    __slots__ = ("func",)

    def __init__(self, func: Callable[[E, S], tuple[A, S]]):
        self.func = func

    def __iter__(self) -> Generator[None, None, A]: ...

    def flat_map(self, func: Callable[[A], "Fold[E, S, B]"]):
        def next(env: E, state: S) -> tuple[B, S]:
            value, n_state = self.func(env, state)
            return func(value).func(env, n_state)

        return Fold[E, S, B](next)


def pure(value: A) -> Fold[E, S, A]:
    return Fold(lambda _, state: (value, state))


def get(_: Proxy[S]) -> Fold[E, S, S]:
    return Fold(lambda _, state: (state, state))


# class Get[E, S]:
#     def __new__(cls, /) -> Fold[E, S, S]:
#         return Fold(lambda _, state: (state, state))


def put(new_state: S) -> Fold[E, S, Unit]:
    return Fold(lambda _, __: (Unit(), new_state))


def ask(_: Proxy[E]) -> Fold[E, S, E]:
    return Fold(lambda env, state: (env, state))


# def collect_even_numbers(num: int):
#     def func(state: set[int]):
#         if num % 2 == 0:
#             state = state | {num}

#         return num, state

#     return StateMonad(func)


# @do()
# def example(init: int) -> Generator[None, None, Fold[int, int, int]]:
#     x = yield from get(Proxy[int]())
#     y = yield from pure(x + 1)
#     z = yield from put(y + 1)
#     r = yield from get(Proxy[int]())
#     return pure(y + r)


# # @do()
# # def example(init: int) -> Generator[None,xNone, StateMonad[set[int], int]]:
# #     x = yield from collect_even_numbers(init+1)
# #     y = yield from collect_even_numbers(x+1)
# #     z = yield from collect_even_numbers(y+1)
# #     return pure(z+1)

# # Correct type hint is inferred
# m = example(3)
# print(m.func(1, 1))


# @do()
# def example(
#     target: Tensor,
# ) -> Generator[None, None, Fold[str, Tensor, Tensor]]:
#     x = yield from get(Proxy[Tensor]())
#     loss = yield from pure(torch.nn.functional.mse_loss(x**2, target))
#     e = 2 * loss
#     s: str
#     s = yield from ask(Proxy[str]())

#     def temp(x: Tensor):
#         return 1 * x

#     t = 2 * e
#     return pure(temp(loss))


# class Test:

#     @do()
#     def test(self, target: Tensor) -> Generator[None, None, Fold[str, Tensor, Tensor]]:
#         x = yield from get(Proxy[Tensor]())
#         loss = yield from pure(torch.nn.functional.mse_loss(x**2, target))
#         e = 2 * loss
#         s: str
#         s = yield from ask(Proxy[str]())

#         def temp(x: Tensor):
#             return 1 * x

#         t = 2 * e
#         return pure(torch.tensor([temp(loss), t]))


# target = torch.tensor([1.0, 2.0, 3.0])
# init = torch.tensor([1.0, 2.0, 3.0])

# m = example(target)
# f: Callable[[Tensor], tuple[Tensor, Tensor]]
# f = lambda s: m.func("hi", s)[0]


# def test():
#     for _ in range(100_000):
#         j = f(init)
#         # j = torch.func.jvp(f, (init,), (torch.ones_like(init),))


# import cProfile, pstats

# profiler = cProfile.Profile()
# profiler.enable()
# test()
# profiler.disable()
# stats = pstats.Stats(profiler).sort_stats("cumtime")
# stats.print_stats()
# stats.dump_stats("profile_results.prof")


# # t = Test()
# # y = t.test(target).func("hi", init)[0]
# # print(y)


# def f(z):
#     a = torch.sum(z**2)
#     b = torch.sum(2 * z)
#     return torch.cat([a.unsqueeze(0), b.unsqueeze(0)])


# x = torch.tensor([2.0, 3.0])

# jac = torch.func.jacrev(f)(x)

# y = torch.tensor([2.0, 3.0])
# z = y @ jac
# print(z)
# print(z.shape)

# x_ = torch.tensor(x, requires_grad=True)
# l = f(x_)
# l.backward()
# print(x_.grad)


from typing import Generator, Generic, TypeVar
from donotation import do


class Test[T: int]():
    @do()
    def process[
        E: float
    ](self, xx: T, yy: E) -> Generator[Never, Never, Fold[int, int, float]]:
        # Step 1: Use the input value `x`
        # y = yield from pure(2)
        y = yield from pure(xx)

        # Step 2: Apply some transformation
        # z = y * 2  # Example transformation for numeric `T`

        # Step 3: Return the transformed value
        return pure(y * 2 + yy)


a = Test()
print(a.process(1, 2).func(1, 1))
