from typing import Callable, Iterable, Tuple, TypeVar, Generic, cast
from recurrent.myfunc import foldr
from pyrsistent import pvector, PVector

R = TypeVar("R")  # Environment type
S = TypeVar("S")  # State type
A = TypeVar("A")  # Result type
B = TypeVar("B")  # Result type


class Unit:
    pass


class ReaderState(Generic[R, S, A]):
    def __init__(self, run: Callable[[R, S], Tuple[A, S]]):
        self.run = run

    def fmap(self, func: Callable[[A], B]) -> "ReaderState[R, S, B]":
        def new_run(env: R, state: S) -> Tuple[B, S]:
            value, new_state = self.run(env, state)
            return func(value), new_state

        return ReaderState(new_run)

    def bind(
        self, func: Callable[[A], "ReaderState[R, S, B]"]
    ) -> "ReaderState[R, S, B]":
        def new_run(env: R, state: S) -> Tuple[B, S]:
            value, new_state = self.run(env, state)
            return func(value).run(env, new_state)

        return ReaderState[R, S, B](new_run)

    def then(self, monad: "ReaderState[R, S, B]") -> "ReaderState[R, S, B]":
        return self.bind(lambda _: monad)

    @staticmethod
    def pure(value: A) -> "ReaderState[R, S, A]":
        return ReaderState(lambda _, state: (value, state))

    @staticmethod
    def get() -> "ReaderState[R, S, S]":
        return ReaderState(lambda _, state: (state, state))

    @staticmethod
    def put(new_state: S) -> "ReaderState[R, S, Unit]":
        return ReaderState(lambda _, __: (Unit(), new_state))

    @staticmethod
    def ask() -> "ReaderState[R, S, R]":
        return ReaderState(lambda env, state: (env, state))

    @staticmethod
    def ask_get() -> "ReaderState[R, S, Tuple[R, S]]":
        return ReaderState(lambda env, state: ((env, state), state))


def reader(run: Callable[[R, S], Tuple[A, S]]) -> ReaderState[R, S, A]:
    return ReaderState[R, S, A](run)


# these guys need to be strict
def foldM(
    func: Callable[[B], ReaderState[A, S, B]], init: B
) -> ReaderState[Iterable[A], S, B]:
    def wrapper(xs: Iterable[A], state: S) -> tuple[B, S]:
        def fold(x: A, pair: tuple[B, S]) -> tuple[B, S]:
            acc, state = pair
            return func(acc).run(x, state)

        return foldr(fold)(xs, (init, state))

    return ReaderState[Iterable[A], S, B](wrapper)


def traverse(func: ReaderState[A, S, B]) -> ReaderState[Iterable[A], S, Iterable[B]]:

    def traverse_(x: A, pair: tuple[PVector[B], S]) -> tuple[PVector[B], S]:
        acc, state = pair
        b, s = func.run(x, state)
        return acc.append(b), s

    def wrapper(xs: Iterable[A], state: S) -> tuple[PVector[B], S]:
        return foldr(traverse_)(xs, (pvector([]), state))

    return ReaderState[Iterable[A], S, Iterable[B]](wrapper)
