from typing import Callable, Iterable, Tuple, TypeVar, Generic, cast
from myfunc import foldr
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
        return ReaderState.ask().bind(
            lambda env: ReaderState.get().fmap(lambda state: (env, state))
        )


class State(Generic[S, A]):
    def __init__(self, run: Callable[[S], Tuple[A, S]]):
        self.run = run

    def fmap(self, func: Callable[[A], B]) -> "State[S, B]":
        def new_run(state: S) -> Tuple[B, S]:
            value, new_state = self.run(state)
            return func(value), new_state

        return State(new_run)

    def bind(self, func: Callable[[A], "State[S, B]"]) -> "State[S, B]":
        def new_run(state: S) -> Tuple[B, S]:
            value, new_state = self.run(state)
            return func(value).run(new_state)

        return State(new_run)

    @staticmethod
    def pure(value: A) -> "State[S, A]":
        return State(lambda state: (value, state))

    @staticmethod
    def get() -> "State[S, S]":
        return State(lambda state: (state, state))

    @staticmethod
    def put(new_state: S) -> "State[S, Unit]":
        return State(lambda _: (Unit(), new_state))


def reader(run: Callable[[R, S], Tuple[A, S]]) -> ReaderState[R, S, A]:
    return ReaderState[R, S, A](run)


def runReader(reader_state: ReaderState[R, S, A]) -> Callable[[R], State[S, A]]:
    def env_to_state(env: R) -> State[S, A]:
        def state_function(state: S) -> Tuple[A, S]:
            return reader_state.run(env, state)

        return State(state_function)

    return env_to_state


def toReader(reader_state: Callable[[R], State[S, A]]) -> ReaderState[R, S, A]:
    def run(env: R, state: S) -> Tuple[A, S]:
        state_monad = reader_state(env)
        result, new_state = state_monad.run(state)
        return result, new_state

    return ReaderState(run)


def foldM(
    func: Callable[[A, B], State[S, B]], init: B
) -> Callable[[Iterable[A]], State[S, B]]:
    def foldM_(xs: Iterable[A]) -> State[S, B]:
        def fold(x: A, st: State[S, B]) -> State[S, B]:
            return st.bind(lambda acc: func(x, acc))

        return foldr(fold)(xs, State[S, B].pure(init))

    return foldM_


def traverse(
    func: Callable[[A], State[S, B]]
) -> Callable[[Iterable[A]], State[S, Iterable[B]]]:

    def traverse_(iterable: Iterable[A]) -> State[S, Iterable[B]]:
        def go(a: A, acc: PVector[B]) -> State[S, PVector[B]]:
            return func(a).fmap(lambda b: acc.append(b))

        init: PVector[B] = pvector([])
        return foldM(go, init)(iterable).fmap(lambda x: cast(Iterable[B], x))

    return traverse_
