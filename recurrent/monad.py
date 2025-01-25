from typing import Callable, Generator, Iterable, NamedTuple

from donotation import do
from pyrsistent import pvector
from pyrsistent.typing import PVector

from recurrent.myfunc import foldr
import jax

from recurrent.mytypes import Traversable, G


class Unit(NamedTuple): ...


class Fold[D, E, S, A](NamedTuple):

    func: Callable[[D, E, S], tuple[A, S]]

    def __iter__(self) -> Generator[None, None, A]: ...  # type: ignore

    def flat_map[B](self, func: Callable[[A], "Fold[D, E, S, B]"]):
        def next(d: D, env: E, state: S) -> tuple[B, S]:
            value, n_state = self.func(d, env, state)
            return func(value).func(d, env, n_state)

        return Fold(next)

    def fmap[B](self, func: Callable[[A], B]):
        def next(d: D, env: E, state: S) -> tuple[B, S]:
            value, n_state = self.func(d, env, state)
            return func(value), n_state

        return Fold(next)

    def then[B](self, m: "Fold[D, E, S, B]"):
        return self.flat_map(lambda _: m)

    def switch_dl[Pidgen](self, dl: D):
        def next(_: Pidgen, env: E, state: S):
            return self.func(dl, env, state)

        return Fold(next)  # type: ignore


def pure[D, E, S, A](value: A) -> Fold[D, E, S, A]:
    return Fold(lambda _d, _e, state: (value, state))


def get[D, E, S]() -> Fold[D, E, S, S]:
    return Fold(lambda _d, _e, state: (state, state))


class ProxyS[S]:
    @staticmethod
    def get[D, E]() -> Fold[D, E, S, S]:
        return Fold(lambda _d, _e, state: (state, state))


def put[D, E, S](new_state: S) -> Fold[D, E, S, Unit]:
    return Fold(lambda _d, _e, _s: (Unit(), new_state))


def ask[D, E, S]() -> Fold[D, E, S, E]:
    return Fold(lambda _d, env, state: (env, state))


class ProxyR[E]:
    @staticmethod
    def ask[D, S]() -> Fold[D, E, S, E]:
        return Fold(lambda _d, env, state: (env, state))


def asks[D, E, S, A](func: Callable[[E], A]) -> Fold[D, E, S, A]:
    return Fold(lambda _d, env, state: (func(env), state))


def gets[D, E, S, A](func: Callable[[S], A]) -> Fold[D, E, S, A]:
    return Fold(lambda _d, _e, state: (func(state), state))


def modifies[D, E, S](func: Callable[[S], S]) -> Fold[D, E, S, Unit]:
    return Fold(lambda _d, _e, state: (Unit(), func(state)))


@do()
def modifies_[D, E, S, A](func: Callable[[S], tuple[A, S]]) -> G[Fold[D, E, S, A]]:
    a, s = yield from gets(func)
    _ = yield from put(s)
    return pure(a)


def askDl[D, E, S]() -> Fold[D, E, S, D]:
    return Fold(lambda d, _e, state: (d, state))


class ProxyDl[D]:
    @staticmethod
    def askDl[E, S]() -> Fold[D, E, S, D]:
        return Fold(lambda d, _e, state: (d, state))


def stateToFold[D, E, S, T](func: Callable[[S], tuple[T, S]]) -> Fold[D, E, S, T]:
    return Fold(lambda _d, _e, state: func(state))


def toFold[D, E, S, T](func: Callable[[E, S], tuple[T, S]]):
    return Fold[D, E, S, T](lambda _d, env, state: func(env, state))  # type: ignore


# def repeatM[Dl, D, E, A](m: "Fold[Dl, D, E, A]"):
#     def wrapper(dl: Dl, ds: Iterable[D], state: E) -> tuple[Unit, E]:
#         def repeat(d: D, pair: tuple[Unit, E]) -> tuple[Unit, E]:
#             _u, state = pair
#             __, new_state = m.func(dl, d, state)
#             return _u, new_state

#         return foldr(repeat)(ds, (Unit(), state))

#     return Fold(wrapper)


# these guys need to be strict
# def foldM[Dl, D, E, B](func: Callable[[B], "Fold[Dl, D, E, B]"], init: B):
#     def wrapper(dl: Dl, ds: Iterable[D], state: E) -> tuple[B, E]:
#         def fold(d: D, pair: tuple[B, E]) -> tuple[B, E]:
#             acc, state = pair
#             return func(acc).func(dl, d, state)

#         return foldr(fold)(ds, (init, state))

#     return Fold(wrapper)


def foldM[Dl, D, E, B](func: Callable[[B], Fold[Dl, D, E, B]], init: B):
    def wrapper(dl: Dl, ds: Traversable[D], state: E):
        def fold(pair: tuple[B, E], d: D):
            acc, state = pair
            return func(acc).func(dl, d, state), None

        return jax.lax.scan(fold, (init, state), ds.value)[0]

    return Fold(wrapper)


def traverse[Dl, D, E, B](m: "Fold[Dl, D, E, B]"):

    def wrapper(dl: Dl, ds: Traversable[D], state: E) -> tuple[Traversable[B], E]:
        def traverse_(s_prev: E, d: D):
            b, s = m.func(dl, d, s_prev)
            return s, b

        new_state, bs = jax.lax.scan(traverse_, state, ds.value)
        return Traversable(bs), new_state

    return Fold(wrapper)


# def reader(run: Callable[[R, S], Tuple[A, S]]) -> ReaderState[R, S, A]:
#     return ReaderState[R, S, A](run)


# # these guys need to be strict
# def foldM(
#     func: Callable[[B], ReaderState[A, S, B]], init: B
# ) -> ReaderState[Iterable[A], S, B]:
#     def wrapper(xs: Iterable[A], state: S) -> tuple[B, S]:
#         def fold(x: A, pair: tuple[B, S]) -> tuple[B, S]:
#             acc, state = pair
#             return func(acc).run(x, state)

#         return foldr(fold)(xs, (init, state))

#     return ReaderState[Iterable[A], S, B](wrapper)


# def executeM(func: ReaderState[A, S, B]) -> ReaderState[Iterable[A], S, S]:
#     def call(a: A, state: S) -> S:
#         _, state = func.run(a, state)
#         return state

#     def wrapper(xs: Iterable[A], state: S) -> tuple[S, S]:
#         s = foldr(call)(xs, state)
#         return s, s

#     return ReaderState[Iterable[A], S, S](wrapper)


# def traverse(func: ReaderState[A, S, B]) -> ReaderState[Iterable[A], S, Iterable[B]]:

#     def traverse_(x: A, pair: tuple[PVector[B], S]) -> tuple[PVector[B], S]:
#         acc, state = pair
#         b, s = func.run(x, state)
#         return acc.append(b), s

#     def wrapper(xs: Iterable[A], state: S) -> tuple[PVector[B], S]:
#         return foldr(traverse_)(xs, (pvector([]), state))

#     return ReaderState[Iterable[A], S, Iterable[B]](wrapper)
