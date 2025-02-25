from dataclasses import dataclass
from typing import Callable, Generator, Iterable, NamedTuple

from donotation import do
from pyrsistent import pvector
from pyrsistent.typing import PVector

from recurrent.myfunc import foldr
import jax
import jax.numpy as jnp
import equinox as eqx

from recurrent.mytypes import Traversable, G


class Unit(NamedTuple): ...


class PX[T](NamedTuple): ...


class PX3[A, B, C](NamedTuple): ...


# maybe for jax
class Maybe[T](eqx.Module):
    isJust: jax.Array  # bool... why jax
    payload: T

    def __iter__(self) -> Generator[None, None, T]: ...  # type: ignore

    def flat_map[B](self, func: Callable[[T], "Maybe[B]"]) -> "Maybe[B]":
        shape_info = jax.eval_shape(func, self.payload)
        return jax.lax.cond(
            self.isJust,
            lambda _: func(self.payload),
            lambda _: nothing_shape(shape_info),
            None,
        )

    def fmap[B](self, func: Callable[[T], B]) -> "Maybe[B]":
        shape_info = jax.eval_shape(func, self.payload)
        return jax.lax.cond(
            self.isJust,
            lambda _: just(func(self.payload)),
            lambda _: nothing_shape(shape_info),
            None,
        )


def just[T](value: T) -> Maybe[T]:
    return Maybe(True, value)


def nothing[T](value: T) -> Maybe[T]:
    return Maybe(False, value)


def nothing_shape[T](shape_info) -> Maybe[T]:
    return nothing(jax.tree.map(lambda x: jnp.zeros(x.shape, dtype=x.dtype), shape_info))


class App[I, R, S, A](NamedTuple):
    func: Callable[[I, R, S], tuple[Maybe[A], S]]

    def __iter__(self) -> Generator[None, None, A]: ...  # type: ignore

    def flat_map[B](self, func: Callable[[A], "App[I, R, S, B]"]):
        def next(i: I, r: R, s: S) -> tuple[Maybe[B], S]:
            value, n_state = self.func(i, r, s)

            shape_info: Maybe[B] = jax.eval_shape(lambda x: func(x).func(i, r, n_state)[0], value.payload)
            return jax.lax.cond(
                value.isJust,
                lambda _: func(value.payload).func(i, r, n_state),
                lambda _: (nothing_shape(shape_info), n_state),
                None,
            )

        return App(next)

    def fmap[B](self, func: Callable[[A], B]):
        def next(i: I, r: R, s: S) -> tuple[Maybe[B], S]:
            value, n_state = self.func(i, r, s)

            shape_info: B = jax.eval_shape(lambda x: func(x), value.payload)
            return jax.lax.cond(
                value.isJust,
                lambda _: (just(func(value.payload)), n_state),
                lambda _: (nothing_shape(shape_info), n_state),
                None,
            )

        return App(next)

    def then[B](self, m: "App[I, R, S, B]"):
        return self.flat_map(lambda _: m)

    def switch_dl[Pidgen](self, dl: I):
        def next(_: Pidgen, r: R, s: S):
            return self.func(dl, r, s)

        return App(next)  # type: ignore

    def switch_data[R_new](self, data: R):
        def next(i: I, _: R_new, s: S):
            return self.func(i, data, s)

        return App(next)  # type: ignore


def pure[I, R, S, A](value: A, _: PX3[I, R, S]) -> App[I, R, S, A]:
    return App(lambda _i, _r, s: (just(value), s))


def app_nothing[I, R, S, A](value: A, _: PX3[I, R, S]) -> App[I, R, S, A]:
    return App(lambda _i, _r, s: (nothing(value), s))


def lift[I, R, S, A](value: Maybe[A], _: PX3[I, R, S]) -> App[I, R, S, A]:
    return App(lambda _i, _r, s: (value, s))


def get[I, R, S](_: PX[S]) -> App[I, R, S, S]:
    return App(lambda _i, _r, s: (just(s), s))


def put[I, R, S](new_state: S) -> App[I, R, S, Unit]:
    return App(lambda _i, _r, _s: (just(Unit()), new_state))


def ask[I, R, S](_: PX[R]) -> App[I, R, S, R]:
    return App(lambda _i, r, s: (just(r), s))


def asks[I, R, S, A](func: Callable[[R], A]) -> App[I, R, S, A]:
    return App(lambda _i, r, s: (just(func(r)), s))


def gets[I, R, S, A](func: Callable[[S], A]) -> App[I, R, S, A]:
    return App(lambda _i, _r, s: (just(func(s)), s))


def modifies[I, R, S](func: Callable[[S], S]) -> App[I, R, S, Unit]:
    return App(lambda _i, _r, s: (just(Unit()), func(s)))


@do()
def modifies_[I, R, S, A](func: Callable[[S], tuple[A, S]]) -> G[App[I, R, S, A]]:
    a, s = yield from gets(func)
    _ = yield from put(s)
    return pure(a, PX[I, R, S]())


def askForInterpreter[I, R, S](_: PX[I]) -> App[I, R, S, I]:
    return App(lambda i, _r, s: (just(i), s))


def repeatM[I, R, S, A](m: App[I, R, S, A]):
    return traverseM(m).fmap(lambda _: Unit())


def traverseM[I, R, S, A](m: App[I, R, S, A]):
    def wrapper(i: I, rs: Traversable[R], s: S) -> tuple[Maybe[Traversable[A]], S]:
        def traverse(s_prev: S, r: R):
            b, s_curr = m.func(i, r, s_prev)
            return s_curr, b

        new_state, bs = jax.lax.scan(traverse, s, rs.value)
        bs_check = Maybe(jnp.all(bs.isJust), Traversable(bs.payload))
        return bs_check, new_state

    return App(wrapper)


def accumulateM[I, R, S, A](m: App[I, R, S, A], fn: Callable[[A, A], A], init: A):
    return foldM(lambda accum: m.fmap(lambda value: fn(accum, value)), init)


def foldM[I, R, S, A](func: Callable[[A], App[I, R, S, A]], init: A):
    def wrapper(i: I, rs: Traversable[R], s: S):
        def fold(pair: tuple[Maybe[A], S], r: R) -> tuple[tuple[Maybe[A], S], None]:
            acc, s = pair
            pair_new = jax.lax.cond(
                acc.isJust,
                lambda _: func(acc.payload).func(i, r, s),
                lambda _: (nothing(acc.payload), s),
                None,
            )
            return pair_new, None

        accum, _ = jax.lax.scan(fold, (just(init), s), rs.value)
        return accum

    return App(wrapper)
