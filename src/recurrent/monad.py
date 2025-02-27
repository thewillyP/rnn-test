from dataclasses import dataclass
from typing import Callable, Generator, NamedTuple

from donotation import do
from pyrsistent import pvector
from pyrsistent.typing import PVector

from recurrent.myfunc import compose2, foldr
import jax
import jax.numpy as jnp
import equinox as eqx

from recurrent.mytypes import Traversable, G


class Unit(NamedTuple): ...


class PX[T](NamedTuple): ...


type Fold[D, I, S, A] = Callable[[D], App[I, S, A]]


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


class App[I, S, A](NamedTuple):
    func: Callable[[I, S], tuple[Maybe[A], S]]

    def __iter__(self) -> Generator[None, None, A]: ...  # type: ignore

    def flat_map[B](self, func: Callable[[A], "App[I, S, B]"]):
        def next(i: I, s: S) -> tuple[Maybe[B], S]:
            value, n_state = self.func(i, s)

            shape_info: Maybe[B] = jax.eval_shape(lambda x: func(x).func(i, n_state)[0], value.payload)
            return jax.lax.cond(
                value.isJust,
                lambda _: func(value.payload).func(i, n_state),
                lambda _: (nothing_shape(shape_info), n_state),
                None,
            )

        return App(next)

    def fmap[B](self, func: Callable[[A], B]):
        def next(i: I, s: S) -> tuple[Maybe[B], S]:
            value, n_state = self.func(i, s)

            shape_info: B = jax.eval_shape(lambda x: func(x), value.payload)
            return jax.lax.cond(
                value.isJust,
                lambda _: (just(func(value.payload)), n_state),
                lambda _: (nothing_shape(shape_info), n_state),
                None,
            )

        return App(next)

    def then[B](self, m: "App[I, S, B]"):
        return self.flat_map(lambda _: m)

    def switch_dl[Pidgen](self, dl: I):
        def next(_: Pidgen, s: S):
            return self.func(dl, s)

        return App(next)  # type: ignore


def pure[I, S, A](value: A, _: PX[tuple[I, S]]) -> App[I, S, A]:
    return App(lambda _i, s: (just(value), s))


def app_nothing[I, S, A](value: A, _: PX[tuple[I, S]]) -> App[I, S, A]:
    return App(lambda _i, s: (nothing(value), s))


def lift[I, S, A](value: Maybe[A], _: PX[tuple[I, S]]) -> App[I, S, A]:
    return App(lambda _i, s: (value, s))


def get[I, S](_: PX[S]) -> App[I, S, S]:
    return App(lambda _i, s: (just(s), s))


def put[I, S](new_state: S) -> App[I, S, Unit]:
    return App(lambda _i, _s: (just(Unit()), new_state))


def ask[I, S](_: PX[I]) -> App[I, S, I]:
    return App(lambda _i, s: (just(_i), s))


def asks[I, S, A](func: Callable[[I], A]) -> App[I, S, A]:
    return App(lambda _i, s: (just(func(_i)), s))


def gets[I, S, A](func: Callable[[S], A]) -> App[I, S, A]:
    return App(lambda _i, s: (just(func(s)), s))


def modifies[I, S](func: Callable[[S], S]) -> App[I, S, Unit]:
    return App(lambda _i, s: (just(Unit()), func(s)))


@do()
def modifies_[I, S, A](func: Callable[[S], tuple[A, S]]) -> G[App[I, S, A]]:
    a, s = yield from gets(func)
    _ = yield from put(s)
    return pure(a, PX[I, S]())


def kleisli[I, S, A, B, C](mf1: Fold[A, I, S, B], mf2: Fold[B, I, S, C]) -> Fold[A, I, S, C]:
    return lambda a: mf1(a).flat_map(mf2)


def kleisli2[I, S, A, B, C](mf1: Fold[A, I, S, B], f: Callable[[B], C]) -> Fold[A, I, S, C]:
    f_ = compose2(f, lambda c: pure(c, PX[tuple[I, S]]()))
    return kleisli(mf1, f_)


def sequenceM[D, I, S, A, B](mf1: Fold[D, I, S, A], mf2: Fold[D, I, S, B]) -> Fold[D, I, S, B]:
    return lambda d: mf1(d).then(mf2(d))


def repeatM[D, I, S, A](mf: Fold[D, I, S, A]) -> Fold[Traversable[D], I, S, Unit]:
    return kleisli2(traverseM(mf), lambda _: Unit())


def accumulateM[D, I, S, A](mf: Fold[D, I, S, A], fn: Callable[[A, A], A], init: A) -> Fold[Traversable[D], I, S, A]:
    return foldM(lambda accum: kleisli2(mf, lambda value: fn(accum, value)), init)


def traverseM[D, I, S, A](mf: Fold[D, I, S, A]) -> Fold[Traversable[D], I, S, Traversable[A]]:
    def fold_(ds: Traversable[D]) -> App[I, S, Traversable[A]]:
        def wrapper(i: I, s: S) -> tuple[Maybe[Traversable[A]], S]:
            def traverse(s_prev: S, d: D) -> tuple[S, Maybe[A]]:
                b, s_curr = mf(d).func(i, s_prev)
                return s_curr, b

            new_state, bs = jax.lax.scan(traverse, s, ds.value)
            bs_check = Maybe(jnp.all(bs.isJust), Traversable(bs.payload))
            return bs_check, new_state

        return App(wrapper)

    return fold_


def foldM[D, I, S, A](func: Callable[[A], Fold[D, I, S, A]], init: A) -> Fold[Traversable[D], I, S, A]:
    def fold_(ds: Traversable[D]):
        def wrapper(i: I, s: S) -> tuple[Maybe[A], S]:
            def fold(pair: tuple[Maybe[A], S], d: D) -> tuple[tuple[Maybe[A], S], None]:
                acc, s = pair
                pair_new = jax.lax.cond(
                    acc.isJust,
                    lambda _: func(acc.payload)(d).func(i, s),
                    lambda _: (nothing(acc.payload), s),
                    None,
                )
                return pair_new, None

            accum, _ = jax.lax.scan(fold, (just(init), s), ds.value)
            return accum

        return App(wrapper)

    return fold_
