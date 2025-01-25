from typing import TypeVar, Callable, Generator, Iterator, Iterable
from functools import reduce

# from toolz.curried import curry, map, concat, compose
# import itertools
from recurrent.mytypes import *


def foldr[A, B](f: Callable[[A, B], B]) -> Callable[[Iterable[A], B], B]:
    def foldr_(xs: Iterable[A], x: B) -> B:
        result = x
        for item in xs:
            result = f(item, result)
        return result

    return foldr_


def flipTuple[A, B](pair: tuple[A, B]) -> tuple[B, A]:
    a, b = pair
    return (b, a)


# def sequenceF2(
#     fs: Iterable[Callable[[A, B], C]]
# ) -> Callable[[A], Iterator[Callable[[B], C]]]:
#     return lambda a: map(lambda f: curry(f)(a), fs)


# def collapseF(fs: Iterable[Callable[[A, B], B]]) -> Callable[[A, B], B]:
#     # return lambda a, b: foldr(compose2)(map(lambda f: curry(f)(a), fs), id)(b)
#     def collapseF_(a: A, b: B) -> B:
#         for f in fs:
#             b = f(a, b)
#         return b

#     return collapseF_


# def sequenceA(fs: Iterable[Callable[[A], B]]) -> Callable[[A], Iterator[B]]:
#     return lambda a: map(lambda f: f(a), fs)


# def liftAN(f: Callable[[B, B], B], fs: Iterable[Callable[[A], B]]) -> Callable[[A], B]:
#     return lambda a: reduce(f, map(lambda f: f(a), fs))


# @curry
# def scan(f: Callable[[X, T], T], it: Iterable[X], state: T) -> Generator[T, None, None]:
#     yield state
#     for x in it:
#         state = f(x, state)
#         yield state


# def scan0(
#     f: Callable[[X, T], T], it: Iterable[X], state: T
# ) -> Generator[T, None, None]:
#     for x in it:
#         state = f(x, state)
#         yield state


# @curry
# def uncurry(f: Callable[[T, X], Y]) -> Callable[[tuple[T, X]], Y]:
#     def _uncurry(pair):
#         x, y = pair
#         return f(x, y)

#     return _uncurry


# @curry
# def swap(f: Callable[[X, Y], T]) -> Callable[[Y, X], T]:
#     def swap_(y, x):
#         return f(x, y)

#     return swap_


# @curry
# def map2(f1: Callable, f2: Callable):
#     return map(uncurry(lambda x, y: (f1(x), f2(y))))


# def fst(pair: tuple[X, Y]) -> X:
#     x, _ = pair
#     return x


# def snd(pair: tuple[X, Y]) -> Y:
#     _, y = pair
#     return y


# # def flatFst(stream: Iterator[tuple[Iterator[X], Y]]) -> Iterator[tuple[X, Y]]:
# #     i1 = concat(map(fst, stream))
# #     i2 = map(snd, stream)
# #     return map(lambda x, y: (x, y), i1, i2)


# reduce_ = curry(lambda fn, x, xs: reduce(fn, xs, x))


# # reverse of sequenceA? which doesn't exist so custom logic
# @curry
# def traverseTuple(pair: tuple[Iterable[X], Y]) -> Iterator[tuple[X, Y]]:
#     xs, y = pair
#     return ((x, y) for x in xs)


# @curry
# def mapTuple1(f, pair):
#     a, b = pair
#     return (f(a), b)


# cycle_efficient = compose(itertools.chain.from_iterable, itertools.repeat)


# def composeSnd(f: Callable[[A, B], C], g: Callable[[C], D]) -> Callable[[A, B], D]:
#     def composeSnd_(a: A, b: B) -> D:
#         return g(f(a, b))

#     return composeSnd_


# def liftA2(
#     f: Callable[[A, B], C], g: Callable[[D], A], h: Callable[[D], B]
# ) -> Callable[[D], C]:
#     def liftA2_(d: D) -> C:
#         return f(g(d), h(d))

#     return liftA2_


# def liftA2Compose(
#     g: Callable[[E, B], C], h: Callable[[E, C], D]
# ) -> Callable[[E, B], D]:
#     def _liftA2Compose(e: E, b: B) -> D:
#         return h(e, g(e, b))

#     return _liftA2Compose


# # def liftA1(f: Callable[[B], C], g: Callable[[A], B]) -> Callable[[A], C]:
# #     return flip(compose2)


# def flip(f: Callable[[A, B], C]) -> Callable[[B, A], C]:
#     def flip_(b: B, a: A) -> C:
#         return f(a, b)

#     return flip_


# def apply(f: Callable[[A], B], x: A) -> B:
#     return f(x)


# def const(x: A) -> Callable[[B], A]:
#     return lambda _: x


# def compose2(f: Callable[[A], B], g: Callable[[B], C]) -> Callable[[A], C]:
#     def compose2_(a: A) -> C:
#         return g(f(a))

#     return compose2_


# def fmap(g: Callable[[B], C], f: Callable[[A], B]) -> Callable[[A], C]:
#     # fmap = flip(compose2)  # want the syntax highlight so resort to this code dup
#     def fmap_(a: A) -> C:
#         return g(f(a))

#     return fmap_


# # def foldr(f: Callable[[A, B], B]) -> Callable[[Iterable[A], B], B]:
# #     def foldr_(xs: Iterable[A], x: B) -> B:
# #         return reduce(flip(f), xs, x)

# #     return foldr_


# def foldrWithLen(f: Callable[[A, B], B]) -> Callable[[Iterable[A], B], tuple[B, int]]:
#     def foldrWithLen_(x: A, pair: tuple[B, int]) -> tuple[B, int]:
#         b, i = pair
#         return (f(x, b), i + 1)

#     return lambda xs, x: foldr(foldrWithLen_)(xs, (x, 0))


# def foldl(f: Callable[[B, A], B]) -> Callable[[B, Iterable[A]], B]:
#     def foldl_(
#         x: B,
#         xs: Iterable[A],
#     ) -> B:
#         return reduce(f, xs, x)

#     return foldl_


# def fuse(
#     f: Callable[[X, A], B], g: Callable[[Y, B], C]
# ) -> Callable[[tuple[X, Y], A], C]:
#     """g . f"""

#     def fuse_(pair: tuple[X, Y], a: A) -> C:
#         x, y = pair
#         return g(y, f(x, a))

#     return fuse_


# def fmapSuffix(g: Callable[[B], C], f: Callable[[X, A], B]) -> Callable[[X, A], C]:
#     def fmapSuffix_(x: X, a: A) -> C:
#         return g(f(x, a))

#     return fmapSuffix_


# def fmapPrefix(g: Callable[[A], B], f: Callable[[X, B], C]) -> Callable[[X, A], C]:
#     def fmapPrefix_(x: X, a: A) -> C:
#         return f(x, g(a))

#     return fmapPrefix_
