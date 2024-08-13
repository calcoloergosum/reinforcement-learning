"""Basic monte carlo tree search"""

import collections
from typing import Callable, Generic, List, Literal, Tuple, TypeVar, Union, Optional

import numpy as np
import tqdm
from typing_extensions import Self

from . import utils

T = TypeVar("T")
A = TypeVar("A")

class UCTNode(Generic[T, A]):
    __slots__ = [ "inner", "move", "is_expanded", "parent", "children",
                 "child_priors", "child_total_value", "child_number_visits", "turn"]

    def __init__(self, inner: T, move: A, turn: Union[Literal[-1], Literal[0], Literal[1]], parent: Optional[Self] = None):
        assert move is not None or isinstance(parent, self.DummyNode)
        self.inner = inner
        self.move = move
        self.is_expanded = False
        self.parent: Self | None = parent
        self.turn = turn
        self.children: dict[A, Self] = {}
        self.child_priors       : np.ndarray | None = None
        self.child_total_value  : np.ndarray | None = None
        self.child_number_visits: np.ndarray | None = None

    @property
    def number_visits(self) -> int:
        return self.parent.child_number_visits[self.move]

    @number_visits.setter
    def number_visits(self, value: int) -> int:
        self.parent.child_number_visits[self.move] = value

    @property
    def total_value(self) -> float:
        return self.parent.child_total_value[self.move]

    @total_value.setter
    def total_value(self, value: float) -> float:
        self.parent.child_total_value[self.move] = value

    def child_Q(self) -> float:
        child_total_value = self.child_total_value.copy()
        no_visit = self.child_number_visits == 0
        # child_total_value[no_visit] = 1e5 + 100 * np.random.random(no_visit.sum())
        child_total_value[no_visit] = self.child_priors[no_visit]
        return child_total_value / self.child_number_visits.clip(1)

    def child_U(self) -> float:
        return np.sqrt(2 * np.log(max(1, self.number_visits)) / self.child_number_visits.clip(1))

    def best_child(self) -> int:
        if self.turn == 1:
            return np.argmax(self.child_Q() + self.child_U())
        if self.turn == -1:
            return np.argmin(self.child_Q() + self.child_U())
        if self.turn == 0:
            return np.argmin(self.child_number_visits)
        raise NotImplementedError("Should be unreachable")

    def best_children(self, at_most: int) -> List[int]:
        """Returns at most `k` best children, sorted"""
        k = min(at_most, len(self.child_number_visits))
        if self.turn == 1:
            vals = self.child_Q() + self.child_U()
            idxs = utils.argmax_many(vals, k)
            return idxs[np.argsort(vals[idxs])][::-1]
        if self.turn == -1:
            vals = self.child_Q() + self.child_U()
            idxs = utils.argmin_many(vals, k)
            return idxs[np.argsort(vals[idxs])]
        if self.turn == 0:
            vals = self.child_number_visits
            idxs = utils.argmin_many(vals, k)
            np.random.shuffle(idxs)
            return idxs
        raise NotImplementedError("Should be unreachable")

    def select_leaf(self, play: Callable[[T, A], T],
                    turn: Callable[[T], Union[Literal[-1], Literal[0], Literal[1]]],) -> Self:
        current = self
        while current.is_expanded:
            best_move = current.best_child()
            _, current = current.maybe_add_child(best_move, play, turn)
        return current

    def select_leaves(self, play: Callable[[T, A], T],
                      turn: Callable[[T], Union[Literal[-1], Literal[0], Literal[1]]], n: int) -> List[Self]:
        if not self.is_expanded:
            return [self]

        ret = []
        best_moves = self.best_children(n)
        for m in best_moves:
            is_added, next = self.maybe_add_child(m, play, turn)
            if is_added:
                ret += [next]
            else:
                xs = list(next.select_leaves(play, turn, n))
                n -= len(xs)
                ret += xs
        return ret

    def expand(self, child_priors: np.ndarray) -> None:
        assert not self.is_expanded
        self.is_expanded = True
        self.child_priors        = child_priors
        self.child_total_value   = np.zeros(len(child_priors), dtype=np.float32)
        self.child_number_visits = np.zeros(len(child_priors), dtype=np.uint16)

    def maybe_add_child(self, move: A, play: Callable[[T, A], T],
                        turn: Callable[[T], Union[Literal[-1], Literal[0], Literal[1]]]) -> Tuple[bool, Self]:
        is_added = False
        if move not in self.children:
            next_state = play(self.inner, move)
            self.children[move] = UCTNode(next_state, move, parent=self, turn=turn(next_state))
            is_added = True
        return is_added, self.children[move]

    def backup(self, value_estimate: float) -> None:
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value_estimate
            current = current.parent

    class DummyNode:
        def __init__(self):
            self.parent = None
            self.child_total_value = collections.defaultdict(float)
            self.child_number_visits = collections.defaultdict(float)

        @property
        def number_visits(self) -> int:
            return self.child_number_visits[None]

    @classmethod
    def uct_search(cls,
                   game_state: T,
                   num_reads: int,
                   evaluate: Callable[[T], float],
                   play: Callable[[T, A], T],
                   turn: Callable[[T], Union[Literal[-1], Literal[0], Literal[1]]],
                   thread_pool,
    ) -> Tuple[Self, int]:
        root = cls(game_state, move=None, turn=turn(game_state), parent=cls.DummyNode())
        if thread_pool:
            return cls.uct_search_continue_parallel(root, num_reads, evaluate, play, turn, thread_pool)
        return cls.uct_search_continue(root, num_reads, evaluate, play, turn)

    @classmethod
    def uct_search_continue(cls,
        root: Self,
        num_reads: int,
        evaluate: Callable[[T], float],
        play: Callable[[T, A], T],
        turn: Callable[[T], Union[Literal[-1], Literal[0], Literal[1]]],
    ) -> Tuple[Self, int]:
        for _ in tqdm.tqdm(range(num_reads)):
            leaf = root.select_leaf(play, turn)
            leaf.maybe_expand_and_backup(evaluate)
        return root, np.argmax(root.child_number_visits)
    
    @classmethod
    def uct_search_continue_parallel(cls,
        root: Self,
        num_reads: int,
        evaluate: Callable[[T], float],
        play: Callable[[T, A], T],
        turn: Callable[[T], Union[Literal[-1], Literal[0], Literal[1]]],
        thread_pool,
    ) -> Tuple[Self, int]:
        if thread_pool is None:
            for _ in tqdm.tqdm(range(num_reads)):
                leaf = root.select_leaf(play, turn)
                leaf.maybe_expand_and_backup(evaluate)
            return root, np.argmax(root.child_number_visits)

        def expand(node: Self) -> float:
            child_priors, value_estimate = evaluate(node.inner)
            node.expand(child_priors)
            return value_estimate

        pool_size = len(thread_pool._pool)
        progress = tqdm.tqdm(range(num_reads))
        while num_reads > 0:
            leaves = root.select_leaves(play, turn, pool_size)
            value_estimates = thread_pool.imap(expand, leaves)
            for leaf, value_estimate in zip(leaves, value_estimates):
                leaf.backup(value_estimate)
            progress.update(min(len(leaves), num_reads))
            num_reads -= len(leaves)
        return root, np.argmax(root.child_number_visits)

    def safe_get_child(self, action,
        evaluate: Callable[[T], float],
        play: Callable[[T, A], T],
        turn: Callable[[T], Union[Literal[-1], Literal[0], Literal[1]]],
    ) -> Self:
        self.maybe_add_child(action, play, turn)
        child = self.children.get(action)
        child.maybe_expand_and_backup(evaluate)
        return child

    def maybe_expand_and_backup(
        self,
        evaluate: Callable[[T], float],
    ):
        if not self.is_expanded:
            child_priors, value_estimate = evaluate(self.inner)
            self.expand(child_priors)
            self.backup(value_estimate)
