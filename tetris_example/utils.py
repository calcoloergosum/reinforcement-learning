import heapq
import random
from collections import deque
from typing import Dict, Generic, List, TypeVar

import numpy as np


def argmax_many(l, k):
    idxs = np.argpartition(l, -k)[-k:]
    return idxs


def argmin_many(l, k):
    idxs = np.argpartition(l, k - 1)[:k]
    return idxs


class DequeSARSA:
    def __init__(self, maxlen: int, sa2t) -> None:
        self.sar = None
        self.queue = deque(maxlen=maxlen)
        self.sa2t = sa2t

    def push(self, sarsd):
        t_, r_, _, done = self.sa2t(*sarsd[:2]), *sarsd[2:]
        if self.sar is not None:
            self.queue.append((*self.sar, t_))

        if done:
            # assert self.sar is not None
            self.queue.append((*self.sar, None))
            self.sar = None
            return

        self.sar = (t_, r_)
        return

    def size(self):
        return len(self.queue)


class DequeSAR:
    def __init__(self, maxlen: int, acceptance_ratio: float, sa2t, s2t) -> None:
        self.queue = deque(maxlen=maxlen)
        self.acceptance_ratio = acceptance_ratio
        self.sa2t = sa2t
        self.s2t = s2t

    def push(self, sarsd):
        accept = random.random() < self.acceptance_ratio
        if accept:
            t_, r_, s, done = self.sa2t(*sarsd[:2]), sarsd[2], self.s2t(sarsd[3]), sarsd[4]
            self.queue.append((t_, r_, s, done))
        return

    def size(self):
        return len(self.queue)


def weighted_sample_without_replacement(population, weights, k, rng=random):
    v = [rng.random() ** (1 / w) for w in weights]
    order = sorted(range(len(population)), key=lambda i: v[i])
    return [population[i] for i in order[-k:]]


T = TypeVar("T")


class IndexedPriorityQueue(Generic[T]):
    def __init__(self, maxlen: int, on_pushpop) -> None:
        self.maxlen = maxlen
        self.heap: List[List[float, int, T]] = []
        self.i_push2item: Dict[int, List[float, int, T]] = {}
        self.n_push: int = 0
        self.on_pushpop = on_pushpop

    def push(self, priority: float, item: T):
        x = [priority, self.n_push, item]
        self.i_push2item[self.n_push] = x
        if self.size() < self.maxlen:
            heapq.heappush(self.heap, x)
            self.on_pushpop(self.n_push, None)
        elif self.size() > self.maxlen:
            raise RuntimeError("Should not reach here. " +
                               "Injecting item arbitrarily can lead to unexpected bahavior")
        else:
            self.i_push2item.keys()
            priority, i_push, item = heapq.heappushpop(self.heap, x)
            del self.i_push2item[i_push]
            assert self.size() == len(self.i_push2item)
            self.on_pushpop(self.n_push, i_push)

        self.n_push += 1
        return x

    def size(self) -> int:
        return len(self.heap)

    def update(self) -> None:
        heapq.heapify(self.heap)


class DiscountedUCB:
    def __init__(self, maxlen: int, interval_constant: float, discount: float) -> None:
        assert 0 <= discount <= 1
        self.interval_constant = interval_constant
        self.i2sum = np.zeros(maxlen)
        self.i2n = np.zeros(maxlen)
        self.discount = discount
        self.n_arms = 0

        # for logging
        self.i2n_raw = np.zeros(maxlen)

    def add_arm(self) -> int:
        self.n_arms += 1
        return self.n_arms - 1

    def weights(self) -> np.ndarray:
        is_zero = np.asarray(self.i2n[:self.n_arms] == 0)
        i2sum = self.i2sum[:self.n_arms] + 1e5 * is_zero
        i2n = self.i2n[:self.n_arms] + is_zero
        avg = i2sum / i2n

        if self.interval_constant == 0:
            return avg
        cb = self.interval_constant * (np.sqrt(2 * i2n.sum() / i2n) - np.sqrt(2))
        return avg + cb

    def sample_random(self, k: int):
        if k == self.n_arms:
            return list(range(self.n_arms))
        weights = self.weights()
        weights = weights / weights.sum()
        weights[weights <= 0] = 1e-7
        return weighted_sample_without_replacement(range(self.n_arms), weights, k)

    def set(self, i: int, sum: float, n: int) -> None:
        assert i < self.n_arms
        self.i2sum[i] = sum
        self.i2n[i] = n
        # for logging
        self.i2n_raw[i] = n

    def update(self, i: int, reward: float) -> None:
        assert i < self.n_arms
        self.i2sum[:self.n_arms] *= self.discount
        self.i2n  [:self.n_arms] *= self.discount
        self.i2sum[i] += reward * (1 - self.discount)
        self.i2n  [i] +=          (1 - self.discount)
        # for logging
        self.i2n_raw[i] += 1

    def update_many(self, idxs: np.ndarray, rewards: np.ndarray) -> None:
        assert (idxs < self.n_arms).all()
        self.i2sum[:self.n_arms] *= self.discount
        self.i2n  [:self.n_arms] *= self.discount
        self.i2sum[idxs] += rewards * (1 - self.discount)
        self.i2n  [idxs] +=           (1 - self.discount)
        # for logging
        self.i2n_raw[idxs] += 1


class SimpleSAR:
    """
    Also, replay samples are selected by discounted UCB (D-UCB; Kocsis and Szepesva ́ri 2006)
    """
    def __init__(self, maxlen: int, batch_size: int,
                 seed: int = 0,
        ) -> None:
        self.batch_size = batch_size
        self.last_pushed = None
        self.rng = np.random.default_rng(seed=seed)
        self.maxlen = maxlen
        self.items = deque(maxlen=maxlen)

    def push(self, sarsd):
        sp, a, r, sp_, d = sarsd
        item = [sp, a, r, sp_, d]
        self.items.append(item)
        return

    def sample(self):
        batch_size = min(self.size(), self.batch_size)
        return random.sample(self.items, batch_size), lambda *_: None

    def size(self):
        return len(self.items)

class PrioritizedSAR:
    """Prioritized Experience Replay (PER) Queue

    Also, replay samples are selected by discounted UCB (D-UCB; Kocsis and Szepesva ́ri 2006)
    """
    def __init__(self, maxlen: int, batch_size: int,
                 per_power: float = 0.5,
                 per_discount: float = 0.8,
                 per_propagate_backwards: float = 0.7,
                 per_propagate_limit: int = 30,
                 ducb_interval_constant: float = 0,
                 ducb_discount: float = 0.1,  # discount per queue replacement
                 seed: int = 0,
        ) -> None:
        self.batch_size = batch_size

        # D-UCB
        self.ucb = DiscountedUCB(maxlen,
                                 interval_constant=ducb_interval_constant,
                                 # discount per update
                                 discount=ducb_discount ** (batch_size / maxlen)
        )
        self.i_pq2i_ucb = {}
        self.i_ucb2i_pq = []

        # Priority queue
        # sum, n_visit, piece, state_action, reward, state, done, parent
        def on_pushpop(i_push: int, i_pop: int | None) -> None:
            if i_pop is None:
                i_ucb = self.ucb.add_arm()
                assert i_ucb == len(self.i_ucb2i_pq), f"{i_ucb} != {len(self.i_ucb2i_pq)}"
                self.i_ucb2i_pq.append(i_push)
                self.i_pq2i_ucb[i_push] = i_ucb
                return
            assert self.ucb.n_arms == maxlen
            # replace arm info
            i_ucb = self.i_pq2i_ucb[i_pop]
            self.i_pq2i_ucb[i_push] = i_ucb
            del self.i_pq2i_ucb[i_pop]
            self.i_ucb2i_pq[i_ucb] = i_pop
            # replace arm info done

            # reset arm
            self.ucb.set(i_ucb, 0, 0)

        self.ipq = IndexedPriorityQueue(maxlen, on_pushpop=on_pushpop)
        self.last_pushed = None

        # Prioritized Experience Replay
        # When PER power is 0, same as uniform sampling
        # When PER power is 1, same as loss-proprtional sampling
        assert 0 <= per_power < 1.
        self.per_discount            = per_discount
        self.per_power               = per_power
        self.per_propagate_backwards = per_propagate_backwards
        self.per_propagate_limit     = per_propagate_limit

        # some additional states
        self.rng = np.random.default_rng(seed=seed)

    def push(self, sarsd):
        sp, a, r, sp_, d = sarsd
        item = [sp, a, r, sp_, d, self.last_pushed]
        x = self.ipq.push(1e5, item)

        self.last_pushed = None if d else x
        return

    def size(self):
        return self.ipq.size()

    def sample(self):
        batch_size = min(self.size(), self.batch_size)

        # d-ucb sampling
        # idxs_ucb = self.ucb.sample_random(batch_size)
        # idxs_pq = np.asarray([self.i_ucb2i_pq[i_ucb] for i_ucb in idxs_ucb])

        # def update(priorities: np.ndarray) -> None:
        #     """Update into discounted average"""
        #     # Update loss info
        #     self.ucb.update_many(np.asarray(idxs_ucb), priorities)

        #     # propagate backwards
        #     # Propagate 10% most surprising samples backwards in time
        #     # And overwrite reward score
        #     idxs_topk = np.argpartition(priorities, -len(idxs_pq) // 10)[-len(idxs_pq) // 10:]
        #     for i_pq in idxs_pq[idxs_topk]:
        #         rsum = self.ucb.i2sum[self.i_pq2i_ucb[i_pq]]
        #         item = self.ipq.i_push2item[i_pq]
        #         for _ in range(self.per_propagate_limit):
        #             # get parent
        #             item = self.ipq.heap[i_pq][-1][-1]
        #             if item is None:
        #                 break
        #             i_pq = item[1]
        #             # get parent done

        #             rsum *= self.per_propagate_backwards
        #             rsum_ = self.ucb.i2sum[self.i_pq2i_ucb[i_pq]]
        #             self.ucb.i2sum[self.i_pq2i_ucb[i_pq]] = max(rsum_, rsum)
        #     # propagate backwards done

        #     # heapify again
        #     weights = self.ucb.weights()
        #     for i_ucb, w in enumerate(weights):
        #         self.ipq.i_push2item[self.i_ucb2i_pq[i_ucb]][0] = w
        #     self.ipq.update()

        #     return
        # d-ucb sampling done

        # uniform sampling
        # idxs_pq = random.sample(range(len(self.ipq.heap)), batch_size)
        # def update(priorities):
        #     for i_pq, v in zip(idxs_pq, priorities):
        #         self.ipq.heap[i_pq][0] = v
        #     self.ipq.update()
        # uniform sampling done

        # simple weighted sampling
        weights = np.asarray([x for x, *_ in self.ipq.heap])
        weights /= weights.sum()
        weights[weights <= 0] = 1e-7
        idxs_pq = weighted_sample_without_replacement(range(self.ipq.size()), weights, batch_size)
        def update(priorities):
            # heapify again
            for i_pq, v in zip(idxs_pq, priorities):
                self.ipq.heap[i_pq][0] = v ** self.per_power
            self.ipq.update()
        # simple weighted sampling done

        # backward propatating weighted sampling
        # weights = np.asarray([x ** .5 for x, *_ in self.ipq.heap])
        # weights /= weights.sum()
        # weights[weights <= 0] = 1e-7
        # idxs_pq = np.asarray(weighted_sample_without_replacement(range(self.ipq.size()), weights, batch_size))
        # def update(priorities):
        #     # Overwrite reward score first
        #     for i_pq, v in zip(idxs_pq, priorities):
        #         self.ipq.heap[i_pq][0] = v ** self.per_power

        #     # propagate backwards
        #     # Propagate 10% most surprising samples backwards in time
        #     # And overwrite reward score
        #     idxs_topk = np.argpartition(priorities, -len(idxs_pq) // 10)[-len(idxs_pq) // 10:]
        #     for i_pq, v in zip(idxs_pq[idxs_topk], priorities[idxs_topk]):
        #         item = self.ipq.heap[i_pq]
        #         v = v ** self.per_power
        #         for _ in range(self.per_propagate_limit):
        #             item[0] = max(item[0], v)

        #             # get parent
        #             item = item[-1][-1]
        #             if item is None:
        #                 break

        #             v *= self.per_propagate_backwards
        #             # get parent done
        #     # propagate backwards done

        #     # heapify again
        #     self.ipq.update()
        # backward propatating weighted sampling done

        # backward propatating weighted sampling
        # weights = np.asarray([x for x, *_ in self.ipq.heap])
        # weights /= weights.sum()
        # weights[weights <= 0] = 1e-7
        # idxs_pq = np.asarray(weighted_sample_without_replacement(range(self.ipq.size()), weights, batch_size))
        # def update(priorities):
        #     # Overwrite reward score first
        #     vs = priorities ** self.per_power
        #     for i_pq, v in zip(idxs_pq, vs):
        #         self.ipq.heap[i_pq][0] = v

        #     # propagate backwards
        #     # Propagate 10% most surprising samples backwards in time
        #     # And overwrite reward score
        #     idxs_topk = np.argpartition(vs, -len(idxs_pq) // 10)[-len(idxs_pq) // 10:]
        #     for i_pq, v in zip(idxs_pq[idxs_topk], vs[idxs_topk]):
        #         item = self.ipq.heap[i_pq]
        #         for _ in range(self.per_propagate_limit):
        #             item[0] = max(item[0], v)

        #             # get parent
        #             item = item[-1][-1]
        #             if item is None:
        #                 break

        #             v *= self.per_propagate_backwards
        #             # get parent done
        #     # propagate backwards done

        #     # heapify again
        #     self.ipq.update()
        # backward propatating weighted sampling done

        return [self.ipq.heap[i][2][:-1] for i in idxs_pq], update


class Accumulator():
    def __init__(self):
        self.sum = 0
        self.n = 0
        self._max = -1e7
        self._min = 1e7

    def clear(self):
        self.sum = 0
        self.n = 0
        self._max = -1e7
        self._min = 1e7

    def add(self, v):
        self.sum += v
        self.n += 1
        self._min = min(self._min, v)
        self._max = max(self._max, v)

    @property
    def average(self):
        return 0 if self.n == 0 else self.sum / self.n

    @property
    def min(self):
        return 0 if self.n == 0 else self._min

    @property
    def max(self):
        return 0 if self.n == 0 else self._max
