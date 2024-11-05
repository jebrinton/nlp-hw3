# SYSTEM IMPORTS
from collections.abc import Iterable, Mapping, Sequence
from collections import defaultdict
from typing import Tuple, Type


# PYTHON PROJECT IMPORTS


class Production(object):
    def __init__(self: Type["Production"],
                 nonterm: str,
                 productions: Tuple[str, ...]
                 ) -> None:
        self.nonterm: str = nonterm
        self.productions: Tuple[str, ...] = tuple(productions)

        if (len(self.productions) < 1 or len(self.productions) > 2):
            raise Exception(f"ERROR: production {self.nonterm}->{self.productions} is not in CNF!")

        self.is_terminal: bool = len(self.productions) == 1

    def __eq__(self: Type["Production"],
               other: object
               ) -> bool:
        return isinstance(other, Production) and\
            (self.nonterm, self.productions, self.is_terminal) == (other.nonterm, other.productions, other.is_terminal)

    def __ne__(self: Type["Production"],
               other: object
               ) -> bool:
        return not self == other

    def __hash__(self: Type["Production"]) -> int:
        return hash((self.nonterm, self.productions, self.is_terminal))


# a grammer will be a graph object. Each
class PCFG(object):
    def __init__(self: Type["PCFG"]):
        self.productions_from: Mapping[str, Mapping[Production, float]] = defaultdict(lambda: defaultdict(float))
        self.productions_to: Mapping[Tuple[str, ...], Set[str]] = defaultdict(set)

        self.nonterminals: Set[str] = set()
        self.terminals: Set[str] = set()
        self.start: str = None

    def _validate_self(self: Type["PCFG"]) -> None:
        for nonterm in self.nonterminals:
            for t in self.productions_from[nonterm].keys():
                assert(t.productions in self.productions_to and
                       t.nonterm in self.productions_to[t.productions])

    def set_start(self: Type["PCFG"],
                  nonterm: str
                  ) -> None:
        self.nonterminals.add(nonterm)
        self.start = nonterm

    def get_rule_prob(self: Type["PCFG"],
                      nonterm: str,
                      productions: Tuple[str, ...]
                      ) -> float:
        t = Production(nonterm, productions)

        if nonterm not in self.nonterminals: return 0.0
        if t not in self.productions_from[nonterm]: return 0.0

        return self.productions_from[nonterm][t]

    def add_rule(self: Type["PCFG"],
                 nonterm: str,
                 productions: Tuple[str, ...],
                 wt: int = 1
                 ) -> None:
        self.nonterminals.add(nonterm)

        t: Production = Production(nonterm, productions)

        if len(productions) == 2:
            self.nonterminals.update(productions)
        else:
            self.terminals.update(productions)

        self.productions_from[t.nonterm][t] += wt
        self.productions_to[t.productions].add(nonterm)

    def get_rules_from(self: Type["PCFG"],
                       nonterm: str
                       ) -> Iterable[Tuple[Production, float]]:
        if nonterm not in self.nonterminals:
            raise Exception("ERROR: [%s] not in PCFG nonterminal set" % nonterm)
        return self.productions_from[nonterm].items()

    def get_rules_to(self: Type["PCFG"],
                     *symbols: Tuple[str] # *symbols will guarantee aguments are in tuple form
                     ) -> Sequence[Tuple[str, float]]:
        for symb in symbols:
            if symb not in self.terminals and symb not in self.nonterminals:
                raise Exception("ERROR: [%s] not in PCFG" % symb)

        rules: Sequence[Tuple[str, float]] = list()
        for nonterm in self.productions_to[symbols]:
            for t, wt in self.productions_from[nonterm].items():
                if symbols == t.productions:
                    rules.append((t.nonterm, wt))
        return rules

    def reset(self: Type["PCFG"]) -> Type["PCFG"]:
        # keep the grammar structure the same, but reset the weights
        for nonterm in self.nonterminals:
            for t, wt in self.productions_from[nonterm].items():
                self.productions_from[nonterm][t] = 0.0
        return self

    def normalize_joint(self: Type["PCFG"],
                        add: float = 0.0
                        ) -> None:
        for nonterm in self.nonterminals:
            total = 0.0
            for t, wt in self.productions_from[nonterm].items():
                total += wt + add
            for t, wt in self.productions_from[nonterm].items():
                if wt + add == 0.0: continue
                self.productions_from[nonterm][t] = (wt + add) / total

    def __str__(self: Type["PCFG"]) -> str:
        s: str = ""
        for nonterm in self.nonterminals:
            for t, wt in self.productions_from[nonterm].items():
                s += "{0} -> {1} # {2:.3f}\n".format(t.nonterm, " ".join(t.productions), wt)
        return s[:-1]

    def __repr__(self: Type["PCFG"]) -> str:
        return "%s" % self

