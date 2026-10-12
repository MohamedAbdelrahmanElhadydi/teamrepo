#!/usr/bin/env python3
"""
Sequence Pattern Predictor
=====================================================================
A model that discovers the pattern in a number sequence and continues it.

Idea (Hypothesis Search / Occam's Razor):
  1) We keep a list of "hypotheses" (models), ordered from simplest to most complex.
  2) Each hypothesis tries to explain *every* number in the input exactly.
  3) The first hypothesis that succeeds is the pattern. We use it to
     generate the next N numbers.
  4) If none succeeds -> Fallback: a statistical estimate (Linear Regression).

Usage:
  python ai.py                       -> interactive mode
  python ai.py 10,12,11,13,12,14,13  -> direct mode
  python ai.py --test                -> run self-tests
Dependencies: standard Python only (no pip install needed).
"""
import sys
from fractions import Fraction

STEPS = 5  # how many numbers to predict


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def fmt(x):
    x = Fraction(x)
    return str(x.numerator) if x.denominator == 1 else f"{float(x):.6g}"


def fmt_signed(x):
    return ("+" if Fraction(x) >= 0 else "") + fmt(x)


def diff(seq):
    return [b - a for a, b in zip(seq, seq[1:])]


def find_period(xs, max_p):
    """Smallest period p such that the values repeat (at least one full repeat)."""
    for p in range(1, max_p + 1):
        if len(xs) >= 2 * p and all(xs[i] == xs[i + p] for i in range(len(xs) - p)):
            return p
    return None


def extend_from_top(levels, new_top):
    """
    levels  = [sequence, its differences, differences of differences, ...]
    new_top = new values for the top level; we sum them back down
              to the original sequence level.
    """
    cur_new = new_top
    for j in range(len(levels) - 2, -1, -1):
        last = levels[j][-1]
        new = []
        for d in cur_new:
            last = last + d
            new.append(last)
        cur_new = new
    return cur_new


class Result:
    def __init__(self, name, explanation, predict):
        self.name, self.explanation, self.predict = name, explanation, predict


# ----------------------------------------------------------------------
# Hypotheses (models)
# ----------------------------------------------------------------------
def model_polynomial(seq, max_deg=3):
    """Constant / arithmetic / quadratic / cubic (d-th differences are constant)."""
    names = {0: "constant", 1: "arithmetic (constant difference)",
             2: "quadratic (second differences are constant)", 3: "cubic"}
    levels = [seq]
    for d in range(0, max_deg + 1):
        if d > 0:
            levels.append(diff(levels[-1]))
        top = levels[-1]
        min_top = 2 if d <= 1 else 3
        if len(top) >= min_top and len(set(top)) == 1:
            c = top[0]
            lv = [l[:] for l in levels]
            return Result(
                "Polynomial: " + names[d],
                f"Differences of order {d} are constant = {fmt(c)}" if d
                else f"All numbers are equal to {fmt(c)}",
                lambda steps, lv=lv, c=c: extend_from_top(lv, [c] * steps),
            )
    return None


def model_geometric(seq, max_period=3):
    """Geometric: the ratio to the previous number is constant or repeats in a cycle."""
    if any(x == 0 for x in seq):
        return None
    ratios = [b / a for a, b in zip(seq, seq[1:])]
    p = find_period(ratios, max_period)
    if p is None:
        return None
    cycle = ratios[:p]
    n = len(ratios)

    def predict(steps):
        out, last = [], seq[-1]
        for i in range(steps):
            last = last * cycle[(n + i) % p]
            out.append(last)
        return out

    if p == 1:
        return Result("Geometric", f"Each number = previous number x {fmt(cycle[0])}", predict)
    return Result("Periodic ratios",
                  "Ratios repeat: " + ", ".join("x" + fmt(r) for r in cycle), predict)


def model_periodic_diff(seq, max_order=2, max_period=4):
    """
    The pattern from your example: the differences repeat in a cycle.
    10,12,11,13,... -> (+2, -1) and so on.
    order=2 means the differences of the differences are what repeats.
    """
    levels = [seq]
    for k in range(1, max_order + 1):
        levels.append(diff(levels[-1]))
        top = levels[-1]
        p = find_period(top, max_period)
        if p is not None and p >= 2:
            cycle = top[:p]
            n = len(top)
            lv = [l[:] for l in levels]
            what = "Differences" if k == 1 else "Second differences"
            return Result(
                f"Periodic differences (order {k})",
                f"{what} repeat every {p} steps: " + ", ".join(fmt_signed(c) for c in cycle),
                lambda steps, lv=lv, cycle=cycle, p=p, n=n: extend_from_top(
                    lv, [cycle[(n + i) % p] for i in range(steps)]),
            )
    return None


def _solve(A, b):
    """Gaussian elimination with fractions (exact, no float rounding errors)."""
    n = len(A)
    M = [row[:] + [bb] for row, bb in zip(A, b)]
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] != 0), None)
        if piv is None:
            return None
        M[c], M[piv] = M[piv], M[c]
        pv = M[c][c]
        M[c] = [x / pv for x in M[c]]
        for r in range(n):
            if r != c and M[r][c] != 0:
                f = M[r][c]
                M[r] = [x - f * y for x, y in zip(M[r], M[c])]
    return [M[i][n] for i in range(n)]


def model_recurrence(seq, orders=(1, 2, 3)):
    """a[n] = c1*a[n-1] + ... + ck*a[n-k] + c0   (e.g. Fibonacci)."""
    n = len(seq)
    for k in orders:
        if n - k < k + 2:      # need extra equations to verify, not just solve
            continue
        rows = [[seq[i - j] for j in range(1, k + 1)] + [Fraction(1)] for i in range(k, n)]
        ys = [seq[i] for i in range(k, n)]
        coef = _solve(rows[:k + 1], ys[:k + 1])
        if coef is None:
            continue
        if all(sum(c * x for c, x in zip(coef, r)) == y for r, y in zip(rows, ys)):
            def predict(steps, coef=coef, k=k, seq=seq):
                s, out = list(seq), []
                for _ in range(steps):
                    row = [s[-j] for j in range(1, k + 1)] + [Fraction(1)]
                    v = sum(c * x for c, x in zip(coef, row))
                    s.append(v)
                    out.append(v)
                return out

            terms = [f"{fmt(c)}*a[n-{j}]" for j, c in enumerate(coef[:-1], 1) if c != 0]
            if coef[-1] != 0:
                terms.append(fmt(coef[-1]))
            return Result(f"Linear recurrence (order {k})",
                          "a[n] = " + " + ".join(terms), predict)
    return None


def model_interleaved(seq):
    """Two (or three) interleaved sequences: 1,10,2,20,3,30 -> (1,2,3) and (10,20,30)."""
    n = len(seq)
    for s in (2, 3):
        subs = [seq[i::s] for i in range(s)]
        if any(len(sb) < 4 for sb in subs):
            continue
        fits = []
        for sb in subs:
            r = (model_polynomial(sb, 2) or model_geometric(sb, 1)
                 or model_recurrence(sb, (1,)))
            if r is None:
                break
            fits.append(r)
        else:
            def predict(steps, fits=fits, s=s, n=n):
                need = steps // s + 2
                preds = [f.predict(need) for f in fits]
                cnt, out = [0] * s, []
                for j in range(steps):
                    i = (n + j) % s
                    out.append(preds[i][cnt[i]])
                    cnt[i] += 1
                return out

            return Result(
                f"Interleaved ({s} sequences)",
                f"The sequence is made of {s} interleaved sequences: "
                + " | ".join(f"({f.explanation})" for f in fits),
                predict)
    return None


def model_fallback(seq):
    """No exact pattern found -> Linear Regression on the index (rough estimate)."""
    n = len(seq)
    xs = list(range(n))
    ys = [float(v) for v in seq]
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den
    icpt = my - slope * mx
    return Result("Fallback: Linear Regression (rough estimate)",
                  f"y ~ {slope:.4g}*x + {icpt:.4g}  (no exact pattern found)",
                  lambda steps: [Fraction(slope * (n + i) + icpt).limit_denominator(10 ** 6)
                                 for i in range(steps)])


# Ordered from simplest to most complex (Occam's Razor)
MODELS = [model_polynomial, model_geometric, model_periodic_diff,
          model_interleaved, model_recurrence]


def predict_sequence(numbers, steps=STEPS):
    seq = [Fraction(x) for x in numbers]
    if len(seq) < 4:
        raise ValueError("I need at least 4 numbers to detect a pattern.")
    for model in MODELS:
        res = model(seq)
        if res:
            return res, res.predict(steps), True
    res = model_fallback(seq)
    return res, res.predict(steps), False


# ----------------------------------------------------------------------
# User interface
# ----------------------------------------------------------------------
def parse(text):
    text = text.replace("\u060c", ",").replace(",", " ")  # also accepts the Arabic comma
    return [Fraction(t) for t in text.split()]


def run(text):
    nums = parse(text)
    res, preds, exact = predict_sequence(nums)
    print(f"\nInput      : {', '.join(fmt(x) for x in nums)}")
    print(f"Pattern    : {res.name}")
    print(f"Explanation: {res.explanation}")
    print(f"Next {STEPS}     : {', '.join(fmt(x) for x in preds)}")
    print(f"Confidence : {'High (the pattern explains every number exactly)' if exact else 'Low (estimate only)'}\n")


def self_test():
    cases = [
        ("10,12,11,13,12,14,13", [15, 14, 16, 15, 17]),
        ("2,4,6,8,10", [12, 14, 16, 18, 20]),
        ("1,4,9,16,25", [36, 49, 64, 81, 100]),
        ("3,6,12,24,48", [96, 192, 384, 768, 1536]),
        ("1,1,2,3,5,8,13,21", [34, 55, 89, 144, 233]),
        ("1,10,2,20,3,30,4,40", [5, 50, 6, 60, 7]),
        ("5,3,4,2,3,1,2", [0, 1, -1, 0, -2]),
        ("1,2,4,7,11,16", [22, 29, 37, 46, 56]),
    ]
    ok = True
    for text, expected in cases:
        _, preds, _ = predict_sequence(parse(text))
        got = [int(p) if p.denominator == 1 else float(p) for p in preds]
        status = "PASS" if got == expected else "FAIL"
        ok &= got == expected
        print(f"[{status}] {text:28} -> {got}")
    print("\nAll tests passed." if ok else "\nSome tests failed.")


def main():
    args = sys.argv[1:]
    if args and args[0] == "--test":
        return self_test()
    if args:
        return run(" ".join(args))
    print("Enter a number sequence (separated by commas or spaces), or q to quit")
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line.lower() in ("q", "quit", "exit"):
            break
        if not line:
            continue
        try:
            run(line)
        except ValueError as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
