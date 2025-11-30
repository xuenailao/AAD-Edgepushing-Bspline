# aad/core/engine.py
from __future__ import annotations
import numpy as np
from typing import Sequence, Union
from .tape import global_tape
from .var import ADVar

def zero_adjoints():
    """
    Set all adjoints (bar variables) on the current tape to zero.
    We scan the recorded nodes to find all reachable ADVars (outputs and parents)
    and zero their `.adj` fields.
    """
    seen = set()
    for node in global_tape.nodes:
        if id(node.out) not in seen:
            _zero(node.out); seen.add(id(node.out))
        for p, _ in node.parents:
            if id(p) not in seen:
                _zero(p); seen.add(id(p))

def _zero(v: ADVar):
    if isinstance(v.adj, np.ndarray):
        v.adj.fill(0.0)
    else:
        v.adj = 0.0

def reverse(outputs: Union[ADVar, Sequence[ADVar]], seed=1.0):
    """
    Run a single reverse pass from the given output(s).

    Args:
        outputs: an ADVar or a (list/tuple) of ADVars to seed.
        seed: scalar or same-shaped array used as the adjoint seed. If `outputs`
              is a sequence, each output is seeded with 1.0 (the `seed` arg is
              ignored in that case).

    Notes:
        - For each node, we propagate: p.adj += y.adj * (∂y/∂p).
        - NumPy broadcasting is allowed when shapes are compatible.
    """
    # Seed adjoints
    if isinstance(outputs, (list, tuple)):
        for y in outputs:
            _seed(y, 1.0)
    else:
        _seed(outputs, seed)

    # Backward sweep
    for node in reversed(global_tape.nodes):
        y = node.out
        if _is_zero(y.adj):
            continue  # nothing to propagate
        for (p, local_partial) in node.parents:
            if not p.requires_grad:
                continue
            # Accumulate: p.adj += y.adj * (∂y/∂p)
            p.adj = p.adj + y.adj * local_partial  # NumPy will broadcast if arrays

def _seed(v: ADVar, seed):
    if isinstance(v.adj, np.ndarray):
        v.adj += np.ones_like(v.val, dtype=float) * seed
    else:
        v.adj += float(seed)

def _is_zero(x):
    try:
        return (x == 0).all()
    except Exception:
        return x == 0

def zero_tangents():
    """
    Zero all first-order tangents and their reverse-mode companions
    that have appeared on the current tape.

    Concretely:
        - v.dot     := 0
        - v.adj_dot := 0

    This is useful before running a new JVP/FoR computation so that no
    stale tangent information leaks into the next pass.
    """
    seen = set()
    for node in global_tape.nodes:
        if id(node.out) not in seen:
            _zero_dot(node.out); seen.add(id(node.out))
        for p, _ in node.parents:
            if id(p) not in seen:
                _zero_dot(p); seen.add(id(p))

def _zero_dot(v: ADVar):
    if isinstance(v.dot, np.ndarray):
        v.dot.fill(0.0); v.adj_dot.fill(0.0)
    else:
        v.dot = 0.0; v.adj_dot = 0.0

# ---------------- FoR: Hessian-vector product H·v ---------------- #
def hvp_for(f, inputs: dict, v: dict):
    """
    Forward-over-Reverse (FoR) core routine: compute H·v for a scalar output y=f(x).

    Args:
        f      : callable taking a dict[name -> ADVar] and returning an ADVar (scalar output).
        inputs : dict name -> scalar (or ADVar). Mirrors your grads(dict) signature.
                 If a value is not an ADVar, it will be wrapped as ADVar(requires_grad=True).
        v      : dict name -> scalar; direction vector aligned with `inputs` keys.

    Returns:
        np.ndarray of shape [n], in the order of inputs.keys(), containing (H·v).

    Algorithm (Pearlmutter / R-operator, Forward-over-Reverse):
        1) Forward JVP pass: set input tangents x.dot = v and evaluate f, letting
           each primitive compute its out.dot recursively.
        2) Enhanced reverse pass: propagate both (bar, bar_dot) using:
               p.adj     += y.adj * (∂y/∂p)
               p.adj_dot += y.adj_dot * (∂y/∂p) + y.adj * R{∂y/∂p}
           where R{·} is the directional derivative along `.dot`.
        3) For leaf inputs, adj_dot equals the desired H·v component.
    """
    # Suggestion: use this under `with use_tape():` to isolate graphs externally.

    # 1) Build graph & forward: create/wrap inputs as ADVars and seed tangents with v
    vars_ad = {}
    for k, x0 in inputs.items():
        x = x0 if isinstance(x0, ADVar) else ADVar(x0, requires_grad=True, name=k)
        vars_ad[k] = x
    # Seed input tangents
    for k, x in vars_ad.items():
        val = x.val
        step = float(v[k])
        if isinstance(val, np.ndarray):
            x.dot = np.ones_like(val, dtype=float) * step
        else:
            x.dot = float(step)

    y = f(vars_ad)
    if not isinstance(y, ADVar):
        y = ADVar(y, requires_grad=False, name="y")

    # 2) Enhanced reverse: clear bars and bar_dots; seed y.adj = 1, y.adj_dot = 0
    zero_adjoints()
    # Keep input tangents `.dot` intact; only clear `.adj_dot`.
    _zero_adj_dot_only()

    _seed(y, 1.0)  # y.adj += 1.0 ; y.adj_dot remains 0 by default

    # Reverse sweep: propagate (bar, bar_dot)
    for node in reversed(global_tape.nodes):
        yv = node.out
        if _is_zero(yv.adj) and _is_zero(yv.adj_dot):
            continue
        # parents: List[(p, local_partial)]
        for idx, (p, a) in enumerate(node.parents):
            if not p.requires_grad:
                continue
            # bar propagation
            p.adj = p.adj + yv.adj * a
            # bar_dot propagation: dot{bar_p} += dot{bar_y} * a + bar_y * R{a}
            Ra = _R_local_partial(node, idx)  # uses op_tag and parent index
            p.adj_dot = p.adj_dot + (yv.adj_dot * a + yv.adj * Ra)

    # 3) Collect H·v in the input order as adj_dot
    return np.array([float(vars_ad[k].adj_dot) for k in inputs.keys()], dtype=float)

def _zero_adj_dot_only():
    """
    Zero only the reverse-mode companions for tangents, i.e., v.adj_dot.
    Leaves v.dot untouched, so the forward JVP seeds remain valid.
    """
    seen = set()
    for node in global_tape.nodes:
        if id(node.out) not in seen:
            _zero_one_adj_dot(node.out); seen.add(id(node.out))
        for p, _ in node.parents:
            if id(p) not in seen:
                _zero_one_adj_dot(p); seen.add(id(p))

def _zero_one_adj_dot(v: ADVar):
    try:
        v.adj_dot.fill(0.0)   # ndarray case
    except Exception:
        v.adj_dot = 0.0       # scalar case

# -------- R{local_partial} rules for supported primitives -------- #
def _R_local_partial(node, parent_idx):
    """
    Return the forward-directional derivative of a local partial:
        R{a} where a = ∂(node.out) / ∂(parent_idx-th parent)

    Notation
    --------
    - node.out = y (ADVar)
    - node.parents[k] = (p_k, a_k) where a_k = ∂y/∂p_k (numeric, shaped)
    - For unary ops, parent_idx == 0.
    - R{•} is taken along the existing JVP 'dot' channel stored on ADVars.

    Supported ops
    -------------
    add, sub, neg, mul, div, exp, log, sqrt, norm_cdf, pow
    """
    tag = node.op_tag
    y = node.out  # ADVar produced by this op

    # Convenience accessors for the chosen parent and its local partial
    p = node.parents[parent_idx][0]      # ADVar parent
    a = node.parents[parent_idx][1]      # numeric local partial a = ∂y/∂p

    # ---------- Linear ops: R{constant} = 0 ----------
    if tag in ("add", "sub", "neg"):
        # y = x + z, y = x - z, y = -x  -> local partials are ±1 constants
        # Their directional derivatives are zero.
        return 0.0

    # ---------- Product ----------
    if tag == "mul":
        # y = x * z
        # ∂y/∂x = z, ∂y/∂z = x
        # R{∂y/∂x} = R{z} = z.dot
        # R{∂y/∂z} = R{x} = x.dot
        other_idx = 1 - parent_idx
        other_parent = node.parents[other_idx][0]
        return other_parent.dot

    # ---------- Division ----------
    if tag == "div":
        # y = x / z
        # ∂y/∂x = 1/z              => R{1/z} = -(1/z^2) * z.dot
        # ∂y/∂z = -x / z^2         => R{-x/z^2} = -(x.dot)/z^2 + 2x*z.dot / z^3
        x = node.parents[0][0]
        z = node.parents[1][0]
        if parent_idx == 0:
            return -(1.0 / (z.val * z.val)) * z.dot
        else:
            return -(x.dot) / (z.val * z.val) + (2.0 * x.val * z.dot) / (z.val ** 3)

    # ---------- Exponential ----------
    if tag == "exp":
        # y = exp(x)
        # ∂y/∂x = y.val  => R{a} = y.val * x.dot
        return y.val * p.dot

    # ---------- Logarithm ----------
    if tag == "log":
        # y = log(x)
        # ∂y/∂x = 1/x    => R{a} = -(1/x^2) * x.dot
        return -(1.0 / (p.val * p.val)) * p.dot

    # ---------- Square root ----------
    if tag == "sqrt":
        # y = sqrt(x)
        # ∂y/∂x = 0.5 / sqrt(x) = 0.5 / y.val
        # R{a} = d/dx(0.5*x^{-1/2}) * x.dot = -0.25 * x^{-3/2} * x.dot = -(0.25) * p.dot / y^3
        return -(0.25) * p.dot / (y.val ** 3)

    # ---------- Normal CDF ----------
    if tag == "norm_cdf":
        # y = Φ(x),  ∂y/∂x = φ(x) where φ(x) = exp(-x^2/2)/sqrt(2π)
        # R{φ(x)} = φ'(x) * x.dot = (-x * φ(x)) * x.dot
        phi = np.exp(-0.5 * p.val * p.val) / np.sqrt(2.0 * np.pi)
        return (-p.val * phi) * p.dot

    # ---------- Power ----------
    if tag == "pow":
        # y = x^p  (assume x>0 for non-integer p)
        base = node.parents[0][0]  # x
        expo = node.parents[1][0]  # p
        xv, pv = base.val, expo.val
        xd, pd = base.dot, expo.dot
        yv     = y.val

        # JVP ẏ (for safety re-compute if missing):
        # ẏ = y * ( ṗ * log x + p * ẋ / x )
        if np.all(xv > 0):
            ydot = getattr(y, "dot", None)
            if ydot is None:
                ydot = yv * (pd * np.log(xv) + pv * (xd / xv))
        else:
            # Outside valid domain for non-integer powers; return conservative zero.
            return 0.0

        if parent_idx == 0:
            # a_x = ∂y/∂x = p * x^(p-1) = y * p / x
            # R{a_x} = ẏ * (p/x) + y * ( ṗ / x - p * ẋ / x^2 )
            return (ydot * (pv / xv)) + (yv * (pd / xv - pv * xd / (xv * xv)))
        else:
            # a_p = ∂y/∂p = y * log x
            # R{a_p} = ẏ * log x + y * ( ẋ / x )
            return (ydot * np.log(xv)) + (yv * (xd / xv))

    # ---------- Fallback (unsupported op): safe zero ----------
    return 0.0

# ---------------- Edge-Pushing: Componentwise (Algorithm 4) ---------------- #
from collections import defaultdict
import numpy as np
from typing import Dict, Sequence, Union

from .tape import global_tape, use_tape
from .var import ADVar

# ---------------- Edge‑Pushing: Componentwise (Algorithm 4) ---------------- #
# This version follows the paper’s control flow literally: for i = L…1 do
#   (1) Pushing   (push masses involving y_i onto parents)
#   (2) Creating  (add vbar[i] * local 2nd derivatives into {·} / {·,·} space)
#   (3) Adjoint   (standard first‑order reverse propagation)
# After the sweep, (4) Projection: H = P W P^T — project all masses that land on
# input–input pairs into the final Hessian buckets. No on‑the‑fly projection.


def edge_push_hessian(
    f,
    inputs: Dict[str, Union[float, int, np.ndarray, ADVar]],
    *,
    sparse: bool = False,
):
    """
    Compute the full Hessian ∇²f(x) via the componentwise edge‑pushing algorithm
    (Algorithm 4). This implementation keeps *all* intermediate pair masses in a
    variable‑pair accumulator `W_var_pairs` keyed by
        frozenset({id_a})  for singletons {a}
        frozenset({id_a, id_b}) for unordered pairs {a,b}
    and only performs the Projection step at the very end (H = P W Pᵀ).

    Parameters
    ----------
    f : callable({name: ADVar}) -> ADVar   # scalar output
    inputs : dict[name -> numeric or ADVar]  # wrapped as ADVar(requires_grad=True)
    sparse : bool, default False
        If True, return a dict with keys as frozenset({i}) or frozenset({i,j}).
        If False, return a dense np.ndarray [n, n] in the order of inputs.keys().
    """
    with use_tape():
        # 1) Build graph
        vars_ad = {
            k: (v if isinstance(v, ADVar) else ADVar(v, requires_grad=True, name=k))
            for k, v in inputs.items()
        }
        y = f(vars_ad)
        if not isinstance(y, ADVar):
            y = ADVar(y, requires_grad=False, name="y")

        L = len(global_tape.nodes)
        if L == 0:
            return {} if sparse else np.zeros((0, 0))

        # 2) Indexing maps
        input_order = list(inputs.keys())
        input_col = {id(vars_ad[k]): i for i, k in enumerate(input_order)}  # id(leaf) -> col
        node_index = {id(nd.out): i for i, nd in enumerate(global_tape.nodes)}  # id(out)->idx

        def is_input_id(aid: int) -> bool:
            # Leaf inputs: appear as a parent but not as any node.out on tape
            return (aid in input_col) and (aid not in node_index)

        # 3) Accumulators
        vbar = defaultdict(float)         # first‑order adjoints on nodes (indexed by tape idx)
        W_var_pairs = defaultdict(float)  # variable‑pair masses (ids of ADVars)

        # Seed output adjoint
        vbar[L - 1] = 1.0

        # ---------- Reverse sweep over the tape ----------
        for i in range(L - 1, -1, -1):
            node = global_tape.nodes[i]
            y_ad = node.out
            y_id = id(y_ad)
            parents = node.parents  # [(ADVar, ∂y/∂p), ...]
            m = len(parents)

            # ===== (1) Pushing =====
            if W_var_pairs:
                # Identify all masses currently involving y: {y} or {y,q}
                touched = [k for k in list(W_var_pairs.keys()) if y_id in k]
                for pk in touched:
                    w = W_var_pairs.pop(pk)
                    ids = list(pk)

                    if len(ids) == 1:
                        # Case {y}: push to all unordered parent pairs {p_r,p_s} (r < s)
                        # and for diagonal we add once (no factor 2 here — matches Algorithm 4)
                        for r in range(m):
                            p_r, a_r = parents[r]
                            for s in range(r + 1, m):
                                p_s, a_s = parents[s]
                                new_key = frozenset({id(p_r), id(p_s)})
                                W_var_pairs[new_key] += w * a_r * a_s
                        # Diagonal (r == s): contributes to singleton {p_r}
                        for r in range(m):
                            p_r, a_r = parents[r]
                            W_var_pairs[frozenset({id(p_r)})] += w * (a_r * a_r)

                    else:
                        # Case {y, q} with q != y: let q_id be the other endpoint
                        ids.remove(y_id)
                        q_id = ids[0]
                        for r in range(m):
                            p_r, a_r = parents[r]
                            if id(p_r) == q_id:
                                # Special j == p case in the paper: factor 2 to singleton {p}
                                W_var_pairs[frozenset({q_id})] += 2.0 * w * a_r
                            else:
                                # General j ≠ p: new unordered pair {j, p}
                                new_key = frozenset({id(p_r), q_id})
                                W_var_pairs[new_key] += w * a_r

            # ===== (2) Creating =====
            vb = vbar[i]
            if vb != 0.0:
                sec = _second_locals(node)
                if sec:
                    # Prepare quick lookups by parent position
                    parent_ids = [id(parents[u][0]) for u in range(m)]

                    def diag_weight_for_pos(u):
                        # Accept: ("diag", u) or legacy ("diag_node", node_idx_of_parent)
                        if ("diag", u) in sec:
                            return sec[("diag", u)]
                        # Legacy support by node index if present
                        p_idx = node_index.get(parent_ids[u])
                        if p_idx is not None and ("diag_node", p_idx) in sec:
                            return sec[("diag_node", p_idx)]
                        return None

                    def cross_weight_for_pos(u, v):
                        key = ("cross", (u, v))
                        if key in sec:
                            return sec[key]
                        # Legacy by node indices (unordered stored with u<v)
                        pu, pv = node_index.get(parent_ids[u]), node_index.get(parent_ids[v])
                        if pu is not None and pv is not None:
                            key_legacy = ("cross", (min(pu, pv), max(pu, pv)))
                            if key_legacy in sec:
                                return sec[key_legacy]
                        return None

                    # Diagonals → singleton {p}
                    for u in range(m):
                        d2 = diag_weight_for_pos(u)
                        if d2 is not None:
                            W_var_pairs[frozenset({parent_ids[u]})] += float(d2 * vb)

                    # Cross terms → unordered pair {p_u, p_v}
                    for u in range(m - 1):
                        for v in range(u + 1, m):
                            d2 = cross_weight_for_pos(u, v)
                            if d2 is not None:
                                key = frozenset({parent_ids[u], parent_ids[v]})
                                W_var_pairs[key] += float(d2 * vb)

            # ===== (3) Adjoint =====
            if vbar[i] != 0.0:
                for (p_ad, a) in parents:
                    p_idx = node_index.get(id(p_ad))
                    if p_idx is not None:
                        vbar[p_idx] += vbar[i] * a

        # ===== (4) Projection: H = P W Pᵀ =====
        n = len(inputs)
        if sparse:
            H_sparse = defaultdict(float)
            for pair_key, w in W_var_pairs.items():
                ids = list(pair_key)
                if len(ids) == 1:
                    aid = ids[0]
                    if is_input_id(aid):
                        H_sparse[frozenset({input_col[aid]})] += float(w)
                else:
                    a, b = ids
                    if is_input_id(a) and is_input_id(b):
                        i, j = input_col[a], input_col[b]
                        H_sparse[frozenset({min(i, j), max(i, j)})] += float(w)
            return H_sparse

        # Dense output
        H = np.zeros((n, n), dtype=float)
        for pair_key, w in W_var_pairs.items():
            ids = list(pair_key)
            if len(ids) == 1:
                aid = ids[0]
                if is_input_id(aid):
                    i = input_col[aid]
                    H[i, i] += float(w)
            else:
                a, b = ids
                if is_input_id(a) and is_input_id(b):
                    i, j = input_col[a], input_col[b]
                    if i == j:
                        H[i, i] += float(w)
                    else:
                        H[i, j] += float(w)
                        H[j, i] += float(w)
        return H

def edge_push_pattern(f, inputs: dict):
    """
    Edge-pushing (sparsity discovery) companion.
    Returns the overestimated sparsity pattern on INPUT columns using the same traversal,
    but only with 'Creating' (nonlinearity) marks, ignoring numeric values.

    Returns
    -------
        set[frozenset({i,j})] where i,j are input indices
    """
    with use_tape():
        vars_ad = {k: (v if isinstance(v, ADVar) else ADVar(v, requires_grad=True, name=k))
                   for k, v in inputs.items()}
        y = f(vars_ad)
        if not isinstance(y, ADVar):
            y = ADVar(y, requires_grad=False, name="y")
        topo = _topo_indexing()
        node_idx_of_advar, input_col = topo["node_idx_of_advar"], topo["input_col"]
        pat = set()
        for i in range(len(global_tape.nodes) - 1, -1, -1):
            node = global_tape.nodes[i]
            parents = node.parents
            # unordered parent pairs
            m = len(parents)
            for u in range(m):
                j_ad, _ = parents[u]
                cj = input_col.get(id(j_ad))
                for v in range(u+1, m):
                    k_ad, _ = parents[v]
                    ck = input_col.get(id(k_ad))
                    if cj is not None and ck is not None and _is_nonlinear(node.op_tag):
                        pat.add(frozenset({cj, ck}))
                # diagonal if unary nonlinear
                if m == 1 and _is_nonlinear(node.op_tag) and cj is not None:
                    pat.add(frozenset({cj, cj}))
        return pat

def _topo_indexing():
    """
    Assign a topological index to each node-out ADVar and build:
      - node_ids: [0..L-1]
      - node_idx_of_advar: { id(ADVar) -> node_index }
      - input_col: { id(ADVar) -> input column index }  (leaf ADVars constructed from user inputs)
    """
    node_idx_of_advar = {}
    for idx, nd in enumerate(global_tape.nodes):
        node_idx_of_advar[id(nd.out)] = idx
    # inputs are ADVars that never appear as .out of any node but are used as parents; we recognize them
    input_advars = []
    seen_out = set(node_idx_of_advar.keys())
    for nd in global_tape.nodes:
        for (p, _) in nd.parents:
            if id(p) not in seen_out and p.requires_grad:
                input_advars.append(p)
                seen_out.add(id(p))  # avoid duplicates
    # final column order = order of user dict keys; build a map {id(ADVar)->col}
    input_col = {}
    col = 0
    for nd in global_tape.nodes:
        for (p, _) in nd.parents:
            if p in input_advars and id(p) not in input_col:
                input_col[id(p)] = col
                col += 1
    return {
        "node_ids": list(range(len(global_tape.nodes))),
        "node_idx_of_advar": node_idx_of_advar,
        "input_col": input_col,
    }
    
def _second_locals(node):
    """
    Return local second derivatives for a primitive op.

    Keys:
        ("diag", u) -> ∂²y / ∂p_u²
        ("cross", (u,v)) -> ∂²y / ∂p_u ∂p_v   (u < v)

    u,v are parent indices (0,1,...).

    Supported ops: add, sub, neg, mul, div, pow, exp, log, sqrt, norm_cdf
    """
    tag = node.op_tag
    out = {}

    if tag in ("add", "sub", "neg"):
        return out

    if tag == "mul":
        # y = x*z ; ∂²y/∂x∂z = 1
        out[("cross", (0, 1))] = 1.0
        return out

    if tag == "div":
        # y = x / z
        x = node.parents[0][0].val
        z = node.parents[1][0].val
        out[("cross", (0, 1))] = -1.0 / (z * z)     # ∂²y/∂x∂z
        out[("diag", 1)] = 2.0 * x / (z ** 3)       # ∂²y/∂z²
        return out

    if tag == "pow":
        # y = x^p (x>0)
        x = node.parents[0][0].val
        p = node.parents[1][0].val
        if x <= 0:
            return out
        yval = x ** p
        out[("diag", 0)] = p * (p - 1.0) * (x ** (p - 2.0))                 # ∂²/∂x²
        out[("diag", 1)] = yval * (np.log(x) ** 2)                           # ∂²/∂p²
        out[("cross", (0, 1))] = (x ** (p - 1.0)) * (1.0 + p*np.log(x))      # ∂²/∂x∂p
        return out

    if tag == "exp":
        # y = exp(g)
        # local second derivative w.r.t. its single parent g is y itself
        y = node.out.val
        out[("diag", 0)] = y
        return out

    if tag == "log":
        x = node.parents[0][0].val
        out[("diag", 0)] = -1.0 / (x * x)
        return out

    if tag == "sqrt":
        x = node.parents[0][0].val
        out[("diag", 0)] = -0.25 * (x ** -1.5)
        return out

    if tag == "norm_cdf":
        x = node.parents[0][0].val
        phi = np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)
        out[("diag", 0)] = -x * phi
        return out

    return out


def _is_nonlinear(op_tag: str) -> bool:
    # used for sparsity pattern only
    return op_tag not in ("add", "sub", "neg")

# ===== Debug helpers: focus-and-log contributions to selected Hessian entries =====
import numpy as _np
from collections import defaultdict as _dd

def diff_hot_pairs(H_for: _np.ndarray, H_ep: _np.ndarray, names, *, top_k=5, tol=1e-8):
    """
    Return the largest-diff Hessian pairs (i<=j) as a list of dicts sorted by |diff|.
    """
    diffs = []
    n = H_for.shape[0]
    for i in range(n):
        for j in range(i, n):
            d = float(H_for[i, j] - H_ep[i, j])
            if abs(d) > tol:
                diffs.append({"i": i, "j": j, "name_i": list(names)[i], "name_j": list(names)[j], "diff": d})
    diffs.sort(key=lambda x: abs(x["diff"]), reverse=True)
    return diffs[:top_k]

def edge_push_hessian_debug(f, inputs: dict, focus_pairs=None):
    """
    Run a 'logging' version of edge-pushing: we compute the same Hessian as edge_push_hessian,
    but in addition we record every Creating-step contribution that lands on an INPUT-INPUT pair.
    focus_pairs: optional set of pairs {(i,j),...} (with i<=j) to log; if None, log all.
    Returns: (H, logs) where logs is a list of entries:
        {
          "pair": (i,j),
          "node_idx": i_node,
          "op_tag": node.op_tag,
          "kind": "diag" or "cross",
          "local_second": float,   # d2 * vbar at that node (before projection)
          "contrib": float,        # the amount added to H[i,j]
          "parents": [name_j, name_k]
        }
    """
    from .tape import use_tape
    from .tape import global_tape as _gt
    from .var import ADVar as _AD

    with use_tape():
        # (build graph)
        vars_ad = {k: (v if isinstance(v, _AD) else _AD(v, requires_grad=True, name=k))
                   for k, v in inputs.items()}
        y = f(vars_ad)
        if not isinstance(y, _AD):
            y = _AD(y, requires_grad=False, name="y")

        topo = _topo_indexing()
        node_ids = topo["node_ids"]
        node_idx_of_advar = topo["node_idx_of_advar"]
        input_col = topo["input_col"]
        L = len(node_ids)
        if L == 0:
            return _np.zeros((0,0)), []

        # seed vbar
        vbar = _dd(float)
        vbar[L-1] = 1.0

        # results and logs
        n = len(inputs)
        H = _np.zeros((n, n), dtype=float)
        logs = []

        # reverse sweep
        for i in range(L-1, -1, -1):
            node = _gt.nodes[i]
            parents = node.parents

            # ---- Creating: project second derivatives to input-input pairs ----
            sec = _second_locals(node)
            if vbar[i] != 0.0 and sec:
                # diag
                for (p_ad, _a) in parents:
                    pj = node_idx_of_advar.get(id(p_ad))
                    if ("diag", pj) in sec:
                        d2 = float(sec[("diag", pj)] * vbar[i])
                        cj = input_col.get(id(p_ad))
                        if cj is not None and d2 != 0.0:
                            pair = (cj, cj)
                            i0, j0 = min(pair), max(pair)
                            H[i0, j0] += d2
                            if focus_pairs is None or (i0, j0) in focus_pairs:
                                logs.append({
                                    "pair": (i0, j0),
                                    "node_idx": i,
                                    "op_tag": node.op_tag,
                                    "kind": "diag",
                                    "local_second": d2,
                                    "contrib": d2,
                                    "parents": [getattr(p_ad, "name", None)]
                                })
                # cross
                plist = [(node_idx_of_advar.get(id(pa)), pa) for (pa, _) in parents]
                m = len(plist)
                for u in range(m):
                    j_idx, j_ad = plist[u]
                    if j_idx is None: 
                        continue
                    for v2 in range(u+1, m):
                        k_idx, k_ad = plist[v2]
                        if k_idx is None: 
                            continue
                        key = ("cross", (min(j_idx, k_idx), max(j_idx, k_idx)))
                        if key in sec:
                            d2 = float(sec[key] * vbar[i])
                            cj = input_col.get(id(j_ad))
                            ck = input_col.get(id(k_ad))
                            if (cj is not None) and (ck is not None) and d2 != 0.0:
                                i0, j0 = (min(cj, ck), max(cj, ck))
                                H[i0, j0] += d2
                                if focus_pairs is None or (i0, j0) in focus_pairs:
                                    logs.append({
                                        "pair": (i0, j0),
                                        "node_idx": i,
                                        "op_tag": node.op_tag,
                                        "kind": "cross",
                                        "local_second": d2,
                                        "contrib": d2,
                                        "parents": [
                                            getattr(j_ad, "name", None),
                                            getattr(k_ad, "name", None),
                                        ],
                                    })

            # ---- Adjoint: vbar propagation ----
            if vbar[i] != 0.0:
                for (j_ad, a_j) in parents:
                    j_idx = node_idx_of_advar.get(id(j_ad))
                    if j_idx is not None:
                        vbar[j_idx] += (vbar[i] * a_j)

        # symmetrize
        for i in range(n):
            for j in range(i):
                H[j, i] = H[i, j]
        return H, logs


