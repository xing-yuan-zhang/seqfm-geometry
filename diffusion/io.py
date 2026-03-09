import networkx as nx

def read_nodes(path):
    s = set()
    with open(path, "r") as f:
        for line in f:
            t = line.strip()
            if t:
                s.add(t.split()[0])
    return s

def load_edges_tsv(path, node_whitelist=None, min_w=0.0):
    G = nx.Graph()
    with open(path, "r") as f:
        for line in f:
            a = line.rstrip("\n").split("\t")
            if len(a) < 3:
                continue
            u, v = a[0], a[1]
            if u == v:
                continue
            if node_whitelist is not None and (u not in node_whitelist or v not in node_whitelist):
                continue
            try:
                w = float(a[2])
            except:
                continue
            if w <= min_w:
                continue
            if G.has_edge(u, v):
                if w > G[u][v].get("weight", 0.0):
                    G[u][v]["weight"] = w
            else:
                G.add_edge(u, v, weight=w)
    return G

def load_labels(path, nodes_universe=None):
    y = {}
    with open(path, "r") as f:
        for line in f:
            a = line.rstrip("\n").split("\t")
            if len(a) < 2:
                continue
            n = a[0].strip()
            if not n:
                continue
            if nodes_universe is not None and n not in nodes_universe:
                continue
            try:
                v = float(a[1])
            except:
                continue
            y[n] = 1.0 if v > 0 else 0.0
    return y

def load_seeds(path, nodes_universe=None):
    w = {}
    with open(path, "r") as f:
        for line in f:
            a = line.rstrip("\n").split("\t")
            if not a:
                continue
            n = a[0].strip()
            if not n:
                continue
            if nodes_universe is not None and n not in nodes_universe:
                continue
            wt = 1.0
            if len(a) >= 2:
                try:
                    wt = float(a[1])
                except:
                    wt = 1.0
            if wt <= 0:
                continue
            w[n] = w.get(n, 0.0) + wt
    if not w:
        raise RuntimeError("no valid seeds")
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}