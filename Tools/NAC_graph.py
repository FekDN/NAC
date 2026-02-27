# NAC_graph.py
# Copyright (c) 2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0
#
# Interactive computation graph visualizer for .nac files.
# Serves an interactive D3.js force-directed graph on http://localhost:5050
#
# Usage:
#   python NAC_graph.py <model.nac>
#   python NAC_graph.py <model.nac> --port 8080 --no-browser

import sys
import os
import struct
import json
import argparse
import webbrowser
import threading
from typing import Dict, Any, List, Tuple, Optional

try:
    from flask import Flask, jsonify
except ImportError:
    print("Flask is required. Install with: pip install flask")
    sys.exit(1)

try:
    from NAC_kernels import NAC_OPS
except ImportError:
    NAC_OPS = {}


# ─────────────────────────────────────────────────────────────────────────────
# NAC Parser — produces a graph {nodes, edges} from a .nac file
# ─────────────────────────────────────────────────────────────────────────────

class NACGraphParser:
    DTYPE_MAP = {
        0:'float32', 1:'float64', 2:'float16', 3:'bfloat16',
        4:'int32', 5:'int64', 6:'int16', 7:'int8', 8:'uint8', 9:'bool'
    }
    CODE_TO_CATEGORY = {
        'Q':'offset','K':'offset','V':'offset','M':'offset',
        'B':'offset','W':'offset','T':'offset','P':'offset',
        'S':'const','A':'const','f':'const','i':'const','b':'const','s':'const','c':'const',
    }

    def __init__(self):
        self.id_to_canonical: Dict[int, str] = {
            2: "<INPUT>", 3: "<OUTPUT>", 6: "<CONTROL_FLOW>", 7: "<CONVERGENCE>"
        }
        self.id_to_canonical.update(NAC_OPS)
        self.constants: Dict[int, Any] = {}
        self.permutations: Dict[int, Tuple] = {}
        self.param_id_to_name: Dict[int, str] = {}
        self.input_node_idx_to_name: Dict[int, str] = {}
        self.param_metadata: Dict[int, Dict] = {}
        self.parsed_nodes: List[Dict] = []
        self.io_counts: Tuple[int, int] = (0, 0)
        self.d_model: int = 0
        self.quantization_method: str = "none"
        self.mep_bytecode_len: int = 0
        self.mep_const_count: int = 0

    def _infer_special_cd_lengths(self, A, B):
        name = self.id_to_canonical.get(A)
        if name == "<INPUT>":  return (2 if B in (1, 2, 3) else 0), 0
        if name == "<OUTPUT>": return self.io_counts[1] + 1, self.io_counts[1]
        if name == "<CONTROL_FLOW>": return 3, 1
        if name == "<CONVERGENCE>": return -1, -1
        return 0, 0

    def _read_op(self, f):
        A, B = struct.unpack('<BB', f.read(2))
        C, D = [], []
        if A < 10:
            nC, nD = self._infer_special_cd_lengths(A, B)
            if nC > 0: C = list(struct.unpack(f'<{nC}h', f.read(nC * 2)))
            if nD > 0: D = list(struct.unpack(f'<{nD}h', f.read(nD * 2)))
        else:
            perm = self.permutations.get(B)
            if perm:
                n_const = sum(1 for p in perm if self.CODE_TO_CATEGORY.get(p) == 'const')
                if n_const > 0:
                    nc, = struct.unpack('<h', f.read(2))
                    C = [nc] + list(struct.unpack(f'<{nc}h', f.read(nc*2))) if nc > 0 else [0]
                nD = len(perm)
                if nD > 0: D = list(struct.unpack(f'<{nD}h', f.read(nD*2)))
        return {'A': A, 'B': B, 'C': C, 'D': D}

    def load(self, nac_path: str):
        with open(nac_path, 'rb') as f:
            if f.read(3) != b'NAC': raise ValueError("Not a NAC file")
            version, quant_byte = struct.unpack('<BB', f.read(2))
            self.weights_stored_internally = (quant_byte & 0x80) != 0
            qmap = {0:'none',1:'FP16',2:'INT8_TENSOR',3:'INT8_CHANNEL',4:'BLOCK_FP8'}
            self.quantization_method = qmap.get(quant_byte & 0x7F, f'q{quant_byte&0x7F}')
            n_in, n_out, _ = struct.unpack('<HHB', f.read(5))
            self.io_counts = (n_in, n_out)

            hdr_fmt = '<H9Q4x'
            raw = struct.unpack(hdr_fmt, f.read(struct.calcsize(hdr_fmt)))
            self.d_model = raw[0]
            mmap_off, ops_off, cmap_off, cnst_off, perm_off, \
            data_off, proc_off, orch_off, rsrc_off = raw[1:]

            if cmap_off:
                f.seek(cmap_off); f.read(4)
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    oid, nl = struct.unpack('<HB', f.read(3))
                    self.id_to_canonical[oid] = f.read(nl).decode()

            if cnst_off:
                f.seek(cnst_off); f.read(4)
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    cid, tc, ln = struct.unpack('<HBH', f.read(5))
                    v = None
                    if   tc == 1: v = struct.unpack('<?', f.read(ln))[0]
                    elif tc == 2: v = struct.unpack('<q', f.read(ln))[0]
                    elif tc == 3: v = struct.unpack('<d', f.read(ln))[0]
                    elif tc == 4: v = f.read(ln).decode('utf-8')
                    elif tc == 5: v = list(struct.unpack(f'<{ln}i', f.read(ln*4))) if ln else []
                    elif tc == 6: v = list(struct.unpack(f'<{ln}f', f.read(ln*4))) if ln else []
                    self.constants[cid] = v

            if perm_off:
                f.seek(perm_off); f.read(4)
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    pid, pl = struct.unpack('<HB', f.read(3))
                    self.permutations[pid] = tuple(f.read(pl).decode())

            if ops_off:
                f.seek(ops_off); f.read(4)
                n_ops = struct.unpack('<I', f.read(4))[0]
                self.parsed_nodes = [self._read_op(f) for _ in range(n_ops)]

            if data_off:
                f.seek(data_off); f.read(4)
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    pid, nl = struct.unpack('<HH', f.read(4))
                    self.param_id_to_name[pid] = f.read(nl).decode()
                for _ in range(struct.unpack('<I', f.read(4))[0]):
                    iidx, nl = struct.unpack('<HH', f.read(4))
                    self.input_node_idx_to_name[iidx] = f.read(nl).decode()
                if self.weights_stored_internally:
                    for _ in range(struct.unpack('<I', f.read(4))[0]):
                        pid, ml, dl = struct.unpack('<HIQ', f.read(14))
                        mb = f.read(ml); f.seek(dl, 1)
                        mo, meta = 0, {}
                        did, rank = struct.unpack_from('<BB', mb, mo); mo += 2
                        meta['dtype'] = self.DTYPE_MAP.get(did, 'unk')
                        meta['shape'] = list(struct.unpack_from(f'<{rank}I', mb, mo)) if rank else []
                        mo += rank * 4
                        qtc, = struct.unpack_from('<B', mb, mo); mo += 1
                        qts = {0:'none',1:'FP16',2:'INT8_TENSOR',3:'INT8_CHANNEL',4:'BLOCK_FP8'}.get(qtc,'none')
                        if qts != 'none': meta['quant'] = qts
                        meta['data_bytes'] = dl
                        self.param_metadata[pid] = meta

            if orch_off:
                f.seek(orch_off); f.read(4)  # skip magic
                blen, cc = struct.unpack('<II', f.read(8))
                self.mep_bytecode_len = blen
                self.mep_const_count = cc

    def build_graph(self) -> Dict:
        """Convert parsed NAC ops into {nodes, edges, meta}."""
        nodes = []
        edges = []

        # Group categories
        OP_GROUPS = {
            'linear': ['aten.linear', 'aten.mm', 'aten.addmm', 'aten.matmul'],
            'norm':   ['aten.layer_norm', 'aten.batch_norm', 'aten.group_norm'],
            'act':    ['aten.relu', 'aten.gelu', 'aten.silu', 'aten.tanh', 'aten.sigmoid'],
            'attn':   ['aten.scaled_dot_product_attention', 'aten.softmax'],
            'embed':  ['aten.embedding'],
            'arith':  ['aten.add', 'aten.mul', 'aten.sub', 'aten.div', 'aten.pow'],
            'shape':  ['aten.view', 'aten.reshape', 'aten.permute', 'aten.transpose',
                       'aten.unsqueeze', 'aten.squeeze', 'aten.expand', 'aten.contiguous',
                       'aten.slice', 'aten.select', 'aten.cat', 'aten.stack'],
            'reduce': ['aten.sum', 'aten.mean', 'aten.max', 'aten.min', 'aten.argmax'],
        }
        def get_group(name):
            for g, ops in OP_GROUPS.items():
                if any(name.endswith(o) or o in name for o in ops):
                    return g
            return 'other'

        param_node_ids = set()  # node indices that are param loads

        for i, node in enumerate(self.parsed_nodes):
            A, B, C, D = node['A'], node['B'], node['C'], node['D']
            op_name = self.id_to_canonical.get(A, f'<OP_{A}>')
            nid = f'n{i}'

            if op_name == '<INPUT>':
                if B == 0:
                    # User input — shown as input terminal
                    label = self.input_node_idx_to_name.get(i, f'input_{i}')
                    nodes.append({
                        'id': nid, 'index': i, 'type': 'input',
                        'label': label, 'group': 'input',
                        'detail': {'name': label}
                    })
                elif B == 1:
                    # Parameter / weight tensor — shown as a block
                    param_id = C[1] if len(C) > 1 else -1
                    pname = self.param_id_to_name.get(param_id, f'param_{param_id}')
                    meta  = self.param_metadata.get(param_id, {})
                    short = pname.split('.')[-1] if '.' in pname else pname
                    nodes.append({
                        'id': nid, 'index': i, 'type': 'param',
                        'label': short, 'group': 'param',
                        'detail': {
                            'full_name': pname,
                            'shape':     meta.get('shape', []),
                            'dtype':     meta.get('dtype', '?'),
                            'quant':     meta.get('quant', 'none'),
                            'bytes':     meta.get('data_bytes', 0),
                        }
                    })
                    param_node_ids.add(i)
                elif B == 3:
                    # Lifted scalar constant
                    cid = C[1] if len(C) > 1 else -1
                    val = self.constants.get(cid)
                    nodes.append({
                        'id': nid, 'index': i, 'type': 'const',
                        'label': repr(val)[:16], 'group': 'const',
                        'detail': {'value': repr(val)}
                    })
                else:
                    nodes.append({
                        'id': nid, 'index': i, 'type': 'const',
                        'label': f'state_{i}', 'group': 'const',
                        'detail': {}
                    })

            elif op_name == '<OUTPUT>':
                # Collect what the output depends on (D contains offsets)
                label = 'OUTPUT'
                nodes.append({
                    'id': nid, 'index': i, 'type': 'output',
                    'label': label, 'group': 'output',
                    'detail': {}
                })
                for offset in D:
                    src = i + offset
                    edges.append({
                        'source': f'n{src}', 'target': nid,
                        'var': f'v{src}', 'type': 'data'
                    })

            elif op_name in ('<CONTROL_FLOW>', '<CONVERGENCE>', '<NONE>'):
                nodes.append({
                    'id': nid, 'index': i, 'type': 'control',
                    'label': op_name, 'group': 'control',
                    'detail': {}
                })

            else:
                # Regular operation node
                perm = self.permutations.get(B, ())
                c_ids = C[1:] if C and C[0] > 0 else []
                c_iter = iter(c_ids)
                const_vals = []

                for di, d_val in enumerate(D):
                    if d_val != 0:
                        src = i + d_val
                        edges.append({
                            'source': f'n{src}', 'target': nid,
                            'var': f'v{src}', 'type': 'data',
                            'is_param': src in param_node_ids
                        })
                    else:
                        try:
                            cid = next(c_iter)
                            const_vals.append(self.constants.get(cid))
                        except StopIteration:
                            const_vals.append(None)

                # Clean up display name
                display = op_name.replace('aten.', '').replace('_', '\u2009')
                if display.endswith('.default'): display = display[:-8]
                group = get_group(op_name)

                perm_inputs = [p for p in perm if self.CODE_TO_CATEGORY.get(p) == 'offset']
                perm_consts = [p for p in perm if self.CODE_TO_CATEGORY.get(p) == 'const']

                nodes.append({
                    'id': nid, 'index': i, 'type': 'op',
                    'label': display, 'group': group,
                    'detail': {
                        'full_name': op_name,
                        'perm': ''.join(perm),
                        'n_tensor_inputs': len(perm_inputs),
                        'n_scalar_inputs': len(perm_consts),
                        'const_vals': [repr(v)[:40] for v in const_vals],
                    }
                })

        return {
            'nodes': nodes,
            'edges': edges,
            'meta': {
                'd_model':      self.d_model,
                'quant':        self.quantization_method,
                'n_ops':        len(self.parsed_nodes),
                'n_params':     len(self.param_metadata),
                'io_counts':    list(self.io_counts),
                'mep_bytes':    self.mep_bytecode_len,
                'mep_consts':   self.mep_const_count,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# Flask App
# ─────────────────────────────────────────────────────────────────────────────


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>NAC Graph — {filename}</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');
:root {
  --bg:#08090c; --bg2:#0e1117; --bg3:#161b24; --grid:#171f2e; --border:#1e2d45;
  --text:#c8d8f0; --dim:#3a5070; --cyan:#00e5ff; --w:#f8f8f2;
  --ci:#00ff88; --co:#00e5ff; --cp:#ff8c00; --ck:#6272a4; --cc:#ff3c5a;
  --cl:#bd93f9; --cn:#8be9fd; --ca:#ffb86c; --cat:#ff79c6; --ce:#f1fa8c;
  --car:#5af78e; --cs:#57c7ff; --cr:#ff6ac1; --cx:#4a6080;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{width:100%;height:100%;overflow:hidden}
body{background:var(--bg);color:var(--text);font-family:'IBM Plex Mono',monospace}
body::before{content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:linear-gradient(var(--grid) 1px,transparent 1px),
    linear-gradient(90deg,var(--grid) 1px,transparent 1px);
  background-size:40px 40px}

/* ── Topbar ── */
#tb{position:fixed;top:0;left:0;right:0;z-index:100;height:48px;
  background:#090c12e8;border-bottom:1px solid var(--border);
  backdrop-filter:blur(10px);display:flex;align-items:center;gap:14px;padding:0 18px}
.brand{font-size:13px;font-weight:600;color:var(--cyan);letter-spacing:.12em;flex-shrink:0}
.pill{font-size:11px;padding:2px 9px;border-radius:3px;background:var(--bg3);
  border:1px solid var(--border);color:var(--dim);white-space:nowrap;flex-shrink:0}
.pill b{color:var(--w);font-weight:400}
#ctl{margin-left:auto;display:flex;gap:7px;align-items:center}
.btn{font-family:inherit;font-size:11px;padding:4px 11px;border-radius:3px;cursor:pointer;
  border:1px solid var(--border);background:var(--bg3);color:var(--text);
  transition:border-color .12s,color .12s}
.btn:hover{border-color:var(--cyan);color:var(--cyan)}
.btn.on{border-color:var(--cyan);color:var(--cyan);background:#001c2a80}
.btn:disabled{cursor:not-allowed;color:var(--dim);border-color:var(--border)}
#sw{position:relative}
#sq{font-family:inherit;font-size:11px;padding:4px 9px 4px 26px;width:170px;
  background:var(--bg3);border:1px solid var(--border);border-radius:3px;
  color:var(--text);outline:none;transition:border-color .12s}
#sq:focus{border-color:var(--cyan)}
#sw::before{content:'⌕';position:absolute;left:7px;top:50%;transform:translateY(-50%);
  color:var(--dim);font-size:13px;pointer-events:none}

/* ── Pattern Finder ── */
#pinfo{position:fixed;bottom:18px;right:18px;z-index:100;
  background:#090c12e8;border:1px solid var(--border);border-radius:4px;
  padding:10px 14px;backdrop-filter:blur(8px);
  font-size:11px;line-height:1.9;display:none;width:240px}
.pi-h{font-weight:600;color:var(--cyan);margin-bottom:5px;font-size:12px}
.pi-r{display:flex;gap:8px}
.pi-k{color:var(--dim);min-width:60px;flex-shrink:0}
.pi-v{color:var(--w);word-break:break-all}
.lk.pattern-hl{stroke:#ffeb3b;stroke-width:2.8;opacity:1}
.nd.pattern-hl > * {
  stroke:#ffeb3b !important;stroke-width:2.5px !important;
  filter: drop-shadow(0 0 8px #ffeb3b) !important;
}
.nd.pattern-sel > * {
  stroke:#ff5722 !important;stroke-width:2.8px !important;
  filter: drop-shadow(0 0 10px #ff5722) !important;
}

/* ── SVG canvas ── */
#cv{position:fixed;inset:0;top:48px;z-index:1}
svg{display:block;width:100%;height:100%}

/* ── Edges ── */
.lk{fill:none;stroke:#5a7fb0;stroke-width:1.2;opacity:.65; transition: stroke .2s, opacity .2s;}
.lk.pe{stroke:#ff8c00;stroke-dasharray:4 3;opacity:.4}
.lk.hl{stroke:var(--cyan);stroke-width:2;opacity:1}
.lk.active{stroke:#00e5ff;stroke-width:2.5;opacity:1}

/* ── Nodes ── */
.nd{cursor:pointer;transition:opacity .15s}
.nd.dim{opacity:.08}
.nd.fired > * { filter: drop-shadow(0 0 8px #00e5ff) !important; }
.nd.ready > circle, .nd.ready > rect, .nd.ready > polygon { stroke: #fff !important; stroke-width: 2.2px !important; }

/* ── Tooltip ── */
#tt{position:fixed;z-index:200;pointer-events:none;
  background:#08090cf2;border:1px solid var(--border);
  border-radius:4px;padding:10px 14px;max-width:290px;
  font-size:11px;line-height:1.8;backdrop-filter:blur(8px);display:none}
.tt-h{font-weight:600;color:var(--cyan);margin-bottom:5px;font-size:12px}
.tt-r{display:flex;gap:8px}
.tt-k{color:var(--dim);min-width:76px;flex-shrink:0}
.tt-v{color:var(--w);word-break:break-all}

/* ── Legend ── */
#leg{position:fixed;bottom:18px;left:18px;z-index:100;
  background:#090c12cc;border:1px solid var(--border);border-radius:4px;
  padding:10px 14px;backdrop-filter:blur(8px);
  font-size:10px;line-height:2;
  display:grid;grid-template-columns:1fr 1fr;gap:0 18px}
.li{display:flex;align-items:center;gap:7px;color:var(--dim)}
.ld{width:9px;height:9px;border-radius:50%;flex-shrink:0}
.lr{width:13px;height:7px;border-radius:2px;flex-shrink:0;background:transparent}

/* ── Progress indicator ── */
#prog{position:fixed;bottom:18px;right:18px;z-index:100;font-size:10px;
  color:var(--dim);background:#090c12cc;border:1px solid var(--border);
  border-radius:3px;padding:4px 10px}
</style>
</head>
<body>
<div id="tb">
  <div class="brand">NAC<span style="color:var(--dim)"> // </span>GRAPH</div>
  <div class="pill">file: <b id="hf">-</b></div>
  <div class="pill">ops: <b id="ho">-</b></div>
  <div class="pill">params: <b id="hp">-</b></div>
  <div class="pill">d_model: <b id="hd">-</b></div>
  <div class="pill">quant: <b id="hq">-</b></div>
  <div class="pill" id="hm" style="display:none">mep: <b id="hmv">-</b></div>
  <div id="ctl">
    <div id="sw"><input id="sq" placeholder="filter…" autocomplete="off"/></div>
    <!-- MODIFIED/ADDED buttons -->
    <button class="btn"     id="bS">step</button>
    <button class="btn"     id="bSA">play</button>
    <button class="btn on"  id="bP">weights</button>
    <button class="btn on"  id="bC">consts</button>
    <button class="btn"     id="bU">untangle</button>
    <button class="btn"     id="bPF">find similar</button>
    <button class="btn"     id="bF">fit</button>
    <button class="btn"     id="bR">reset</button>
  </div>
</div>
<div id="cv">
  <svg id="sv">
    <defs>
      <marker id="ar" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L0,8 L8,4 z" fill="#2a4060"/></marker>
      <marker id="arh" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L0,8 L8,4 z" fill="#00e5ff"/></marker>
      <marker id="arp" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="5" markerHeight="5" orient="auto"><path d="M0,0 L0,8 L8,4 z" fill="#5a3010"/></marker>
    </defs>
    <g id="root"><g id="gl"></g><g id="gn"></g></g>
  </svg>
</div>
<div id="tt"></div>
<div id="leg">
  <div class="li"><div class="ld" style="background:var(--ci)"></div>user input</div>
  <div class="li"><div class="ld" style="background:var(--co)"></div>model output</div>
  <div class="li"><div class="lr" style="border:1px solid var(--cp)"></div>weight tensor</div>
  <div class="li"><div class="ld" style="background:var(--cl)"></div>linear/matmul</div>
  <div class="li"><div class="ld" style="background:var(--cat)"></div>attention</div>
  <div class="li"><div class="ld" style="background:var(--cn)"></div>norm</div>
  <div class="li"><div class="ld" style="background:var(--ca)"></div>activation</div>
  <div class="li"><div class="ld" style="background:var(--car)"></div>arithmetic</div>
  <div class="li"><div class="ld" style="background:var(--cs)"></div>reshape/view</div>
  <div class="li"><div class="ld" style="background:var(--ce)"></div>embedding</div>
  <div class="li"><div class="ld" style="background:var(--cr)"></div>reduce</div>
  <div class="li"><div class="ld" style="background:var(--cx)"></div>other</div>
</div>
<div id="prog">settling…</div>
<div id="pinfo"></div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.9.0/d3.min.js"></script>
<script>
// ══════════════════════════════════════════════════════════════════
// Constants & helpers
// ══════════════════════════════════════════════════════════════════

const COL = {
  input:'#00ff88', output:'#00e5ff', param:'#ff8c00', const:'#6272a4',
  control:'#ff3c5a', linear:'#bd93f9', norm:'#8be9fd', act:'#ffb86c',
  attn:'#ff79c6', embed:'#f1fa8c', arith:'#5af78e', shape:'#57c7ff',
  reduce:'#ff6ac1', other:'#4a6080',
};
const nc  = n => COL[n.type] || COL[n.group] || COL.other;
const tr  = (s, n) => (s && s.length > n) ? s.slice(0, n) + '…' : (s || '');

function nr(n){
  if(!n)return 20;
  switch(n.type){
    case 'param': return 18; case 'input': return 38; case 'output': return 38;
    case 'const': return 12; case 'control': return 27; default: return 22;
  }
}
function eid(e, f) { const v = e[f]; return (v !== null && typeof v === 'object') ? v.id : v; }

// ══════════════════════════════════════════════════════════════════
// State
// ══════════════════════════════════════════════════════════════════
let showP = true, showC = true;
let allNodes = [], allEdges = [];
let nodeMap = {}; // Added for quick node lookup by ID
let _sim, _zoom, _svg, _nG, _lk, _hl = null;
let simState = {}, readyQueue = [], isSimActive = false, _autoSimInterval = null;

// --- PATTERN FINDER STATE ---
let isPatternFindMode = false;
let selectedPattern = []; // Array of node objects in selection order


// ══════════════════════════════════════════════════════════════════
// Topological rank-based initial layout
// ══════════════════════════════════════════════════════════════════
function computeRanks(nodes, edges) {
  const rank = {};
  nodes.forEach(n => rank[n.id] = 0);
  const preds = {};
  nodes.forEach(n => preds[n.id] = []);
  edges.forEach(e => { const s=eid(e,'source'), t=eid(e,'target'); if(preds[t]!==undefined) preds[t].push(s); });
  const sorted = [...nodes].sort((a,b) => a.index-b.index);
  sorted.forEach(n => { const ps = preds[n.id]; rank[n.id] = ps.length ? Math.max(...ps.map(p=>(rank[p]||0)))+1 : 0; });
  return rank;
}
function applyRankLayout(nodes, edges, W, H, force) {
  const toPlace = force ? nodes : nodes.filter(n => n.x==null);
  if(!toPlace.length) return;
  const ranks=computeRanks(nodes,edges), maxR=Math.max(...nodes.map(n=>ranks[n.id]||0),1), buckets={};
  nodes.forEach(n => { const r=ranks[n.id]||0; (buckets[r]=buckets[r]||[]).push(n); });
  Object.values(buckets).forEach(b => b.sort((a,b)=>(a.type==='param'?1:0)-(b.type==='param'?1:0)));
  const margin=80, usableW=W-margin*2, usableH=H-margin*2;
  (force?nodes:toPlace).forEach(n=>{
    const r=ranks[n.id]||0, bucket=buckets[r];
    const isParam = n.type==='param'||n.type==='const';
    const mainOps=bucket.filter(x=>x.type!=='param'&&x.type!=='const');
    const auxOps =bucket.filter(x=>x.type==='param'||x.type==='const');
    let yFrac;
    if(!isParam){ const mi=mainOps.indexOf(n); yFrac=mainOps.length>1?0.08+(mi/(mainOps.length-1))*0.58:0.38; }
    else { const ai=auxOps.indexOf(n); yFrac=auxOps.length>1?0.72+(ai/(auxOps.length-1))*0.22:0.83; }
    n.x=margin+(r/maxR)*usableW+(Math.random()-0.5)*18;
    n.y=margin+yFrac*usableH+(Math.random()-0.5)*12;
    n.vx=0; n.vy=0; delete n.fx; delete n.fy;
  });
}

// ══════════════════════════════════════════════════════════════════
// Main build function
// ══════════════════════════════════════════════════════════════════
function buildGraph(nodes, edges, forceLayout) {
  const svg = d3.select('#sv'), W = svg.node().clientWidth, H = svg.node().clientHeight;
  if (!_zoom) {
    _zoom = d3.zoom().scaleExtent([.015,6]).on('zoom',e=>d3.select('#root').attr('transform',e.transform));
    svg.call(_zoom).on('dblclick.zoom',null); _svg=svg;
  }
  stopAllSimulations();
  const visN=nodes.filter(n=>(showP||n.type!=='param')&&(showC||n.type!=='const'));
  const visIds=new Set(visN.map(n=>n.id));
  const visE=edges.filter(e=>visIds.has(eid(e,'source'))&&visIds.has(eid(e,'target')));
  const simE=visE.map(e=>({...e,source:eid(e,'source'),target:eid(e,'target')}));
  applyRankLayout(visN,simE,W,H,forceLayout);
  if(_sim) _sim.stop();
  d3.select('#gl').selectAll('*').remove();
  d3.select('#gn').selectAll('*').remove();
  _hl=null;
  _sim=d3.forceSimulation(visN)
    .force('link',d3.forceLink(simE).id(d=>d.id).distance(d=>d.is_param?36:60).strength(d=>d.is_param?0.9:0.45))
    .force('charge',d3.forceManyBody().strength(n=>n.type==='param'?-35:-100))
    .force('collide',d3.forceCollide(n=>nr(n)+12).strength(1))
    .force('x',d3.forceX().strength(0.1).x(n=>n.type==='input'?W*0.05:(n.type==='output'?W*0.95:n.x)))
    .force('y',d3.forceY().strength(0.03).y(H/2))
    .alphaDecay(0.018).velocityDecay(0.45)
    .on('end',()=>document.getElementById('prog').textContent='settled');
  _lk=d3.select('#gl').selectAll('path').data(simE).join('path')
    .attr('class',d=>'lk'+(d.is_param?' pe':'')).attr('marker-end',d=>d.is_param?'url(#arp)':'url(#ar)');
  _nG=d3.select('#gn').selectAll('g').data(visN,d=>d.id).join('g').attr('class','nd')
    .call(d3.drag().on('start',(ev,d)=>{if(!ev.active)_sim.alphaTarget(.15).restart();d.fx=d.x;d.fy=d.y;})
    .on('drag',(ev,d)=>{d.fx=ev.x;d.fy=ev.y;}).on('end',(ev,d)=>{if(!ev.active)_sim.alphaTarget(0);d.fx=null;d.fy=null;}))
    .on('mouseover',showTT).on('mousemove',moveTT).on('mouseout',hideTT)
    .on('click',(ev,d) => {
      if (isPatternFindMode) { handlePatternClick(d); }
      else { hlNode(d); }
    });
  _nG.each(function(d){
    const g=d3.select(this),c=nc(d),gl=`drop-shadow(0 0 5px ${c}88)`;
    if(d.type==='param'){
      g.append('rect').attr('width',30).attr('height',18).attr('x',-15).attr('y',-9).attr('rx',3)
        .style('fill','#18100080').style('stroke',c).style('stroke-width',1.2).style('filter',gl);
      g.append('text').attr('text-anchor','middle').attr('y',-2).attr('font-size',8).style('fill',c).text(tr(d.label,12));
      const sh=(d.detail.shape||[]).join('×'); if(sh)g.append('text').attr('text-anchor','middle').attr('y',9).attr('font-size',7).style('fill','#485868').text(sh);
    }else if(d.type==='input'){
      g.append('polygon').attr('points','-34,0 -16,-16 30,-16 30,16 -16,16').style('fill','#001a0e90').style('stroke',c).style('stroke-width',1.5).style('filter',gl);
      g.append('text').attr('text-anchor','middle').attr('dominant-baseline','central').attr('x',2).attr('font-size',9).style('fill',c).text('▶ '+tr(d.label,8));
    }else if(d.type==='output'){
      g.append('polygon').attr('points','-30,-16 16,-16 34,0 16,16 -30,16').style('fill','#00141a90').style('stroke',c).style('stroke-width',1.5).style('filter',gl);
      g.append('text').attr('text-anchor','middle').attr('dominant-baseline','central').attr('x',-4).attr('font-size',9).style('fill',c).text(tr(d.label,8)+' ▶');
    }else if(d.type==='const'){
      g.append('polygon').attr('points','0,-12 12,0 0,12 -12,0').style('fill','#0e111780').style('stroke',c).style('stroke-width',.8);
      g.append('text').attr('text-anchor','middle').attr('y',21).attr('font-size',7).style('fill',c).text(tr(d.label,10));
    }else if(d.type==='control'){
      g.append('rect').attr('width',46).attr('height',20).attr('x',-23).attr('y',-10);
      g.append('text').attr('text-anchor','middle').attr('dominant-baseline','central').attr('font-size',8).style('fill',c).text(tr(d.label,8));
    }else{
      g.append('circle').attr('r',19).style('fill','#0b0f18').style('stroke',c).style('stroke-width',1.6).style('filter',gl);
      g.append('text').attr('text-anchor','middle').attr('dominant-baseline','central').attr('font-size',8.5).style('fill','#ccdcf0').text(tr(d.label,9));
    }
  });
  _sim.on('tick',()=>{
    _lk.attr('d',d=>{
      const sx=d.source.x||0,sy=d.source.y||0,tx=d.target.x||0,ty=d.target.y||0;
      const dx=tx-sx,dy=ty-sy,len=Math.sqrt(dx*dx+dy*dy)||1;
      const sR=nr(d.source)+2,tR=nr(d.target)+8;
      if(len<sR+tR+4)return'';
      const sx2=sx+(dx/len)*sR,sy2=sy+(dy/len)*sR,ex=tx-(dx/len)*tR,ey=ty-(dy/len)*tR;
      return `M${sx2},${sy2} L${ex},${ey}`;
    });
    _nG.attr('transform',d=>`translate(${d.x||0},${d.y||0})`);
    updateProgress();
  });
  setTimeout(fitView,2800);
}

// ══════════════════════════════════════════════════════════════════
// Utility Functions (Progress, Fit, Highlight, Tooltip)
// ══════════════════════════════════════════════════════════════════
function updateProgress(){if(!_sim)return;const a=_sim.alpha();const prog=document.getElementById('prog');if(a>0.02)prog.textContent='settling… '+Math.round(a*100)+'%';}
function fitView(){if(!_svg||!_zoom)return;const ns=allNodes.filter(n=>n.x!=null);if(!ns.length)return;const xs=ns.map(n=>n.x),ys=ns.map(n=>n.y);const x0=Math.min(...xs)-70,x1=Math.max(...xs)+70;const y0=Math.min(...ys)-70,y1=Math.max(...ys)+70;const W=_svg.node().clientWidth,H=_svg.node().clientHeight;const sc=Math.min(.95,.95*Math.min(W/(x1-x0),H/(y1-y0)));const tx=W/2-sc*(x0+x1)/2,ty=H/2-sc*(y0+y1)/2;_svg.transition().duration(650).call(_zoom.transform,d3.zoomIdentity.translate(tx,ty).scale(sc));}
function hlNode(d){if(isPatternFindMode) return; if(_hl===d.id){_hl=null;_nG.classed('dim',false);_lk.classed('hl',false).attr('marker-end',e=>e.is_param?'url(#arp)':'url(#ar)');return;}_hl=d.id;const conn=new Set([d.id]);allEdges.forEach(e=>{const s=eid(e,'source'),t=eid(e,'target');if(s===d.id||t===d.id){conn.add(s);conn.add(t);}});_nG.classed('dim',n=>!conn.has(n.id));_lk.classed('hl',e=>eid(e,'source')===d.id||eid(e,'target')===d.id).attr('marker-end',e=>(eid(e,'source')===d.id||eid(e,'target')===d.id)?'url(#arh)':(e.is_param?'url(#arp)':'url(#ar)'));}
const _tt=document.getElementById('tt');
function showTT(ev,d){const r=(k,v)=>`<div class="tt-r"><span class="tt-k">${k}</span><span class="tt-v">${v}</span></div>`;let h=`<div class="tt-h">${d.label}</div>${r('type',d.type)}${r('v-index','v'+d.index)}`;if(d.detail){for(const[k,v] of Object.entries(d.detail)){if(v===null||v===''||(Array.isArray(v)&&!v.length))continue;h+=r(k,Array.isArray(v)?v.join(' × '):String(v));}}_tt.innerHTML=h;_tt.style.display='block';moveTT(ev);}
function moveTT(ev){let x=ev.clientX+15,y=ev.clientY+15;if(x+300>window.innerWidth)x=ev.clientX-300-15;if(y+220>window.innerHeight)y=ev.clientY-220-15;_tt.style.left=x+'px';_tt.style.top=y+'px';}
function hideTT(){_tt.style.display='none';}

// ══════════════════════════════════════════════════════════════════
// Signal Propagation Logic
// ══════════════════════════════════════════════════════════════════
function initSimulationState(nodes,edges){simState={};nodes.forEach(n=>{simState[n.id]={totalInputs:0,receivedInputs:0,status:'idle'};});edges.forEach(e=>{const t=eid(e,'target');if(simState[t])simState[t].totalInputs++;});}
function stopAllSimulations(){if(_autoSimInterval)clearInterval(_autoSimInterval);_autoSimInterval=null;isSimActive=false;const btnS=document.getElementById('bS'),btnA=document.getElementById('bSA');if(btnS){btnS.textContent='step';btnS.disabled=false;}if(btnA){btnA.disabled=false;}if(_nG)_nG.classed('fired ready',false);if(_lk)_lk.classed('active',false);}
function resetSimulation(){stopAllSimulations();readyQueue=[];Object.keys(simState).forEach(id=>{simState[id].receivedInputs=0;simState[id].status='idle';if(simState[id].totalInputs===0){simState[id].status='ready';readyQueue.push(id);}});_nG.classed('fired',false).classed('ready',d=>readyQueue.includes(d.id));_lk.classed('active',false);isSimActive=true;document.getElementById('bS').textContent='next step';}
function propagateStep(){if(!isSimActive||readyQueue.length===0){stopAllSimulations();document.getElementById('bS').textContent='restart';return false;}const current= [...readyQueue],next=new Set();_nG.filter(n=>current.includes(n.id)).classed('ready',false).classed('fired',true);current.forEach(id=>{simState[id].status='fired';});const activeEdges=allEdges.filter(e=>current.includes(eid(e,'source')));_lk.filter(e=>activeEdges.some(ae=>eid(ae,'source')===eid(e,'source')&&eid(ae,'target')===eid(e,'target'))).classed('active',true).transition().delay(300).on('end',function(){d3.select(this).classed('active',false);});activeEdges.forEach(edge=>{const t=eid(edge,'target'),ts=simState[t];if(ts&&ts.status!=='fired'){ts.receivedInputs++;if(ts.receivedInputs>=ts.totalInputs){if(ts.status!=='ready'){ts.status='ready';next.add(t);}}}});readyQueue=Array.from(next);_nG.filter(n=>readyQueue.includes(n.id)).classed('ready',true);return true;}
function runAutoSimulation(){resetSimulation();const btnS=document.getElementById('bS'),btnA=document.getElementById('bSA');btnS.disabled=true;btnA.disabled=true;_autoSimInterval=setInterval(()=>{if(!propagateStep()){/* stop handled inside */}},333);}

// ══════════════════════════════════════════════════════════════════
// Untangle Logic
// ══════════════════════════════════════════════════════════════════
let isUntangling = false;
function untangleGraph() {
  if (!_sim || isUntangling) return;
  isUntangling = true;
  const btn = document.getElementById('bU');
  btn.disabled = true;
  document.getElementById('prog').textContent = 'untangling…';
  const chargeForce = _sim.force('charge');
  const originalStrength = chargeForce.strength();
  chargeForce.strength(-800);
  _sim.alpha(0.7).restart();
  setTimeout(() => {
    chargeForce.strength(originalStrength);
    _sim.alpha(0.3).restart();
    isUntangling = false;
    btn.disabled = false;
    document.getElementById('prog').textContent = 'settling…';
  }, 1800);
}

// ══════════════════════════════════════════════════════════════════
// Pattern Finder Logic
// ══════════════════════════════════════════════════════════════════
// Helper function to check node identity
function areNodesIdentical(n1, n2) {
  if (!n1 || !n2 || !n1.detail || !n2.detail) return false;
  // For 'op' nodes, the core identity is the operation name.
  if (n1.type === 'op' && n2.type === 'op') {
      return n1.detail.full_name === n2.detail.full_name;
  }
  // You could add more sophisticated checks for other types if needed
  return false;
}

// Helper function to check if two nodes are connected in the main graph
function areNodesNeighbors(n1_id, n2_id) {
    return allEdges.some(e =>
        (eid(e, 'source') === n1_id && eid(e, 'target') === n2_id) ||
        (eid(e, 'source') === n2_id && eid(e, 'target') === n1_id)
    );
}

function handlePatternClick(d) {
    if (d.type !== 'op') {
        return; // Only allow 'op' nodes to be part of a pattern.
    }

    const nodeIndex = selectedPattern.findIndex(n => n.id === d.id);

    if (nodeIndex > -1) {
        // Node is already selected, so deselect it.
        selectedPattern.splice(nodeIndex, 1);
    } else {
        // It's a new node. Add it IF it's the first node OR connected to the existing pattern.
        if (selectedPattern.length === 0 || selectedPattern.some(p_node => areNodesNeighbors(p_node.id, d.id))) {
            selectedPattern.push(d);
        } else {
            // Ignore clicks on nodes not adjacent to the current selection.
            return;
        }
    }
    findAndShowPattern();
}

function getPatternAdjacency(pattern) {
    const adj = new Map(pattern.map(n => [n.id, { 'in': [], 'out': [] }]));
    const patternIds = new Set(pattern.map(n => n.id));

    for (const edge of allEdges) {
        const sourceId = eid(edge, 'source');
        const targetId = eid(edge, 'target');
        if (patternIds.has(sourceId) && patternIds.has(targetId)) {
            adj.get(targetId).in.push(sourceId);
            adj.get(sourceId).out.push(targetId);
        }
    }
    return adj;
}

function findAndShowPattern() {
    clearHighlights();
    const pInfo = document.getElementById('pinfo');

    if (selectedPattern.length === 0) {
        pInfo.style.display = 'none';
        return;
    }

    if (selectedPattern.length === 1) {
        const p_node = selectedPattern[0];
        const identicalNodes = allNodes.filter(n => areNodesIdentical(n, p_node));
        const foundNodeIds = new Set(identicalNodes.map(n => n.id));
        updatePatternUI(foundNodeIds, new Set(), identicalNodes.length);
        return;
    }

    // --- Subgraph Isomorphism Search ---
    const patternAdj = getPatternAdjacency(selectedPattern);
    const foundGroups = [];
    const visitedGroups = new Set(); // To store stringified versions of found groups to avoid duplicates

    // Backtracking search function
    function findMatches(patternNodeIndex, currentMapping) {
        if (patternNodeIndex === selectedPattern.length) {
            // --- Found a full match ---
            const matchedGroup = Object.values(currentMapping);
            const groupKey = matchedGroup.map(n => n.id).sort().join(',');
            if (!visitedGroups.has(groupKey)) {
                foundGroups.push(matchedGroup);
                visitedGroups.add(groupKey);
            }
            return;
        }

        const patternNode = selectedPattern[patternNodeIndex];
        const prevPatternNodes = selectedPattern.slice(0, patternNodeIndex);
        const alreadyMappedGraphNodeIds = new Set(Object.values(currentMapping).map(n => n.id));

        // Find candidate nodes in the main graph
        let candidates = allNodes.filter(n =>
            areNodesIdentical(n, patternNode) && !alreadyMappedGraphNodeIds.has(n.id)
        );

        for (const candidate of candidates) {
            // CORRECTED: Removed space in variable name
            let isTopologicallyConsistent = true;
            
            // Check connections to already-mapped nodes
            for (const prevPatternNode of prevPatternNodes) {
                const prevGraphNode = currentMapping[prevPatternNode.id];

                // Check outgoing edges from previous pattern node TO current pattern node
                if (patternAdj.get(patternNode.id).in.includes(prevPatternNode.id)) {
                    if (!allEdges.some(e => eid(e,'source') === prevGraphNode.id && eid(e,'target') === candidate.id)) {
                        // CORRECTED: Removed space in variable name
                        isTopologicallyConsistent = false; 
                        break;
                    }
                }
                
                // Check incoming edges TO previous pattern node FROM current pattern node
                if (patternAdj.get(patternNode.id).out.includes(prevPatternNode.id)) {
                    if (!allEdges.some(e => eid(e,'source') === candidate.id && eid(e,'target') === prevGraphNode.id)) {
                        // CORRECTED: Removed space in variable name
                        isTopologicallyConsistent = false; 
                        break;
                    }
                }
            }
            
            if (isTopologicallyConsistent) {
                const newMapping = { ...currentMapping, [patternNode.id]: candidate };
                findMatches(patternNodeIndex + 1, newMapping);
            }
        }
    }
    
    // Start the search
    findMatches(0, {});

    // Collect all nodes and edges from found groups for highlighting
    const foundNodeIds = new Set();
    const foundEdgeIds = new Set();
    foundGroups.forEach(group => {
        const groupIds = new Set(group.map(n => n.id));
        group.forEach(node => foundNodeIds.add(node.id));
        allEdges.forEach((e, idx) => {
            if (groupIds.has(eid(e, 'source')) && groupIds.has(eid(e, 'target'))) {
                foundEdgeIds.add(idx);
            }
        });
    });

    updatePatternUI(foundNodeIds, foundEdgeIds, foundGroups.length);
}

function updatePatternUI(nodeIds, edgeIds, groupCount) {
    const pInfo = document.getElementById('pinfo');
    if (groupCount === 0) {
        pInfo.style.display = 'none';
        return;
    }

    // Highlight found nodes and edges
    _nG.classed('pattern-hl', d => nodeIds.has(d.id));
    _lk.classed('pattern-hl', (d, i) => edgeIds.has(i));
    
    // Highlight the selected pattern itself more prominently
    const selectedIds = new Set(selectedPattern.map(n => n.id));
    _nG.classed('pattern-sel', d => selectedIds.has(d.id));

    // Update info panel
    const r = (k,v) => `<div class="pi-r"><span class="pi-k">${k}</span><span class="pi-v">${v}</span></div>`;
    let html = '';
    if (selectedPattern.length === 1) {
        html += `<div class="pi-h">Single Node Search</div>`;
        html += r('Operation', selectedPattern[0].detail.full_name);
        html += r('Found', `${groupCount} nodes`);
    } else {
        html += `<div class="pi-h">Pattern Search</div>`;
        const patternStr = selectedPattern.map(n => n.label).join(' → ');
        html += r('Pattern', tr(patternStr, 30));
        html += r('Found', `${groupCount} groups`);
    }
    pInfo.innerHTML = html;
    pInfo.style.display = 'block';
}

function clearHighlights() {
    _hl = null;
    _nG.classed('dim pattern-hl pattern-sel', false);
    _lk.classed('hl pattern-hl', false).attr('marker-end', e => e.is_param ? 'url(#arp)' : 'url(#ar)');
}

function clearPatternSelection() {
    selectedPattern = [];
    document.getElementById('pinfo').style.display = 'none';
    clearHighlights();
}

// ══════════════════════════════════════════════════════════════════
// Controls
// ══════════════════════════════════════════════════════════════════
document.getElementById('bP').addEventListener('click',function(){showP=!showP;this.classList.toggle('on',showP);buildGraph(allNodes,allEdges,false);});
document.getElementById('bC').addEventListener('click',function(){showC=!showC;this.classList.toggle('on',showC);buildGraph(allNodes,allEdges,false);});
document.getElementById('bF').addEventListener('click',fitView);
document.getElementById('bR').addEventListener('click',()=>{allNodes.forEach(n=>{delete n.x;delete n.y;delete n.vx;delete n.vy;delete n.fx;delete n.fy;});document.getElementById('prog').textContent='settling…';buildGraph(allNodes,allEdges,true);});
document.getElementById('bS').addEventListener('click',()=>{if(!isSimActive){resetSimulation();}propagateStep();});
document.getElementById('bSA').addEventListener('click',runAutoSimulation);
document.getElementById('bU').addEventListener('click', untangleGraph);

document.getElementById('bPF').addEventListener('click', function() {
    isPatternFindMode = !isPatternFindMode;
    this.classList.toggle('on', isPatternFindMode);
    clearPatternSelection();
});

let _st;
document.getElementById('sq').addEventListener('input',function(){clearTimeout(_st);_st=setTimeout(()=>{if(!_nG)return;const q=this.value.toLowerCase().trim();if(!q){_nG.classed('dim',false);return;}_nG.classed('dim',d=>!d.label.toLowerCase().includes(q)&&!(d.detail?.full_name||'').toLowerCase().includes(q));},100);});

// ══════════════════════════════════════════════════════════════════
// Bootstrap
// ══════════════════════════════════════════════════════════════════
fetch('/api/graph').then(r=>r.json()).then(({nodes,edges,meta})=>{
  allNodes=nodes;allEdges=edges;
  allNodes.forEach(n => nodeMap[n.id] = n);
  initSimulationState(allNodes,allEdges);
  document.getElementById('hf').textContent=meta.filename||'?';
  document.getElementById('ho').textContent=meta.n_ops;
  document.getElementById('hp').textContent=meta.n_params;
  document.getElementById('hd').textContent=meta.d_model;
  document.getElementById('hq').textContent=meta.quant;
  if(meta.mep_bytes>0){document.getElementById('hm').style.display='';document.getElementById('hmv').textContent=meta.mep_bytes+'B / '+meta.mep_consts+' consts';}
  buildGraph(allNodes,allEdges,true);
});
</script>
</body>
</html>"""


def create_app(nac_path: str) -> Flask:
    app = Flask(__name__)
    app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

    # Parse once at startup
    parser = NACGraphParser()
    try:
        parser.load(nac_path)
        graph = parser.build_graph()
        graph['meta']['filename'] = os.path.basename(nac_path)
    except Exception as e:
        print(f"ERROR parsing {nac_path}: {e}")
        import traceback; traceback.print_exc()
        graph = {'nodes': [], 'edges': [], 'meta': {'error': str(e), 'filename': nac_path}}

    n = len(graph['nodes'])
    e = len(graph['edges'])
    print(f"Graph built: {n} nodes, {e} edges.")

    html = HTML_PAGE.replace('{filename}', os.path.basename(nac_path))

    @app.route('/')
    def index():
        return html, 200, {'Content-Type': 'text/html; charset=utf-8'}

    @app.route('/api/graph')
    def api_graph():
        return jsonify(graph)

    return app


def main():
    ap = argparse.ArgumentParser(description='NAC computation graph visualizer')
    ap.add_argument('nac_file', help='Path to .nac file')
    ap.add_argument('--port', type=int, default=5050)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()

    if not os.path.isfile(args.nac_file):
        print(f"File not found: {args.nac_file}")
        sys.exit(1)

    app = create_app(args.nac_file)
    url = f"http://{args.host}:{args.port}"

    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    print(f"\n  NAC Graph Viewer")
    print(f"  Open: {url}")
    print(f"  File: {os.path.abspath(args.nac_file)}")
    print(f"  Press Ctrl+C to stop.\n")

    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)


if __name__ == '__main__':
    main()
