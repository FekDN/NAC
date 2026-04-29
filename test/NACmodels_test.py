# --- START OF FILE NACmodels_test.py ---
# Copyright (c) 2025-2026 Dmitry Feklin (FeklinDN@gmail.com) GNU General Public License v3.0

import os
import sys
import argparse
import traceback
import numpy as np

def _parse_kv(token: str):
    if '=' not in token: raise argparse.ArgumentTypeError(f"Expected KEY=VALUE, got: {token!r}")
    k_str, v_str = token.split('=', 1)
    def _coerce(s):
        try: return int(s)
        except ValueError: pass
        try: return float(s)
        except ValueError: pass
        if ',' in s:
            try: return [int(x) for x in s.split(',')]
            except ValueError: pass
        return s
    return _coerce(k_str), _coerce(v_str)

def run_nac_model(nac_path: str, exec_mode: str = 'infer_train', cli_args: list = None, patches: list = None, rewrites: list = None):
    from NAC_run import NacRuntime
    from MEP_compiler import MEPPatcher

    if not os.path.isfile(nac_path): raise FileNotFoundError(f"NAC file not found: '{nac_path}'")

    for old_val, new_val in (rewrites or []):
        ok = MEPPatcher.rewrite_constant_in_nac(nac_path, old_val, new_val)
        if ok: print(f"[Patch] Permanently rewrote constant {old_val!r} → {new_val!r} in '{nac_path}'")
        else: print(f"[Patch] WARNING: constant {old_val!r} not found in '{nac_path}'")

    probe = NacRuntime(nac_path)
    bytecode, constants = probe.get_mep_plan()
    if bytecode is None:
        raise RuntimeError(f"'{nac_path}' has no MEP execution plan (empty ORCH section).")

    for old_val, new_val in (patches or []):
        offset = MEPPatcher.find_src_constant_offset(bytecode, constants, old_val)
        if offset is not None:
            bytecode, constants = MEPPatcher.patch_src_constant_value(bytecode, constants, offset, new_val)
            print(f"[Patch] In-memory: constant {old_val!r} → {new_val!r}")
        else: print(f"[Patch] WARNING: constant {old_val!r} not found in bytecode")

    mode_note = {'infer': 'inference only', 'train': 'training only', 'infer_train': 'inference + training'}.get(exec_mode, exec_mode)
    print(f"--- MEP: {len(bytecode)} bytes, {len(constants)} constants, mode={exec_mode} ({mode_note}) ---")

    from MEP_interpreter import MEPInterpreter
    interp = MEPInterpreter(bytecode, constants, pre_answers=cli_args, exec_mode=exec_mode)
    interp.resources[0] = probe
    return interp.run()

def main():
    parser = argparse.ArgumentParser(
        description="Universal NAC Model Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  infer        — Only inference runs (skip training)
  train        — Only training runs (skip inference)
  infer_train  — Inference followed by training (default)

Examples:
  python NACmodels_test.py resnet18.nac
  python NACmodels_test.py resnet18.nac hen.jpg --mode train 42
  python NACmodels_test.py resnet18.nac hen.jpg --mode infer_train 42 --patch 0.001=0.0005
        """)

    parser.add_argument('nac', help='Path to .nac file')
    parser.add_argument('--mode', choices=['infer', 'train', 'infer_train'], default='infer_train', help='Execution mode')
    parser.add_argument('--patch', action='append', default=[], metavar='KEY=VALUE', help='Temporary in-memory constant override')
    parser.add_argument('--rewrite', action='append', default=[], metavar='KEY=VALUE', help='Permanent constant rewrite in .nac')
    parser.add_argument('args', nargs='*', help='CLI pre-answers for src_user_prompt')

    if hasattr(parser, "parse_intermixed_args"): ns = parser.parse_intermixed_args()
    else: ns = parser.parse_args()

    patches = [_parse_kv(t) for t in ns.patch]
    rewrites = [_parse_kv(t) for t in ns.rewrite]
    cli_args = ns.args or None

    try:
        result = run_nac_model(ns.nac, exec_mode=ns.mode, cli_args=cli_args, patches=patches, rewrites=rewrites)
        if result is not None: print(f"\n--- Result: {result} ---")
    except Exception as e:
        print(f"\n!!!!!! ERROR: {e}"); traceback.print_exc(); sys.exit(1)

if __name__ == '__main__': main()

# --- END OF FILE NACmodels_test.py ---