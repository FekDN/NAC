# Copyright (c) 2025 Dmitry Feklin (FeklinDN@gmail.com) Apache License 2.0.
# https://github.com/FekDN/T-ISA-Tokenizer-Instruction-Set-Architecture/

import struct
import json
import regex as re
import unicodedata
import torch
from typing import List, Dict, Any, Tuple, Union, Literal
from dataclasses import dataclass, field
from transformers import AutoTokenizer

@dataclass
class Fragment:
    text: str
    is_protected: bool = False

# --- T-ISA PRIMITIVES v1.0 ---
class Primitives:

    @staticmethod
    def LOWERCASE(state, res, _): state['text'] = state['text'].lower()

    @staticmethod
    def UNICODE_NORM(state, res, args): state['text'] = unicodedata.normalize(args['form'], state['text'])

    @staticmethod
    def REPLACE(state, res, args): state['text'] = state['text'].replace(args['pattern'], args['val'])

    @staticmethod
    def FILTER_CATEGORY(state, res, args):
        cats = set(args['cats'])
        state['text'] = "".join(c for c in state['text'] if unicodedata.category(c) not in cats)

    @staticmethod
    def PREPEND(state, res, args):
        if not state['text'].startswith(args['val']): state['text'] = args['val'] + state['text']

    @staticmethod
    def PARTITION_RULES(state, res, args):
        rules = args.get('rules', [])
        for rule in rules:
            if rule.get('protected'):
                p = rule.get('pattern', '')
                if (p.startswith('<') or p.startswith('[')) and not p.startswith('(?: ?)'):
                    rule['pattern'] = f"(?: ?){p}"
        if not rules:
            state['fragments'] = [Fragment(state['text'])]; return
        parts = [f"(?P<r{i}>{r['pattern']})" for i, r in enumerate(rules)]
        combined = re.compile("|".join(parts))
        raw_text, fragments, last_pos = state['text'], [], 0
        for m in combined.finditer(raw_text):
            preceding_text = raw_text[last_pos:m.start()]
            if not m.lastgroup: continue
            rule = rules[int(m.lastgroup[1:])]
            trim_char = rule.get('trim_preceding_space')
            if trim_char and preceding_text.endswith(trim_char):
                preceding_text = preceding_text[:-len(trim_char)]
            if preceding_text:
                fragments.append(Fragment(preceding_text, is_protected=False))
            match_text = m.group()
            if rule.get('protected', False):
                if match_text.startswith(" ") and len(match_text) > 1:
                     match_text = match_text[1:]
            if rule.get('behavior') != 'REMOVE':
                fragments.append(Fragment(match_text, is_protected=rule.get('protected', False)))
            last_pos = m.end()
        if last_pos < len(raw_text):
            fragments.append(Fragment(raw_text[last_pos:], is_protected=False))
        state['fragments'] = fragments

    @staticmethod
    def BYTE_ENCODE(state, res, args):
        byte_map = res.get('byte_map')
        for frag in state['fragments']:
            if frag.is_protected: continue
            if 'pipeline' in args:
                for op in args['pipeline']:
                    if op['type'] == 'BYTE_ENCODE':
                        frag.text = "".join([byte_map[b] for b in frag.text.encode('utf-8')])
        return state

    @staticmethod
    def UNIGRAM_ENCODE(state, res, _):
        scores, vocab = res['unigram_scores'], res['vocab']
        final_ids, unk_id = [], vocab.get('<unk>', 2)
        for frag in state['fragments']:
            if frag.is_protected:
                final_ids.append(vocab.get(frag.text, unk_id)); continue
            text, n = frag.text, len(frag.text)
            if n == 0: continue
            dp, best_path = [-1e10] * (n + 1), [0] * (n + 1); dp[0] = 0
            for i in range(n):
                for j in range(i + 1, min(n + 1, i + 50)):
                    sub = text[i:j]
                    if sub in scores:
                        score = dp[i] + scores[sub]
                        if score > dp[j]: dp[j], best_path[j] = score, i
            if dp[n] == -1e10:
                for char in text: final_ids.append(vocab.get(char, unk_id))
            else:
                curr, path = n, []
                while curr > 0:
                    prev = best_path[curr]; path.append(vocab.get(text[prev:curr], unk_id)); curr = prev
                final_ids.extend(reversed(path))
        state['ids'] = final_ids

    @staticmethod
    def BPE_ENCODE(state, res, _):
        ranks, vocab = res['ranks'], res['vocab']
        final_ids, unk_id = [], vocab.get('<unk>', vocab.get('[UNK]', 0))
        for frag in state['fragments']:
            if frag.is_protected: 
                tid = vocab.get(frag.text)
                if tid is None: tid = vocab.get(frag.text.strip(), unk_id)
                final_ids.append(tid); continue
            word = tuple(frag.text)
            while len(word) > 1:
                pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
                best_pair = min(pairs, key=lambda p: ranks.get(p, float('inf')))
                if best_pair not in ranks: break
                new_word, i = [], 0
                while i < len(word):
                    if i < len(word)-1 and word[i:i+2] == best_pair:
                        new_word.append(best_pair[0]+best_pair[1]); i += 2
                    else: new_word.append(word[i]); i += 1
                word = tuple(new_word)
            final_ids.extend([vocab.get(t, unk_id) for t in word])
        state['ids'] = final_ids

    @staticmethod
    def WORDPIECE_ENCODE(state, res, args):
        vocab, marker = res['vocab'], args.get('marker', '##')
        unk_id, final_ids = vocab.get('[UNK]', 100), []
        for frag in state['fragments']:
            if frag.is_protected:
                final_ids.append(vocab.get(frag.text, unk_id)); continue
            text, current_ids_for_fragment, start_offset = frag.text, [], 0
            while start_offset < len(text):
                end_offset, found_match = len(text), False
                while start_offset < end_offset:
                    sub_token = text[start_offset:end_offset]
                    token_to_lookup = sub_token if start_offset == 0 else marker + sub_token
                    if token_to_lookup in vocab:
                        current_ids_for_fragment.append(vocab[token_to_lookup])
                        start_offset, found_match = end_offset, True; break
                    end_offset -= 1
                if not found_match:
                    current_ids_for_fragment.append(unk_id); start_offset += 1
            final_ids.extend(current_ids_for_fragment)
        state['ids'] = final_ids

    @staticmethod
    def COMPOSE(state, res, args):
        vocab, out_ids = res['vocab'], []
        for item in args['template']:
            if item[0] == 'FIXED':
                tid = item[1] if isinstance(item[1], int) else vocab.get(item[1], 2)
                out_ids.append(tid)
            elif item[0] == 'SLOT': out_ids.extend(state['ids'])
        state['ids'] = out_ids

class TISAVM:
    def __init__(self, resources):
        self.res = resources
        self.id_to_token = {}
        if self.res and 'vocab' in self.res: self.id_to_token = {v: k for k, v in self.res['vocab'].items()}
        self.dispatch = {
            0x01: Primitives.LOWERCASE,
            0x02: Primitives.UNICODE_NORM,
            0x03: Primitives.REPLACE, 
            0x04: Primitives.FILTER_CATEGORY,
            0x07: Primitives.PREPEND,
            0x10: Primitives.PARTITION_RULES,
            0x15: Primitives.BYTE_ENCODE,
            0x20: Primitives.BPE_ENCODE,
            0x21: Primitives.WORDPIECE_ENCODE, 
            0x22: Primitives.UNIGRAM_ENCODE,
            0x30: Primitives.COMPOSE
        }

    def run(self, manifest_data: Union[bytes, List], text: str):
        state = {'text': text, 'fragments': [], 'ids': []}
        commands = []
        if isinstance(manifest_data, bytes):
            if manifest_data[:4] != b"TISA": raise ValueError("Invalid Magic")
            offset = 5
            while offset < len(manifest_data):
                opcode = manifest_data[offset]; offset += 1
                p_len = struct.unpack("<I", manifest_data[offset:offset+4])[0]; offset += 4
                payload = json.loads(manifest_data[offset:offset+p_len].decode('utf-8')); offset += p_len
                commands.append((opcode, payload))
        else: commands = manifest_data
        for opcode, payload in commands:
            self.dispatch[opcode](state, self.res, payload)
        return state['ids']

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for i in ids:
            token = self.id_to_token.get(i, "")
            if skip_special_tokens and ((token.startswith('<') and token.endswith('>')) or (token.startswith('[') and token.endswith(']'))): continue
            tokens.append(token)
        model_type = "Unknown"
        is_sp_bpe = any('\u2581' in t for t in self.id_to_token.values())
        if is_sp_bpe:
            model_type = "Unigram"
        elif 'ranks' in self.res and self.res['ranks']: model_type = "BPE"
        elif 'unigram_scores' in self.res and self.res['unigram_scores']: model_type = "Unigram"
        elif any(t.startswith("##") for t in self.id_to_token.values()): model_type = "WordPiece"

        if model_type == "BPE":
            try:
                byte_decoder = {v: k for k, v in self.res['byte_map'].items()}
                full_bytes = bytearray()
                for token in tokens:
                    for char in token:
                        if char in byte_decoder: full_bytes.append(byte_decoder[char])
                return full_bytes.decode('utf-8', errors='replace')
            except Exception: return "".join(tokens).replace('Ġ', ' ').strip()
        if model_type == "WordPiece":
            output_string = " ".join(tokens).replace(" ##", "")
            output_string = re.sub(r'\s+([.,!?"\'])', r'\1', output_string).replace(" ' ", "'")
            return output_string
        if model_type == "Unigram": return "".join(tokens).replace('\u2581', ' ').strip()
        return " ".join(tokens)

class TISACompiler:
    @staticmethod
    def _process_normalizer(norm_config: Dict, instructions: List):
        if not norm_config:
            return
        norm_type = norm_config.get('type')
        if norm_type == 'Sequence':
            for n in norm_config.get('normalizers', []):
                TISACompiler._process_normalizer(n, instructions)
        elif norm_type == 'Lowercase':
            instructions.append([0x01, {}])
        elif norm_type == 'BertNormalizer':
            if norm_config.get('clean_text', True):
                instructions.extend([
                    [0x03, {'pattern': '\t', 'val': ' '}],
                    [0x03, {'pattern': '\r', 'val': ' '}],
                    [0x03, {'pattern': '\n', 'val': ' '}]
                ])
                instructions.append([0x04, {'cats': ['Cc', 'Cf']}])
            if norm_config.get('lowercase', True):
                instructions.append([0x01, {}])
                if norm_config.get('strip_accents', False) is not False:
                    instructions.extend([
                        [0x02, {'form': 'NFD'}],
                        [0x04, {'cats': ['Mn']}]
                    ])

    @staticmethod
    def _process_pre_tokenizer(pre_tok_config: Dict, rules: List, frag_pipeline: List, instructions: List) -> bool:
        if not pre_tok_config:
            return False
        pre_tok_type = pre_tok_config.get('type')
        has_space_handler = False
        if pre_tok_type == 'Sequence':
            for p in pre_tok_config.get('pre_tokenizers', []):
                if TISACompiler._process_pre_tokenizer(p, rules, frag_pipeline, instructions):
                    has_space_handler = True
        elif pre_tok_type == 'Metaspace':
            replacement = pre_tok_config.get('replacement', '\u2581')
            instructions.append([0x03, {'pattern': ' ', 'val': replacement}])
            if pre_tok_config.get('add_prefix_space', True):
                instructions.append([0x07, {'val': replacement}])
            has_space_handler = True
        elif pre_tok_type == 'ByteLevel':
            if pre_tok_config.get('add_prefix_space'):
                instructions.append([0x07, {'val': ' '}])
            pattern = pre_tok_config.get('pattern')
            if pattern:
                rules.append({'pattern': pattern, 'regex': True})
            else:
                rules.append({'pattern': r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+", 'regex': True})
            frag_pipeline.append({'type': 'BYTE_ENCODE'})
            has_space_handler = True
        elif pre_tok_type == 'Split' and pre_tok_config.get('invert'):
            pattern_regex = pre_tok_config.get('pattern', {}).get('Regex')
            if pattern_regex:
                rules.append({'pattern': pattern_regex, 'regex': True})
        elif pre_tok_type in ('BertPreTokenizer', 'WhitespaceSplit'):
            if not has_space_handler:
                rules.insert(0, {'pattern': r'\s+', 'behavior': 'REMOVE', 'regex': True})
            if pre_tok_type == 'BertPreTokenizer':
                rules.extend([
                    {'pattern': r'\p{P}', 'behavior': 'ISOLATE', 'regex': True},
                    {'pattern': r'[\u4E00-\u9FFF\u3040-\u309F\u30A0-\u30FF\uAC00-\uD7AF]', 'behavior': 'ISOLATE', 'regex': True}
                ])
        return has_space_handler

    @staticmethod
    def compile_and_calibrate(tokenizer, probe_text: str) -> bytes:
        state = json.loads(tokenizer.backend_tokenizer.to_str())
        model_config = state.get('model', {})
        pre_tokenizer_config = state.get('pre_tokenizer')
        model_type = model_config.get('type')
        instructions = []
        TISACompiler._process_normalizer(state.get('normalizer'), instructions)
        is_sentencepiece_style = (model_type == 'Unigram') or \
                                 (model_type == 'BPE' and not pre_tokenizer_config)
        if is_sentencepiece_style:
            replacement_char = '\u2581'
            instructions.append([0x03, {'pattern': ' ', 'val': replacement_char}])
            add_prefix = pre_tokenizer_config.get('add_prefix_space', True) if pre_tokenizer_config else True
            if add_prefix:
                instructions.append([0x07, {'val': replacement_char}])
        special_tokens = {
            tok for tok in [
                getattr(tokenizer, key, None) for key in 
                ['unk_token', 'sep_token', 'pad_token', 'cls_token', 'mask_token', 'bos_token', 'eos_token']
            ] if tok and isinstance(tok, str)
        }
        for at in state.get('added_tokens', []):
            if at.get('special') or not at.get('normalized'):
                special_tokens.add(at['content'])
        sorted_special_tokens = sorted(list(special_tokens), key=len, reverse=True)
        rules = []
        for token in sorted_special_tokens:
            escaped = re.escape(token)
            rule = {'pattern': escaped, 'protected': True}
            if not is_sentencepiece_style and state.get('pre_tokenizer', {}).get('type') != 'ByteLevel':
                 rule['pattern'] = f"(?: ?){escaped}"
            elif is_sentencepiece_style:
                 rule['trim_preceding_space'] = '\u2581'
            rules.append(rule)
        frag_pipeline = []
        TISACompiler._process_pre_tokenizer(pre_tokenizer_config, rules, frag_pipeline, instructions)
        instructions.append([0x10, {'rules': rules}])
        if frag_pipeline:
            instructions.append([0x15, {'pipeline': frag_pipeline}])
        if model_type == 'BPE':
            if not any(op.get('type') == 'BYTE_ENCODE' for op in frag_pipeline) and \
               state.get('decoder', {}).get('type') == 'ByteLevel':
                instructions.append([0x15, {'pipeline': [{'type': 'BYTE_ENCODE'}]}])
            instructions.append([0x20, {}])
        elif model_type == 'WordPiece':
            instructions.append([0x21, {'marker': model_config.get('continuing_subword_prefix', '##')}])
        elif model_type == 'Unigram':
            instructions.append([0x22, {}])
        post_config = state.get('post_processor')
        template = [('SLOT',)]
        if post_config:
            post_type = post_config['type']
            if post_type == 'TemplateProcessing':
                template_pieces = post_config.get('single') or post_config.get('piece')
                template = [
                    ('FIXED', x['SpecialToken']['id']) if 'SpecialToken' in x else ('SLOT',)
                    for x in template_pieces
                ]
            elif post_type == 'RobertaProcessing':
                template = [('FIXED', post_config['cls'][1]), ('SLOT',), ('FIXED', post_config['sep'][1])]
        else:
            if getattr(tokenizer, 'add_bos_token', False) and hasattr(tokenizer, 'bos_token_id'):
                template.insert(0, ('FIXED', tokenizer.bos_token_id))
            if getattr(tokenizer, 'add_eos_token', False) and hasattr(tokenizer, 'eos_token_id'):
                template.append(('FIXED', tokenizer.eos_token_id))
        if template != [('SLOT',)]:
            instructions.append([0x30, {'template': template}])
        buf = bytearray(b"TISA\x01") # Magic + version
        for op, payload in instructions:
            p_bytes = json.dumps(payload, ensure_ascii=False).encode('utf-8')
            buf.append(op)
            buf.extend(struct.pack("<I", len(p_bytes)))
            buf.extend(p_bytes)
        return bytes(buf)

    @staticmethod
    def _prepare_resources(tok, state):
        res = {'vocab': tok.get_vocab()}
        ranks = {}; scores = {}
        for i, m in enumerate(state['model'].get('merges', [])): ranks[tuple(m.split()) if isinstance(m, str) else tuple(m)] = i
        if state['model']['type'] == 'Unigram':
            for t, s in state['model']['vocab']: scores[t] = s
        res['ranks'] = ranks; res['unigram_scores'] = scores
        # Range 1: 33-126  (from '!' to '~')
        # Range 2: 161-172 (from '¡' to '¬')
        # Range 3: 174-255 (from '®' to 'ÿ')
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        res['byte_map'] = dict(zip(bs + [b for b in range(256) if b not in bs], [chr(c) for c in bs + [256 + i for i in range(256-len(bs))]]))
        return res

# --- Opcode Name Dictionary for Disassembler ---
OPCODE_NAMES = {
    0x01: "LOWERCASE",
    0x02: "UNICODE_NORM",
    0x03: "REPLACE",
    0x04: "FILTER_CATEGORY",
    0x07: "PREPEND",
    0x10: "PARTITION_RULES",
    0x15: "BYTE_ENCODE", # Specific FRAG_TRANSFORM
    0x20: "BPE_ENCODE",
    0x21: "WORDPIECE_ENCODE",
    0x22: "UNIGRAM_ENCODE",
    0x30: "COMPOSE"
}

# --- Disassembler Function---
def disassemble_TISA_manifest(manifest_bytes: bytes) -> Dict[str, Any]:
    if manifest_bytes[:4] != b"TISA":
        raise ValueError("Invalid Magic. Expected b'TISA'.")
    disassembled = {}
    offset = 5
    while offset < len(manifest_bytes):
        opcode = manifest_bytes[offset]; offset += 1
        p_len = struct.unpack("<I", manifest_bytes[offset:offset+4])[0]; offset += 4
        payload_bytes = manifest_bytes[offset:offset+p_len]
        payload = json.loads(payload_bytes.decode('utf-8')); offset += p_len
        op_name = OPCODE_NAMES.get(opcode, f"UNKNOWN_0x{opcode:02x}")
        op_key = f"0x{opcode:02x} ({op_name})"
        if op_key not in disassembled:
            disassembled[op_key] = []
        disassembled[op_key].append(payload)
    for key, value in disassembled.items():
        if len(value) == 1:
            disassembled[key] = value[0]
    return disassembled
