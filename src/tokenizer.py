import ast
import re
import pandas as pd

# flag per tenere traccia se stiamo attualmente tokenizzando una struttura annidata (internalTxs) per evitare di aggiungere più [START]/[END] attorno a ogni transazione interna
is_nested = 0

def setIsNested(is_nested_value):
    global is_nested
    is_nested = is_nested_value


_ADDR_0X = re.compile(r"^0x[0-9a-fA-F]{40}$")
_ADDR_NO0X = re.compile(r"^[0-9a-fA-F]{40}$")
_HEX_0X = re.compile(r"^0x[0-9a-fA-F]+$")
_HEX32_0X = re.compile(r"^0x[0-9a-fA-F]{64}$")
_BIGINT_DIGITS_CUTOFF = 60
_SMALL_INT_BUCKETS = [
    (0, "[INT_0]"),
    (1, "[INT_1]"),
    (10, "[INT_LT10]"),
    (100, "[INT_LT100]"),
    (1000, "[INT_LT1K]"),
    (10_000, "[INT_LT10K]"),
    (1_000_000, "[INT_LT1M]"),
    (1_000_000_000, "[INT_LT1B]"),
]

def _safe_int_token(n: int) -> str:
    absn = abs(n)
    for bound, tok in _SMALL_INT_BUCKETS:
        if absn == bound:
            return tok
        if absn < bound and bound != 0:
            return tok

    # for medium integers we keep keep exact value. we can remove this block if we want to always bucket
    # per integers di media grandezza teniamo il valore esatto, per evitare di bucketizzare troppo e perdere segnalazioni di anomalie basate su valori specifici 
    bits = absn.bit_length()
    approx_digits = int(bits * 0.30103) + 1
    if approx_digits > _BIGINT_DIGITS_CUTOFF:
        return "[BIGINT]"
    return str(n)

def normalize_token(value, field_name=None):
    """
    sostituisce valori ad alta cardinalità con token speciali per ridurre la dimensione del vocabolario e bucketizza interi molto grandi per evitare errori di Python int->str per limiti di cifre
    """

    if isinstance(value, int):
        return _safe_int_token(value)

    if isinstance(value, float):
        if pd.isna(value):
            return "[NA]"
        return str(value)

    if value is None:
        return "[NONE]"

    if isinstance(value, str):
        s = value.strip()
        if not s:
            return value

        if field_name == "transactionHash" and _HEX32_0X.match(s):
            return "[TXHASH]"

        if _ADDR_0X.match(s) or _ADDR_NO0X.match(s):
            return "[ADDR]"

        if _HEX32_0X.match(s):
            return "[HEX32]"

        if _HEX_0X.match(s):
            return "[HEX]"

        return value

    return f"[{type(value).__name__.upper()}]"


def _maybe_parse_literal(x):
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return x
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return x
    return x


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, float) and pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    return [x]


def tokenize_calls(cell_value):
    out = []
    if isinstance(cell_value, str):
        s = cell_value.strip()
        if s == "" or s == "[]":
            return out
        try:
            calls_list = ast.literal_eval(s)
        except Exception:
            calls_list = []
    else:
        calls_list = cell_value

    if not isinstance(calls_list, list):
        return out

    for call in calls_list:
        if not isinstance(call, dict):
            continue
        out.append("[CALLSTART]")

        for k, v in call.items():
            if k == "calls":
                continue

            if k == "inputs":
                out.append("[INsSTART]")
                inputs_list = v
                if isinstance(v, str):
                    try:
                        inputs_list = ast.literal_eval(v)
                    except Exception:
                        inputs_list = []

                if isinstance(inputs_list, dict):
                    inputs_list = [inputs_list]
                if not isinstance(inputs_list, list):
                    inputs_list = []

                for d in inputs_list:
                    if isinstance(d, dict):
                        for kk, vv in d.items():
                            kk_s = str(kk)
                            out.append(kk_s)
                            out.append(str(normalize_token(vv, field_name=kk_s)))
                out.append("[INsEND]")

            else:
                kk_s = str(k)
                out.append(kk_s)
                out.append(str(normalize_token(v, field_name=kk_s)))

        if "calls" in call and call["calls"]:
            out.append("[CALLS_CHILD_START]")
            out.extend(tokenize_calls(call["calls"]))
            out.append("CALLS_CHILD_END")

        out.append("[CALL_END]")
    return out


def tokenizer(dataFrame):
    global is_nested
    tokenizerOutput = []

    for index, row_series in dataFrame.iterrows():
        if is_nested == 0:
            tokenizerOutput.append("[START]")

        for column_name, cell_value in row_series.items():
            col = str(column_name)

            if column_name == "inputs":
                tokenizerOutput.append("[INsSTART]")
                inputs_val = _maybe_parse_literal(cell_value)
                for d in _as_list(inputs_val):
                    d = _maybe_parse_literal(d)
                    if isinstance(d, dict):
                        for k, v in d.items():
                            kk = str(k)
                            tokenizerOutput.append(kk)
                            tokenizerOutput.append(str(normalize_token(v, field_name=kk)))
                    else:
                        tokenizerOutput.append(str(normalize_token(d, field_name=col)))
                tokenizerOutput.append("[INsEND]")

            elif column_name == "timestamp":
                ts = _maybe_parse_literal(cell_value)
                tokenizerOutput.append("timestamp")
                if isinstance(ts, dict) and len(ts) > 0:
                    tokenizerOutput.append(str(normalize_token(next(iter(ts.values())), field_name="timestamp")))
                else:
                    tokenizerOutput.append(str(normalize_token(ts, field_name="timestamp")))

            elif column_name == "internalTxs":
                tokenizerOutput.append("internalTxs")
                tokenizerOutput.append("[INXsSTART]")
                internal_val = _maybe_parse_literal(cell_value)
                internal_list = _as_list(internal_val)
                setIsNested(1)
                try:
                    tokenizerOutput.extend(tokenizer(pd.DataFrame(internal_list)))
                finally:
                    setIsNested(0)
                tokenizerOutput.append("[INXsEND]")

            elif column_name == "calls":
                tokenizerOutput.append("[CALLS_START]")
                tokenizerOutput.extend(tokenize_calls(cell_value))
                tokenizerOutput.append("[CALLS_END]")

            elif column_name == "events":
                tokenizerOutput.append("events")
                tokenizerOutput.append("[EVsSTART]")
                events_val = _maybe_parse_literal(cell_value)
                for ev in _as_list(events_val):
                    ev = _maybe_parse_literal(ev)
                    if not isinstance(ev, dict):
                        tokenizerOutput.append(str(normalize_token(ev, field_name="events")))
                        continue
                    for k, v in ev.items():
                        kk = str(k)
                        if k == "eventValues":
                            tokenizerOutput.append("eventValues")
                            ev_vals = _maybe_parse_literal(v)
                            if isinstance(ev_vals, dict):
                                for k2, v2 in ev_vals.items():
                                    k2s = str(k2)
                                    tokenizerOutput.append(k2s)
                                    tokenizerOutput.append(str(normalize_token(v2, field_name=k2s)))
                            else:
                                tokenizerOutput.append(str(normalize_token(ev_vals, field_name="eventValues")))
                        else:
                            tokenizerOutput.append(kk)
                            tokenizerOutput.append(str(normalize_token(v, field_name=kk)))
                tokenizerOutput.append("[EVsEND]")

            else:
                tokenizerOutput.append(col)
                tokenizerOutput.append(str(normalize_token(cell_value, field_name=col)))

        if is_nested == 0:
            tokenizerOutput.append("[END]")
    return tokenizerOutput


def flatten_tokens(x):
    flat = []
    for el in x:
        if isinstance(el, list):
            flat.extend(flatten_tokens(el))
        else:
            flat.append(str(el))
    return flat


def build_tree_from_output(output):
    tokens = flatten_tokens(output)
    tree = []
    stack = ["0"]
    current = "0"
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        tree.append(current)
        if i > 0 and tokens[i - 1] == "callId":
            new_id = tok
            stack.append(new_id)
            current = new_id
        if tok == "[CALL_END]":
            if len(stack) > 1:
                stack.pop()
            current = stack[-1]
        i += 1
    return tokens, tree


def build_context_from_tokens(tokens):
    context = []
    for i, tok in enumerate(tokens):
        if i > 0 and tokens[i - 1] == "to":
            context.append("TO")
        elif i > 0 and tokens[i - 1] == "from":
            context.append("FROM")
        else:
            context.append("NONE")
    return context
