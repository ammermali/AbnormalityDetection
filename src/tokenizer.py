import ast
import pandas as pd

# Variabile globale usata dalla tua logica ricorsiva
verifica = 0


def setVerifica(v):
    global verifica
    verifica = v


def _maybe_parse_literal(x):
    if isinstance(x, str):
        s = x.strip()
        if not s: return x
        try:
            return ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return x
    return x


def _as_list(x):
    if x is None: return []
    if isinstance(x, float) and pd.isna(x): return []
    if isinstance(x, list): return x
    return [x]


def tokenize_calls(cell_value):
    out = []
    if isinstance(cell_value, str):
        s = cell_value.strip()
        if s == "" or s == "[]": return out
        try:
            calls_list = ast.literal_eval(s)
        except:
            calls_list = []
    else:
        calls_list = cell_value

    if not isinstance(calls_list, list): return out

    for call in calls_list:
        if not isinstance(call, dict): continue
        out.append("[CALLSTART]")

        for k, v in call.items():
            if k == "calls": continue
            if k == "inputs":
                out.append("[INsSTART]")
                inputs_list = v
                if isinstance(v, str):
                    try:
                        inputs_list = ast.literal_eval(v)
                    except:
                        inputs_list = []

                if isinstance(inputs_list, dict): inputs_list = [inputs_list]
                if not isinstance(inputs_list, list): inputs_list = []

                for d in inputs_list:
                    if isinstance(d, dict):
                        for kk, vv in d.items():
                            out.append(str(kk));
                            out.append(str(vv))
                out.append("[INsEND]")
            else:
                out.append(str(k));
                out.append(str(v))

        if "calls" in call and call["calls"]:
            out.append("[CALLS_CHILD_START]")
            out.extend(tokenize_calls(call["calls"]))
            out.append("CALLS_CHILD_END")

        out.append("[CALL_END]")
    return out


def tokenizer(dataFrame):
    global verifica
    tokenizerOutput = []

    for index, row_series in dataFrame.iterrows():
        if verifica == 0: tokenizerOutput.append("[START]")

        for column_name, cell_value in row_series.items():
            if column_name == "inputs":
                tokenizerOutput.append("[INsSTART]")
                inputs_val = _maybe_parse_literal(cell_value)
                for d in _as_list(inputs_val):
                    d = _maybe_parse_literal(d)
                    if isinstance(d, dict):
                        for k, v in d.items(): tokenizerOutput.append(str(k)); tokenizerOutput.append(str(v))
                    else:
                        tokenizerOutput.append(str(d))
                tokenizerOutput.append("[INsEND]")

            elif column_name == "timestamp":
                ts = _maybe_parse_literal(cell_value)
                tokenizerOutput.append("timestamp")
                if isinstance(ts, dict) and len(ts) > 0:
                    tokenizerOutput.append(str(next(iter(ts.values()))))
                else:
                    tokenizerOutput.append(str(ts))

            elif column_name == "internalTxs":
                tokenizerOutput.append("internalTxs");
                tokenizerOutput.append("[INXsSTART]")
                internal_val = _maybe_parse_literal(cell_value)
                internal_list = _as_list(internal_val)
                setVerifica(1)
                try:
                    tokenizerOutput.extend(tokenizer(pd.DataFrame(internal_list)))
                finally:
                    setVerifica(0)
                tokenizerOutput.append("[INXsEND]")

            elif column_name == "calls":
                tokenizerOutput.append("[CALLS_START]")
                tokenizerOutput.extend(tokenize_calls(cell_value))
                tokenizerOutput.append("[CALLS_END]")

            elif column_name == "events":
                tokenizerOutput.append("events");
                tokenizerOutput.append("[EVsSTART]")
                events_val = _maybe_parse_literal(cell_value)
                for ev in _as_list(events_val):
                    ev = _maybe_parse_literal(ev)
                    if not isinstance(ev, dict): tokenizerOutput.append(str(ev)); continue
                    for k, v in ev.items():
                        if k == "eventValues":
                            tokenizerOutput.append("eventValues")
                            ev_vals = _maybe_parse_literal(v)
                            if isinstance(ev_vals, dict):
                                for k2, v2 in ev_vals.items(): tokenizerOutput.append(str(k2)); tokenizerOutput.append(
                                    str(v2))
                            else:
                                tokenizerOutput.append(str(ev_vals))
                        else:
                            tokenizerOutput.append(str(k)); tokenizerOutput.append(str(v))
                tokenizerOutput.append("[EVsEND]")
            else:
                tokenizerOutput.append(str(column_name));
                tokenizerOutput.append(str(cell_value))

        if verifica == 0: tokenizerOutput.append("[END]")
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
            if len(stack) > 1: stack.pop()
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