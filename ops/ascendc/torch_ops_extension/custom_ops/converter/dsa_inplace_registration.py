# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details.

import importlib
import inspect
import operator


_SERVE_ARGS = (
    "plan",
    "full_kv_cache",
    "full_k_rope",
    "pool_kv_cache",
    "pool_k_rope",
    "selection_kv_cache",
    "selection_k_rope",
)
_SERVE_MUTABLE_ARGS = ("selection_kv_cache", "selection_k_rope")
_INSTALL_ARGS = (
    "install_records",
    "selection_kv_cache",
    "selection_k_rope",
    "selection_kv_block_table",
    "pool_kv_cache",
    "pool_k_rope",
    "pool_ids",
    "id_to_slot",
    "lru_counter",
)
_INSTALL_MUTABLE_ARGS = (
    "pool_kv_cache",
    "pool_k_rope",
    "pool_ids",
    "id_to_slot",
    "lru_counter",
)
_GRAPH_PASS_MODULES = (
    "npugraph_ex._acl_concrete_graph.graph_pass",
    "torch_npu.dynamo.npugraph_ex._acl_concrete_graph.graph_pass",
    "torchair._acl_concrete_graph.graph_pass",
    "torch_npu.dynamo.torchair._acl_concrete_graph.graph_pass",
)


def _target_name(node):
    return str(getattr(node, "target", ""))


def _is_auto_functionalized(node):
    return "auto_functionalized" in _target_name(node)


def _effective_target_name(node):
    name = _target_name(node)
    args = getattr(node, "args", ())
    if _is_auto_functionalized(node) and args:
        return f"{name} {args[0]}"
    return name


def _is_dsa_node(node, op_name):
    return f"dsa_{op_name}" in _effective_target_name(node)


def _event_tag(node):
    args = getattr(node, "args", ())
    return args[0] if args and isinstance(args[0], str) else None


def _scope_label(node):
    args = getattr(node, "args", ())
    if len(args) < 2 or not isinstance(args[0], (list, tuple)) or not isinstance(args[1], (list, tuple)):
        return None
    try:
        return args[1][args[0].index("_user_stream_label")]
    except (ValueError, IndexError):
        return None


def _recorded_stream(node):
    args = getattr(node, "args", ())
    return args[1] if len(args) > 1 and isinstance(args[1], str) else None


def _stream_matches_scope(stream_tag, scope_label):
    if not stream_tag or not scope_label:
        return False
    return scope_label == stream_tag or scope_label.endswith(f"_{stream_tag}")


def _node_arg(node, index, name):
    kwargs = getattr(node, "kwargs", {})
    if name in kwargs:
        return kwargs[name]
    args = getattr(node, "args", ())
    return args[index] if index < len(args) else None


def _auto_base(node, name):
    kwargs = getattr(node, "kwargs", {})
    bases = kwargs.get("_all_bases")
    base_index = kwargs.get(f"_{name}_base_index")
    if not isinstance(base_index, int) or not isinstance(bases, (list, tuple)) or base_index >= len(bases):
        return None
    return bases[base_index]


def _tensor_inputs(node, arg_names, mutable_names):
    values = {}
    for index, name in enumerate(arg_names):
        value = _auto_base(node, name) if _is_auto_functionalized(node) and name in mutable_names else _node_arg(
            node, index, name
        )
        if value is None:
            return None
        values[name] = value
    return values


def _getitem_user(node, output_index):
    matches = []
    for user in getattr(node, "users", {}):
        args = getattr(user, "args", ())
        if (getattr(user, "target", None) is operator.getitem or "getitem" in _target_name(user)) and \
                len(args) > 1 and args[0] is node and args[1] == output_index:
            matches.append(user)
    return matches[0] if len(matches) == 1 else None


def _serve_outputs(node):
    values = _tensor_inputs(node, _SERVE_ARGS, _SERVE_MUTABLE_ARGS)
    if values is None:
        return None
    target_name = _effective_target_name(node)
    if not _is_auto_functionalized(node) and "dsa_serve.default" in target_name and "functional" not in target_name:
        outputs = {}
        for output_index, name in enumerate(_SERVE_MUTABLE_ARGS):
            candidates = [values[name]]
            transitional_output = _getitem_user(node, output_index)
            if transitional_output is not None:
                candidates.append(transitional_output)
            outputs[name] = tuple(candidates)
        return outputs
    outputs = {}
    for output_index, name in enumerate(_SERVE_MUTABLE_ARGS):
        if _is_auto_functionalized(node):
            base_index = getattr(node, "kwargs", {}).get(f"_{name}_base_index")
            if not isinstance(base_index, int):
                return None
            output_index = base_index + 1
        output = _getitem_user(node, output_index)
        if output is None:
            return None
        outputs[name] = (output,)
    return outputs


def _enclosing_scope(nodes, node_index):
    stack = []
    for index in range(node_index + 1):
        target = _target_name(nodes[index])
        if target.endswith("scope_enter.default"):
            stack.append(index)
        elif target.endswith("scope_exit.default"):
            if not stack:
                return None
            stack.pop()
    if not stack:
        return None
    enter_index = stack[-1]
    depth = 1
    for index in range(enter_index + 1, len(nodes)):
        target = _target_name(nodes[index])
        if target.endswith("scope_enter.default"):
            depth += 1
        elif target.endswith("scope_exit.default"):
            depth -= 1
            if depth == 0:
                return enter_index, index
    return None


def _previous_scope_exit(nodes, scope_enter):
    for index in range(scope_enter - 1, -1, -1):
        if _target_name(nodes[index]).endswith("scope_exit.default"):
            return index
    return -1


def _matching_event_pair(nodes, record_start, record_end, wait_start, wait_end, min_record_index=-1):
    records = {}
    for index in range(record_start, record_end):
        if index <= min_record_index or not _target_name(nodes[index]).endswith("tagged_event_record.default"):
            continue
        tag = _event_tag(nodes[index])
        if tag is not None:
            records.setdefault(tag, []).append(index)
    waits = {}
    for index in range(wait_start, wait_end):
        if not _target_name(nodes[index]).endswith("tagged_event_wait.default"):
            continue
        tag = _event_tag(nodes[index])
        if tag is not None:
            waits.setdefault(tag, []).append(index)
    matches = sorted(set(records) & set(waits))
    if len(matches) != 1:
        return None
    tag = matches[0]
    if len(records[tag]) != 1 or len(waits[tag]) != 1:
        return None
    return tag, records[tag][-1], waits[tag][0]


def _associated_copy_epilogue(user, base, allowed_installs):
    if not _target_name(user).endswith("copy_.default"):
        return False
    args = getattr(user, "args", ())
    if len(args) < 2 or args[0] is not base:
        return False
    source = _alias_origin(args[1])
    source_args = getattr(source, "args", ())
    return (
        (getattr(source, "target", None) is operator.getitem or "getitem" in _target_name(source))
        and source_args
        and source_args[0] in allowed_installs
    )


def _install_metadata_update(node):
    value = _node_arg(node, len(_INSTALL_ARGS) + 3, "metadata_update")
    return value if isinstance(value, int) else None


def _install_output(node, name):
    output_index = _INSTALL_MUTABLE_ARGS.index(name)
    if _is_auto_functionalized(node):
        base_index = getattr(node, "kwargs", {}).get(f"_{name}_base_index")
        if not isinstance(base_index, int):
            return None
        output_index = base_index + 1
    return _getitem_user(node, output_index)


def _metadata_install_chain(start, nodes):
    """Follow pass-through metadata to the unique owner update."""
    chain = [start]
    current = start
    while _install_metadata_update(current) == 0:
        outputs = {name: _install_output(current, name) for name in _INSTALL_MUTABLE_ARGS[2:]}
        if any(output is None for output in outputs.values()):
            return None
        current_index = nodes.index(current)
        candidates = []
        for candidate in nodes[current_index + 1:]:
            if not _is_dsa_node(candidate, "install"):
                continue
            values = _tensor_inputs(candidate, _INSTALL_ARGS, _INSTALL_MUTABLE_ARGS)
            if values is not None and all(values[name] is outputs[name] for name in outputs):
                candidates.append(candidate)
        if len(candidates) != 1 or candidates[0] in chain:
            return None
        current = candidates[0]
        chain.append(current)
    return set(chain) if _install_metadata_update(current) == 1 else None


def _is_aliasing_view(node):
    target = _target_name(node)
    return any(
        marker in target
        for marker in (
            "alias.default",
            "as_strided.default",
            "detach.default",
            "expand.default",
            "flatten.using_ints",
            "narrow.default",
            "permute.default",
            "reshape.default",
            "select.int",
            "slice.Tensor",
            "squeeze.",
            "t.default",
            "transpose.int",
            "unsqueeze.default",
            "view.default",
        )
    )


def _alias_origin(node):
    visited = set()
    while node not in visited and _is_aliasing_view(node):
        visited.add(node)
        args = getattr(node, "args", ())
        if not args or not hasattr(args[0], "target"):
            break
        node = args[0]
    return node


def _has_conflicting_resident_consumer(
    base, nodes, hazard_start, scope_enter, scope_exit, closure_wait, allowed_copy_installs, registration_nodes
):
    node_index = {node: index for index, node in enumerate(nodes)}
    pending = list(getattr(base, "users", {}))
    visited = set()
    while pending:
        user = pending.pop()
        if user in visited:
            continue
        visited.add(user)
        index = node_index.get(user)
        if index is None or index >= closure_wait:
            continue
        if _is_aliasing_view(user):
            pending.extend(getattr(user, "users", {}))
            continue
        if index <= hazard_start or user in registration_nodes or scope_enter < index < scope_exit:
            continue
        if _associated_copy_epilogue(user, base, allowed_copy_installs):
            continue
        return True
    return False


def _dsa_route_has_explicit_stream_order(node, mutated_arg=None, mutated_args=None):
    """Validate the exact DSA Serve/Install side-stream dataflow and lifetime topology."""
    if not (_is_dsa_node(node, "serve") or _is_dsa_node(node, "install")):
        return False

    nodes = list(node.graph.nodes)
    node_index = nodes.index(node)
    install = node if _is_dsa_node(node, "install") else None
    install_inputs = _tensor_inputs(install, _INSTALL_ARGS, _INSTALL_MUTABLE_ARGS) if install is not None else None

    if install is None:
        serve_outputs = _serve_outputs(node)
        if serve_outputs is None:
            return False
        candidates = []
        for candidate in nodes[node_index + 1:]:
            if not _is_dsa_node(candidate, "install"):
                continue
            candidate_inputs = _tensor_inputs(candidate, _INSTALL_ARGS, _INSTALL_MUTABLE_ARGS)
            if candidate_inputs is not None and all(
                candidate_inputs[name] in serve_outputs[name] for name in _SERVE_MUTABLE_ARGS
            ):
                candidates.append((candidate, candidate_inputs))
        if len(candidates) != 1:
            return False
        install, install_inputs = candidates[0]
    if install_inputs is None:
        return False

    install_index = nodes.index(install)
    scope = _enclosing_scope(nodes, install_index)
    if scope is None:
        return False
    scope_enter, scope_exit = scope
    side_stream_label = _scope_label(nodes[scope_enter])
    if side_stream_label is None:
        return False

    matching_serves = []
    for index in range(install_index):
        candidate = nodes[index]
        if not _is_dsa_node(candidate, "serve"):
            continue
        outputs = _serve_outputs(candidate)
        if outputs is not None and all(install_inputs[name] in outputs[name] for name in _SERVE_MUTABLE_ARGS):
            matching_serves.append((index, candidate))
    if len(matching_serves) != 1:
        return False
    serve_index, serve = matching_serves[0]
    if node is not install and node is not serve:
        return False

    if mutated_arg is not None:
        own_inputs = _tensor_inputs(
            node,
            _SERVE_ARGS if _is_dsa_node(node, "serve") else _INSTALL_ARGS,
            _SERVE_MUTABLE_ARGS if _is_dsa_node(node, "serve") else _INSTALL_MUTABLE_ARGS,
        )
        own_mutable_names = _SERVE_MUTABLE_ARGS if _is_dsa_node(node, "serve") else _INSTALL_MUTABLE_ARGS
        if own_inputs is None or mutated_arg not in [own_inputs[name] for name in own_mutable_names]:
            return False
    if mutated_args is not None:
        own_inputs = _tensor_inputs(
            node,
            _SERVE_ARGS if _is_dsa_node(node, "serve") else _INSTALL_ARGS,
            _SERVE_MUTABLE_ARGS if _is_dsa_node(node, "serve") else _INSTALL_MUTABLE_ARGS,
        )
        own_names = _SERVE_MUTABLE_ARGS if _is_dsa_node(node, "serve") else _INSTALL_MUTABLE_ARGS
        if own_inputs is None or tuple(own_inputs[name] for name in own_names) != tuple(mutated_args):
            return False

    registration_start = _previous_scope_exit(nodes, scope_enter) + 1
    registration_nodes = []
    registered = {}
    for index in range(registration_start, scope_enter):
        candidate = nodes[index]
        if not _target_name(candidate).endswith("record_tagged_stream.default"):
            continue
        args = getattr(candidate, "args", ())
        if len(args) < 2 or not _stream_matches_scope(_recorded_stream(candidate), side_stream_label):
            continue
        registration_nodes.append(candidate)
        registered.setdefault(args[0], []).append(index)
    node_positions = {candidate: index for index, candidate in enumerate(nodes)}
    required_tensors = set()
    for tensor in install_inputs.values():
        producer_index = node_positions.get(tensor)
        if producer_index is None or producer_index >= install_index:
            return False
        lifetime_root = _alias_origin(tensor)
        root_index = node_positions.get(lifetime_root)
        if root_index is None or root_index >= install_index:
            return False
        if root_index < scope_enter:
            required_tensors.add(lifetime_root)
    if not required_tensors.issubset(registered):
        return False

    last_registration = max(index for tensor in required_tensors for index in registered[tensor])
    producer_pair = _matching_event_pair(
        nodes,
        registration_start,
        scope_enter,
        scope_enter + 1,
        install_index,
        min_record_index=max(serve_index, last_registration - 1),
    )
    if producer_pair is None:
        return False
    _, producer_record, _ = producer_pair

    installs_in_scope = {
        candidate for candidate in nodes[scope_enter + 1:scope_exit] if _is_dsa_node(candidate, "install")
    }
    metadata_chain = _metadata_install_chain(install, nodes)
    if metadata_chain is None:
        return False
    allowed_copy_installs = installs_in_scope | metadata_chain
    last_install_index = max(nodes.index(candidate) for candidate in installs_in_scope)
    closure_pair = _matching_event_pair(
        nodes,
        last_install_index + 1,
        scope_exit,
        scope_exit + 1,
        len(nodes),
    )
    if closure_pair is None:
        return False
    _, _, closure_wait = closure_pair

    for name in _INSTALL_MUTABLE_ARGS:
        resident_root = _alias_origin(install_inputs[name])
        if _has_conflicting_resident_consumer(
            resident_root,
            nodes,
            producer_record,
            scope_enter,
            scope_exit,
            closure_wait,
            allowed_copy_installs,
            set(registration_nodes),
        ):
            return False
    return True


def _make_reinplace_check(graph_pass, mutated_indices):
    def _check(node):
        if _is_dsa_node(node, "serve"):
            arg_names, mutable_names = _SERVE_ARGS, _SERVE_MUTABLE_ARGS
        elif _is_dsa_node(node, "install"):
            arg_names, mutable_names = _INSTALL_ARGS, _INSTALL_MUTABLE_ARGS
        else:
            return graph_pass.check_multi_stream_for_multi_reinplace(node)
        expected_indices = tuple(arg_names.index(name) for name in mutable_names)
        inputs = _tensor_inputs(node, arg_names, mutable_names)
        if tuple(mutated_indices) == expected_indices and inputs is not None:
            mutated_args = tuple(inputs[name] for name in mutable_names)
            if _dsa_route_has_explicit_stream_order(node, mutated_args=mutated_args):
                return True
        return graph_pass.check_multi_stream_for_multi_reinplace(node)

    return _check


def _register_auto_functionalized_check(graph_pass, inplace_op, mutated_indices):
    state_name = "_dsa_auto_reinplace_state_v2"
    state = getattr(graph_pass, state_name, None)
    if state is None:
        original_check = graph_pass.check_multi_stream_for_auto_functionalize
        state = {"ops": {}, "original_check": original_check}

        def _check(node, mutated_arg):
            args = getattr(node, "args", ())
            mutable_op = args[0] if args else None
            expected_indices = state["ops"].get(mutable_op)
            if expected_indices is not None:
                if _dsa_route_has_explicit_stream_order(node, mutated_arg=mutated_arg):
                    return True
            return state["original_check"](node, mutated_arg)

        graph_pass.check_multi_stream_for_auto_functionalize = _check
        setattr(graph_pass, state_name, state)
    state["ops"][inplace_op] = tuple(mutated_indices)


def _require_signature(function, expected_names, module_name):
    try:
        names = tuple(inspect.signature(function).parameters)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"cannot inspect {module_name}.{function.__name__}: {error}") from error
    if names[:len(expected_names)] != expected_names:
        raise RuntimeError(
            f"unsupported TorchAir API {module_name}.{function.__name__}{inspect.signature(function)}; "
            f"expected leading parameters {expected_names}"
        )


def _validate_graph_pass(graph_pass):
    module_name = graph_pass.__name__
    required = (
        "check_multi_stream_for_auto_functionalize",
        "check_multi_stream_for_multi_reinplace",
        "inplaceable_npu_ops",
        "InplaceableNpuOp",
    )
    missing = [name for name in required if not hasattr(graph_pass, name)]
    if missing:
        raise RuntimeError(f"unsupported TorchAir graph-pass module {module_name}; missing {missing}")
    if not isinstance(graph_pass.inplaceable_npu_ops, dict):
        raise RuntimeError(f"unsupported TorchAir graph-pass registry type in {module_name}")
    _require_signature(graph_pass.check_multi_stream_for_auto_functionalize, ("node", "mutated_arg"), module_name)
    _require_signature(graph_pass.check_multi_stream_for_multi_reinplace, ("node",), module_name)
    _require_signature(graph_pass.InplaceableNpuOp, ("inplace_op", "mutated_arg", "extra_check"), module_name)


def register_dsa_inplace_pair(functional_op, inplace_op, mutated_args):
    """Register a DSA pair and fail loudly if no compatible TorchAir pass exists."""
    op_name = str(inplace_op)
    if "dsa_serve" in op_name:
        expected_mutated_args = tuple(_SERVE_ARGS.index(name) for name in _SERVE_MUTABLE_ARGS)
    elif "dsa_install" in op_name:
        expected_mutated_args = tuple(_INSTALL_ARGS.index(name) for name in _INSTALL_MUTABLE_ARGS)
    else:
        raise ValueError(f"unsupported DSA inplace op: {inplace_op}")
    if tuple(mutated_args) != expected_mutated_args:
        raise ValueError(
            f"invalid mutated args for {inplace_op}: {tuple(mutated_args)}; expected {expected_mutated_args}"
        )
    registered_modules = set()
    import_errors = []
    for module_name in _GRAPH_PASS_MODULES:
        try:
            graph_pass = importlib.import_module(module_name)
        except ImportError as error:
            import_errors.append(f"{module_name}: {error}")
            continue
        if id(graph_pass) in registered_modules:
            continue
        _validate_graph_pass(graph_pass)
        registered_modules.add(id(graph_pass))
        _register_auto_functionalized_check(graph_pass, inplace_op, mutated_args)
        graph_pass.inplaceable_npu_ops[functional_op] = graph_pass.InplaceableNpuOp(
            inplace_op=inplace_op,
            mutated_arg=list(mutated_args),
            extra_check=_make_reinplace_check(graph_pass, tuple(mutated_args)),
        )
    if not registered_modules:
        details = "; ".join(import_errors) or "no module candidates"
        raise RuntimeError(f"no compatible TorchAir graph-pass module for DSA reinplace registration: {details}")
    return len(registered_modules)
