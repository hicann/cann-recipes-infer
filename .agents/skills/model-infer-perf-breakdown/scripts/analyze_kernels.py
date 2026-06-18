#!/usr/bin/env python3
"""
分析 kernel_details.csv，输出结构化 JSON 统计信息

用法:
  python analyze_kernels.py [-f FILE] [-s STEP] [-o OUTPUT] [-d DETAILS] [-m MD]
  
选项:
  -f, --file FILE    指定 CSV 文件路径 (默认: kernel_details.csv)
  -s, --step STEP    指定要输出详情的 step ID (默认: 自动选择第一个 step)
  -o, --output FILE  输出 operators JSON 文件路径 (默认: kernels.json)
  -d, --details FILE 输出详细 operators JSON 文件路径 (包含 CSV 全部字段)
  -m, --markdown FILE 输出统计摘要 Markdown 文件路径 (不指定则不生成)
  -h, --help         显示帮助信息
"""
import csv
import json
import sys
import argparse
import os
from collections import defaultdict

REQUIRED_COLUMNS = ['Step Id', 'Name', 'Duration(us)', 'Start Time(us)', 'Stream ID']

IGNORED_COLUMNS = {'Step Id'}

# AI_CPU rows are host-side glue (data prep / scalar ops). Dropped from
# raw_ops at generation time by default; pass `--keep-ai-cpu` to retain
# (host-bound analysis only). Sample-mode pipeline assumes drop-by-default.
AI_CPU_ACCELERATOR_CORES = {'AI_CPU', 'AICPU'}
COMMUNICATION_CORE = 'COMMUNICATION'
COMMUNICATION_FRAGMENT_NAME = 'AivKernel'
MISSING_ID_VALUES = {'', 'N/A', 'NA', 'NONE', 'NULL'}

METRIC_COLUMN_PATTERNS = [
    'duration', 'time', 'cycles', 'ratio', 'utilization', 
    'fops', 'rate', 'miss', 'count', 'num', 'dim', 'id'
]

VALUE_COLUMN_PATTERNS = [
    'name', 'type', 'state', 'core', 'shapes', 'formats',
    'eligible', 'context'
]

FLOAT_SUFFIXES = ['(us)', '(%)']


def is_metric_column(col_name):
    col_lower = col_name.lower()
    for pattern in METRIC_COLUMN_PATTERNS:
        if pattern in col_lower:
            return True
    return False


def is_value_column(col_name):
    col_lower = col_name.lower()
    for pattern in VALUE_COLUMN_PATTERNS:
        if pattern in col_lower:
            return True
    return False


def parse_column_value(value, col_name, json_key=None):
    value = value.strip().rstrip('\t')
    if not value or value == 'N/A':
        return None
    
    col_lower = col_name.lower()
    
    for suffix in FLOAT_SUFFIXES:
        if col_lower.endswith(suffix.lower()) or suffix.lower() in col_lower:
            try:
                return float(value)
            except (ValueError, AttributeError):
                return value
    
    if is_metric_column(col_name):
        try:
            if '.' in value:
                return float(value)
            return int(value)
        except (ValueError, AttributeError):
            return value
    
    try:
        if '.' in value:
            fval = float(value)
            return fval if fval != int(fval) else int(fval)
        return int(value)
    except (ValueError, AttributeError):
        pass
    
    return value


def csv_col_name_to_json_key(col_name):
    key = col_name.lower()
    key = key.replace('(us)', '_us')
    key = key.replace('(%)', '_pct')
    key = key.replace(' ', '_')
    key = key.replace('(', '')
    key = key.replace(')', '')
    return key


class ValidationError(Exception):
    pass


class ConsistencyError(Exception):
    pass


def validate_file(file_path):
    if not os.path.exists(file_path):
        raise ValidationError(f'错误: 文件不存在: {file_path}')
    if not os.path.isfile(file_path):
        raise ValidationError(f'错误: 路径不是文件: {file_path}')
    if not os.access(file_path, os.R_OK):
        raise ValidationError(f'错误: 文件不可读: {file_path}')


def validate_csv_structure(fieldnames):
    missing = [col for col in REQUIRED_COLUMNS if col not in fieldnames]
    if missing:
        raise ValidationError(
            f'错误: CSV 文件缺少必需列\n'
            f'缺少的列: {", ".join(missing)}\n'
            f'当前的列: {", ".join(fieldnames)}'
        )


def get_safe_value(row, key, default=''):
    return row.get(key, default).strip().rstrip('\t')


def parse_float(value):
    try:
        return float(value.strip().rstrip('\t'))
    except (ValueError, AttributeError):
        return 0.0


def is_missing_id(value):
    return str(value or '').strip().upper() in MISSING_ID_VALUES


def is_hccl_type(value):
    lower = str(value or '').strip().lower()
    return lower.startswith('hcom') or 'hccl' in lower


def is_communication_summary_row(row):
    accel = get_safe_value(row, 'Accelerator Core').upper()
    if accel != COMMUNICATION_CORE:
        return False
    name = get_safe_value(row, 'Name')
    typ = get_safe_value(row, 'Type') if 'Type' in row else ''
    return (
        name != COMMUNICATION_FRAGMENT_NAME and
        is_missing_id(row.get('Stream ID')) and
        is_missing_id(row.get('Task ID')) and
        is_hccl_type(typ or name)
    )


def is_communication_task_fragment_row(row):
    accel = get_safe_value(row, 'Accelerator Core').upper()
    if accel != COMMUNICATION_CORE:
        return False
    name = get_safe_value(row, 'Name')
    typ = get_safe_value(row, 'Type') if 'Type' in row else ''
    return (
        name == COMMUNICATION_FRAGMENT_NAME and
        not (is_missing_id(row.get('Stream ID')) and is_missing_id(row.get('Task ID'))) and
        is_hccl_type(typ)
    )


def build_communication_summary_index(rows):
    summaries = defaultdict(list)
    for row in rows:
        if not is_communication_summary_row(row):
            continue
        step_id = get_safe_value(row, 'Step Id')
        typ = get_safe_value(row, 'Type') if 'Type' in row else get_safe_value(row, 'Name')
        start = parse_float(get_safe_value(row, 'Start Time(us)'))
        duration = parse_float(get_safe_value(row, 'Duration(us)'))
        summaries[(step_id, typ)].append((start, start + duration))
    return summaries


def has_matching_communication_summary(row, summaries, eps=1e-3):
    step_id = get_safe_value(row, 'Step Id')
    typ = get_safe_value(row, 'Type') if 'Type' in row else get_safe_value(row, 'Name')
    start = parse_float(get_safe_value(row, 'Start Time(us)'))
    for lo, hi in summaries.get((step_id, typ), []):
        if lo - eps <= start <= hi + eps:
            return True
    return False


def extract_kernel_type(name):
    if not name:
        return 'Unknown'
    
    if '/' in name:
        return name
    
    if '_' not in name:
        return name
    
    parts = name.split('_')
    
    while parts and parts[-1].isdigit():
        parts.pop()
    
    if not parts:
        return name
    
    last_part = parts[-1]
    if last_part and last_part[0].isupper() and not last_part.isdigit():
        return last_part
    
    return '_'.join([p for p in parts if p])


def parse_args():
    parser = argparse.ArgumentParser(
        description='分析 kernel_details.csv，输出结构化 JSON 统计信息',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '-f', '--file',
        default='kernel_details.csv',
        help='指定 CSV 文件路径 (默认: kernel_details.csv)'
    )
    parser.add_argument(
        '-s', '--step',
        type=int,
        default=None,
        help='指定要输出详情的 step ID (默认: 自动选择第一个 step)'
    )
    parser.add_argument(
        '-o', '--output',
        default='kernels.json',
        help='输出 operators JSON 文件路径 (默认: kernels.json)'
    )
    parser.add_argument(
        '-d', '--details',
        default=None,
        help='输出详细 operators JSON 文件路径 (包含 CSV 全部字段)'
    )
    parser.add_argument(
        '--keep-ai-cpu',
        action='store_true',
        help='保留 AI_CPU 算子（默认丢弃）。仅 host-bound 分析场景用；sample 模式严禁打开'
    )
    return parser.parse_args()


def _warmup_step_hint(steps_summary, selected_step_id, auto=False):
    """Print a stderr hint about which step looks like warmup / atypical.

    Heuristic is architecture-neutral: compare each step's kept-op
    total_duration_us against the median across steps. A step deviating
    > ~30% (typically the first step due to recompile / cold cache /
    extra collective sync) is flagged. Only a hint — never changes behavior
    or exits. Helps the agent avoid silently profiling a warmup step when
    step ids don't start at 0 (so the doc's `-s 2` default is meaningless).
    """
    rows = [s for s in (steps_summary or []) if s.get('total_duration_us') is not None]
    if len(rows) <= 1:
        return
    durs = sorted(s['total_duration_us'] for s in rows)
    n = len(durs)
    median = durs[n // 2] if n % 2 else (durs[n // 2 - 1] + durs[n // 2]) / 2
    if median <= 0:
        return
    flagged = [s for s in rows
               if abs(s['total_duration_us'] - median) / median > 0.30]
    listing = ", ".join(
        f"step {s['step_id']}={s['total_duration_us']:.0f}us"
        + ("*" if s in flagged else "")
        for s in rows
    )
    print(f"[step 选择] 可用 step (kept-op Σdur, * = 偏离中位>30%, 疑 warmup/异常): "
          f"{listing}", file=sys.stderr)
    sel = next((s for s in rows if str(s['step_id']) == str(selected_step_id)), None)
    if sel and sel in flagged:
        note = "自动选了第一个 step" if auto else "你指定的 step"
        print(f"  ⚠ {note} (step {selected_step_id}) Σdur 偏离中位 >30%，"
              f"可能是 warmup/异常步；建议换一个接近中位的稳定 step (-s <id>)。",
              file=sys.stderr)


def check_consistency(steps_summary):
    if len(steps_summary) <= 1:
        return
    
    first_types = None
    first_step = None
    
    for s in steps_summary:
        if first_types is None:
            first_types = s['kernel_types']
            first_step = s['step_id']
        elif s['kernel_types'] != first_types:
            diff_kernels = []
            all_keys = set(first_types.keys()) | set(s['kernel_types'].keys())
            for k in sorted(all_keys):
                v1 = first_types.get(k, 0)
                v2 = s['kernel_types'].get(k, 0)
                if v1 != v2:
                    diff_kernels.append(f'  {k}: Step {first_step}={v1}, Step {s["step_id"]}={v2}')
            
            raise ConsistencyError(
                f'错误: 各 Step 的 Kernel 分布不一致\n'
                f'差异:\n' + '\n'.join(diff_kernels[:10]) +
                (f'\n  ... 还有 {len(diff_kernels) - 10} 个差异' if len(diff_kernels) > 10 else '')
            )


def analyze_kernels(csv_file, detail_step_id=None, keep_ai_cpu=False):
    validate_file(csv_file)
    
    steps_data = defaultdict(lambda: {
        'kernels': [],
        'kernels_details': [],
        'total_duration': 0.0,
        'kernel_types': defaultdict(int),
        'all_columns': []
    })
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if reader.fieldnames is None:
                raise ValidationError('错误: CSV 文件为空或格式不正确')
            validate_csv_structure(reader.fieldnames)
            
            all_columns = [col for col in reader.fieldnames if col not in IGNORED_COLUMNS]
            csv_rows = list(reader)
            communication_summaries = build_communication_summary_index(csv_rows)
            
            has_accel_col = 'Accelerator Core' in reader.fieldnames
            row_count = 0
            ai_cpu_dropped = 0
            communication_task_fragments_dropped = 0
            communication_summary_rows_kept = 0
            communication_task_fragments_dropped_by_step = defaultdict(int)
            communication_summary_rows_kept_by_step = defaultdict(int)
            for row_num, row in enumerate(csv_rows, start=2):
                try:
                    step_id = get_safe_value(row, 'Step Id')
                    if not step_id:
                        continue

                    accel = get_safe_value(row, 'Accelerator Core') if has_accel_col else ''
                    if accel.upper() in AI_CPU_ACCELERATOR_CORES and not keep_ai_cpu:
                        ai_cpu_dropped += 1
                        continue
                    if (is_communication_task_fragment_row(row) and
                            has_matching_communication_summary(row, communication_summaries)):
                        communication_task_fragments_dropped += 1
                        communication_task_fragments_dropped_by_step[step_id] += 1
                        continue
                    if is_communication_summary_row(row):
                        communication_summary_rows_kept += 1
                        communication_summary_rows_kept_by_step[step_id] += 1

                    name = get_safe_value(row, 'Name')
                    csv_type = get_safe_value(row, 'Type') if 'Type' in row else ''
                    # Prefer CSV's canonical `Type` column over name-derived heuristic.
                    # The Type column already strips per-instance suffixes
                    # (`HcomAllGather_612_0_1` → `HcomAllGather`) and resolves cases
                    # where Name carries no signal (`AivKernel` → `HcomAllGather`).
                    # Fall back to name-derived type only when Type column is missing.
                    kernel_type = csv_type or extract_kernel_type(name)
                    duration = parse_float(get_safe_value(row, 'Duration(us)'))

                    op_info = {
                        'index': len(steps_data[step_id]['kernels']),
                        'org_index': row_num - 2,
                        'original_name': name,
                        'normalized_name': kernel_type,
                        'duration_us': duration,
                        'start_time_us': parse_float(get_safe_value(row, 'Start Time(us)')),
                        'stream_id': get_safe_value(row, 'Stream ID'),
                        'task_type': csv_type,
                        'accelerator_core': accel,
                        'input_shapes': get_safe_value(row, 'Input Shapes').strip('"') if 'Input Shapes' in row else '',
                        'output_shapes': get_safe_value(row, 'Output Shapes').strip('"') if 'Output Shapes' in row else ''
                    }

                    detail_info = {'index': op_info['index'], 'org_index': op_info['org_index']}
                    for col in all_columns:
                        if col in IGNORED_COLUMNS:
                            continue
                        json_key = csv_col_name_to_json_key(col)
                        raw_value = get_safe_value(row, col)
                        if raw_value:
                            detail_info[json_key] = parse_column_value(raw_value, col, json_key)
                            if json_key in ('start_time_us', 'duration_us'):
                                detail_info[f'{json_key}_raw'] = raw_value
                    for key in ('input_shapes', 'output_shapes'):
                        if key in detail_info and isinstance(detail_info[key], str):
                            detail_info[key] = detail_info[key].strip('"')
                    
                    steps_data[step_id]['kernels'].append(op_info)
                    steps_data[step_id]['kernels_details'].append(detail_info)
                    steps_data[step_id]['total_duration'] += duration
                    steps_data[step_id]['kernel_types'][kernel_type] += 1
                    steps_data[step_id]['all_columns'] = all_columns
                    row_count += 1
                    
                except Exception as e:
                    sys.stderr.write(f'警告: 第 {row_num} 行数据解析失败: {e}\n')
            
            if row_count == 0:
                raise ValidationError('错误: CSV 文件没有有效数据行')

            if ai_cpu_dropped > 0:
                sys.stderr.write(
                    f'AI_CPU 算子 {ai_cpu_dropped} 行已丢弃\n'
                )
            if communication_task_fragments_dropped > 0:
                sys.stderr.write(
                    f'通信 task fragment {communication_task_fragments_dropped} 行已丢弃，'
                    f'保留 collective summary {communication_summary_rows_kept} 行\n'
                )
                
    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(f'错误: 解析 CSV 文件失败: {e}')
    
    sorted_steps = sorted(steps_data.keys(), key=lambda x: int(x) if x.isdigit() else x)
    
    result = {
        'step_count': len(steps_data),
        'steps_summary': [],
        'csv_columns': all_columns,
        'communication_filter_mode': 'drop_task_fragments_keep_collective_summary',
        'communication_task_fragments_dropped': communication_task_fragments_dropped,
        'communication_summary_rows_kept': communication_summary_rows_kept,
        'communication_task_fragments_dropped_by_step': dict(communication_task_fragments_dropped_by_step),
        'communication_summary_rows_kept_by_step': dict(communication_summary_rows_kept_by_step),
    }
    
    for step_id in sorted_steps:
        data = steps_data[step_id]
        result['steps_summary'].append({
            'step_id': step_id,
            'total_duration_us': round(data['total_duration'], 1),
            'kernel_count': len(data['kernels']),
            'kernel_types_count': len(data['kernel_types']),
            'kernel_types': dict(sorted(data['kernel_types'].items(), 
                                        key=lambda x: -x[1]))
        })
    
    check_consistency(result['steps_summary'])

    if detail_step_id is None:
        detail_step_id = sorted_steps[0] if sorted_steps else None

    _warmup_step_hint(result['steps_summary'], detail_step_id,
                      auto=(detail_step_id == (sorted_steps[0] if sorted_steps else None)))
    
    if detail_step_id is not None:
        str_step_id = str(detail_step_id)
        if str_step_id in steps_data:
            data = steps_data[str_step_id]
            selected_comm_filter = {
                'mode': 'drop_task_fragments_keep_collective_summary',
                'task_fragments_dropped': communication_task_fragments_dropped_by_step.get(str_step_id, 0),
                'summary_rows_kept': communication_summary_rows_kept_by_step.get(str_step_id, 0),
            }
            result['selected_step_operators'] = {
                'step_id': str_step_id,
                'total_duration_us': round(data['total_duration'], 1),
                'kernel_count': len(data['kernels']),
                'kernel_types_count': len(data['kernel_types']),
                'kernel_types': dict(sorted(data['kernel_types'].items(), 
                                            key=lambda x: -x[1])),
                'communication_filter': selected_comm_filter,
                'operators': data['kernels']
            }
            result['selected_step_operators_details'] = {
                'step_id': str_step_id,
                'total_duration_us': round(data['total_duration'], 1),
                'kernel_count': len(data['kernels']),
                'kernel_types_count': len(data['kernel_types']),
                'kernel_types': dict(sorted(data['kernel_types'].items(), 
                                            key=lambda x: -x[1])),
                'communication_filter': selected_comm_filter,
                'csv_columns': data['all_columns'],
                'operators': data['kernels_details']
            }
        else:
            available = list(sorted_steps)
            hint = ''
            if len(available) == 1:
                hint = f'\n提示: 仅一个 step 可用，重跑加 `-s {available[0]}`（或省略 -s 用默认）。'
            elif available:
                hint = f'\n提示: 选一个重跑，如 `-s {available[0]}`。'
            raise ValidationError(
                f'错误: Step {detail_step_id} 不存在\n'
                f'可用的 Step: {", ".join(available)}'
                f'{hint}'
            )
    
    return result, sorted_steps


def main():
    args = parse_args()

    try:
        result, sorted_steps = analyze_kernels(args.file, args.step, keep_ai_cpu=args.keep_ai_cpu)
        
        if 'selected_step_operators' in result:
            operators_json = result['selected_step_operators']
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(operators_json, f, indent=2, ensure_ascii=False)
            print(f'Operators JSON 已保存到: {args.output}')
        
        if args.details and 'selected_step_operators_details' in result:
            details_json = result['selected_step_operators_details']
            with open(args.details, 'w', encoding='utf-8') as f:
                json.dump(details_json, f, indent=2, ensure_ascii=False)
            print(f'Details JSON 已保存到: {args.details}')

    except ValidationError as e:
        sys.stderr.write(str(e) + '\n')
        sys.exit(1)
    except ConsistencyError as e:
        sys.stderr.write(str(e) + '\n')
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f'错误: 未知错误: {e}\n')
        sys.exit(1)


if __name__ == '__main__':
    main()
