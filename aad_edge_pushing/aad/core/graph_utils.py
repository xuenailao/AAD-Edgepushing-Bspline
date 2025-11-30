"""
计算图工具函数
用于打印和分析AAD计算图结构
"""

import numpy as np
from typing import Dict, List, Tuple
from collections import Counter


def print_graph_summary(tape, detailed: bool = False) -> Dict:
    """
    打印计算图摘要信息

    Args:
        tape: global_tape对象
        detailed: 是否打印详细节点信息

    Returns:
        包含统计信息的字典
    """
    if not tape.nodes:
        print("Empty computation graph")
        return {}

    n_nodes = len(tape.nodes)
    n_edges = sum(len(node.parents) for node in tape.nodes)

    # 统计入度
    fan_ins = [len(node.parents) for node in tape.nodes]
    max_fan_in = max(fan_ins) if fan_ins else 0
    avg_fan_in = np.mean(fan_ins) if fan_ins else 0

    # 统计出度（需要遍历所有节点）
    fan_outs = [0] * n_nodes
    for i, node in enumerate(tape.nodes):
        for parent, _ in node.parents:
            if hasattr(parent, '_tape_idx') and parent._tape_idx is not None:
                if parent._tape_idx < n_nodes:
                    fan_outs[parent._tape_idx] += 1

    max_fan_out = max(fan_outs) if fan_outs else 0
    avg_fan_out = np.mean(fan_outs) if fan_outs else 0

    # 统计操作类型
    op_types = [node.op_tag for node in tape.nodes]
    op_counter = Counter(op_types)

    # 打印摘要
    print("\n" + "="*70)
    print("COMPUTATION GRAPH SUMMARY")
    print("="*70)
    print(f"Total nodes:        {n_nodes:,}")
    print(f"Total edges:        {n_edges:,}")
    print(f"Max fan-in:         {max_fan_in}")
    print(f"Avg fan-in:         {avg_fan_in:.2f}")
    print(f"Max fan-out:        {max_fan_out}")
    print(f"Avg fan-out:        {avg_fan_out:.2f}")
    print()
    print("Operation breakdown:")
    for op_type, count in op_counter.most_common(10):
        pct = 100.0 * count / n_nodes
        print(f"  {op_type:12s}: {count:6,} ({pct:5.1f}%)")

    if detailed and n_nodes <= 100:
        print()
        print("="*70)
        print("DETAILED NODE LIST (first 100 nodes)")
        print("="*70)
        for i, node in enumerate(tape.nodes[:100]):
            parent_info = ", ".join([
                f"Node{parent._tape_idx if hasattr(parent, '_tape_idx') else '?'}"
                for parent, _ in node.parents
            ])
            print(f"Node {i:3d}: {node.op_tag:12s} <- [{parent_info}]")

    print("="*70 + "\n")

    return {
        'nodes': n_nodes,
        'edges': n_edges,
        'max_fan_in': max_fan_in,
        'avg_fan_in': avg_fan_in,
        'max_fan_out': max_fan_out,
        'avg_fan_out': avg_fan_out,
        'operations': dict(op_counter)
    }


def print_computation_graph(tape, max_nodes: int = 20) -> None:
    """
    打印计算图结构（代码输出格式）

    Args:
        tape: global_tape对象
        max_nodes: 最多打印多少个节点
    """
    print("\n" + "="*70)
    print("COMPUTATION GRAPH STRUCTURE")
    print("="*70)

    if not tape.nodes:
        print("Empty graph")
        return

    n_show = min(len(tape.nodes), max_nodes)

    for i, node in enumerate(tape.nodes[:n_show]):
        # 节点基本信息
        out_val = float(node.out) if hasattr(node.out, 'val') else node.out

        # 父节点信息
        if node.parents:
            parent_strs = []
            for parent, deriv in node.parents:
                if hasattr(parent, '_tape_idx') and parent._tape_idx is not None:
                    parent_strs.append(f"Node{parent._tape_idx}")
                else:
                    parent_strs.append("external")
            parent_info = ", ".join(parent_strs)
            print(f"Node {i:4d}: {node.op_tag:12s} ({out_val:10.6f}) <- [{parent_info}]")
        else:
            print(f"Node {i:4d}: {node.op_tag:12s} ({out_val:10.6f}) [leaf/input]")

    if len(tape.nodes) > max_nodes:
        print(f"... ({len(tape.nodes) - max_nodes} more nodes)")

    print("="*70 + "\n")


def get_graph_stats(tape) -> Dict:
    """
    获取计算图统计信息（不打印）

    Returns:
        统计信息字典
    """
    if not tape.nodes:
        return {
            'nodes': 0,
            'edges': 0,
            'max_fan_in': 0,
            'avg_fan_in': 0.0,
            'max_fan_out': 0,
            'avg_fan_out': 0.0,
            'operations': {}
        }

    n_nodes = len(tape.nodes)
    n_edges = sum(len(node.parents) for node in tape.nodes)

    # 统计入度
    fan_ins = [len(node.parents) for node in tape.nodes]
    max_fan_in = max(fan_ins)
    avg_fan_in = np.mean(fan_ins)

    # 统计出度
    fan_outs = [0] * n_nodes
    for node in tape.nodes:
        for parent, _ in node.parents:
            if hasattr(parent, '_tape_idx') and parent._tape_idx is not None:
                if parent._tape_idx < n_nodes:
                    fan_outs[parent._tape_idx] += 1

    max_fan_out = max(fan_outs) if fan_outs else 0
    avg_fan_out = np.mean(fan_outs) if fan_outs else 0.0

    # 统计操作类型
    op_types = [node.op_tag for node in tape.nodes]
    op_counter = Counter(op_types)

    return {
        'nodes': n_nodes,
        'edges': n_edges,
        'max_fan_in': max_fan_in,
        'avg_fan_in': avg_fan_in,
        'max_fan_out': max_fan_out,
        'avg_fan_out': avg_fan_out,
        'operations': dict(op_counter)
    }


def analyze_graph_complexity(tape) -> str:
    """
    分析计算图复杂度并返回文本报告

    Returns:
        复杂度分析的文本报告
    """
    stats = get_graph_stats(tape)

    if stats['nodes'] == 0:
        return "Empty computation graph"

    report = []
    report.append("Graph Complexity Analysis:")
    report.append(f"  Total operations: {stats['nodes']:,}")
    report.append(f"  Total connections: {stats['edges']:,}")
    report.append(f"  Average branching: {stats['avg_fan_out']:.2f}")

    # 复杂度评估
    if stats['nodes'] < 1000:
        complexity = "Low"
    elif stats['nodes'] < 10000:
        complexity = "Medium"
    else:
        complexity = "High"

    report.append(f"  Complexity level: {complexity}")

    # 最常见的操作
    if stats['operations']:
        top_ops = sorted(stats['operations'].items(), key=lambda x: x[1], reverse=True)[:3]
        report.append("  Top operations:")
        for op, count in top_ops:
            pct = 100.0 * count / stats['nodes']
            report.append(f"    - {op}: {pct:.1f}%")

    return "\n".join(report)
