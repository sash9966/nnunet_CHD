#!/usr/bin/env python3
"""Process-local adapter for SeqSeg 2.x: cap detected bifurcation generations at two.

Does NOT modify the installed package. Fail closed if the inspected tracing implementation
does not have the expected scheduling structure. Both continuation and side daughters inherit
the incremented generation; a third split stops the path (no retries beyond that split).
This limits detected tracing generations, not anatomical truth or incidental branches within a
local prediction crop. Record that distinction when comparing annotation extents.
"""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import textwrap


def bound_next(points, radii, angles, step, max_depth, number_chances, events):
    depth = int(step.get('_refinement_depth', 0))
    split = len(radii) > 1
    stop = split and depth >= max_depth
    if split:
        point = step['point']
        events.append({'point': point.tolist() if hasattr(point, 'tolist') else list(point),
                       'parent_depth': depth, 'outlets': len(radii), 'stopped': stop})
    if stop:
        step['chances'] = number_chances  # use normal branch completion/flush, never radius retries
        return points[:0], radii[:0], angles[:0], depth
    return points, radii, angles, depth + int(split)


def tag_step(step, depth):
    step['_refinement_depth'] = depth
    return step


def instrument(source):
    tree = ast.parse(textwrap.dedent(source))
    class Cap(ast.NodeTransformer):
        points_calls = 0
        step_calls = 0

        def visit_Assign(self, node):
            node = self.generic_visit(node)
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == 'get_next_points':
                if ast.dump(node.targets[0], include_attributes=False) != ast.dump(ast.parse('point_tree, radius_tree, angle_change = value').body[0].targets[0], include_attributes=False):
                    raise RuntimeError('SeqSeg next-point assignment changed')
                self.points_calls += 1
                extra = ast.parse('point_tree, radius_tree, angle_change, _refinement_depth = _refinement_bound_next(point_tree, radius_tree, angle_change, step_seg, _refinement_max_depth, number_chances, _refinement_events)').body[0]
                return [node, ast.copy_location(extra, node)]
            return node

        def visit_Call(self, node):
            node = self.generic_visit(node)
            if isinstance(node.func, ast.Name) and node.func.id == 'create_step_dict':
                self.step_calls += 1
                return ast.copy_location(ast.Call(func=ast.Name(id='_refinement_tag_step', ctx=ast.Load()),
                                                  args=[node, ast.Name(id='_refinement_depth', ctx=ast.Load())], keywords=[]), node)
            return node
    cap = Cap(); tree = cap.visit(tree)
    if cap.points_calls != 1 or cap.step_calls != 2:
        raise RuntimeError('Unsupported SeqSeg tracer: expected one next-point call and two daughter-step calls')
    return ast.fix_missing_locations(tree)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--max-bifurcations', type=int, choices=[0, 1, 2], default=2)
    args, argv = parser.parse_known_args()
    if argv and argv[0] == '--':
        argv = argv[1:]
    import SimpleITK as sitk
    from seqseg.modules import tracing
    from seqseg.pipeline import classic
    from seqseg import cli
    out = Path(args.audit_dir); out.mkdir(parents=True, exist_ok=False)
    source = inspect.getsource(tracing.trace_centerline)
    events = []
    namespace = dict(tracing.__dict__, _refinement_bound_next=bound_next,
                     _refinement_tag_step=tag_step, _refinement_max_depth=args.max_bifurcations,
                     _refinement_events=events)
    exec(compile(instrument(source), '<bounded_seqseg>', 'exec'), namespace)
    tracing.trace_centerline = namespace['trace_centerline']
    # Retain the exact probability assembly before classic.py thresholds/filters it.
    original_trace = classic.trace_centerline_from_context
    def audited_trace(ctx):
        result = original_trace(ctx)
        sitk.WriteImage(result.assembly.assembly, str(out/'probability_before_filter.mha'), True)
        depths = [int(s.get('_refinement_depth', 0)) for s in result.vessel_tree.steps]
        if max(depths, default=0) > args.max_bifurcations:
            raise RuntimeError('Depth enforcement failed')
        (out/'depth.json').write_text(json.dumps({'max_bifurcations': args.max_bifurcations,
             'max_recorded_depth': max(depths, default=0), 'events': events,
             'tracer_sha256': hashlib.sha256(source.encode()).hexdigest()}, indent=2))
        return result
    classic.trace_centerline_from_context = audited_trace
    original_keep = classic.sf.keep_component_seeds
    def audited_keep(image, seeds, *a, **kw):
        sitk.WriteImage(image, str(out/'binary_before_component_filter.mha'), True)
        result = original_keep(image, seeds, *a, **kw)
        sitk.WriteImage(result, str(out/'binary_after_component_filter.mha'), True)
        return result
    classic.sf.keep_component_seeds = audited_keep
    cli.dispatch(argv)
    if not (out/'depth.json').is_file() or not (out/'binary_after_component_filter.mha').is_file():
        raise RuntimeError('SeqSeg did not complete the instrumented stages')


if __name__ == '__main__':
    main()
