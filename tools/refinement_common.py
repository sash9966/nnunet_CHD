"""Geometry, prompt generation and conservative arbitration for the four-arm study.

All coordinates passed to nnInteractive are array indices. SeqSeg seeds are LPS mm.
No model imports here: these invariants can be tested without a GPU.
"""
import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

ARMS = ('baseline', 'chambers', 'seqseg', 'combined')
NAMES = {1: 'LV', 2: 'RV', 3: 'LA', 4: 'RA', 5: 'Myo', 6: 'Aorta', 7: 'Pulmonary'}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def load_label(path, reference=None):
    im = nib.load(str(path))
    if reference is not None:
        if im.shape != reference.shape or not np.allclose(im.affine, reference.affine, atol=1e-4, rtol=0):
            raise ValueError('Physical grid mismatch: ' + str(path))
    a = np.asarray(im.dataobj)
    if a.ndim != 3 or not np.isfinite(a).all() or not np.equal(a, np.rint(a)).all():
        raise ValueError('Expected finite integer 3D label: ' + str(path))
    if a.min() < 0 or a.max() > 7:
        raise ValueError('Expected CHD labels 0..7: ' + str(path))
    return im, a.astype(np.uint8)


def save_label(path, a, reference):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    hdr = reference.header.copy()
    hdr.set_data_dtype(np.uint8)
    im = nib.Nifti1Image(a.astype(np.uint8), reference.affine, hdr)
    im.set_qform(reference.affine, code=1)
    im.set_sform(reference.affine, code=1)
    tmp = path.with_name(path.name.replace('.nii.gz', '.tmp.nii.gz'))
    nib.save(im, str(tmp))
    tmp.replace(path)


def largest_component(mask):
    cc, n = ndi.label(mask, structure=np.ones((3, 3, 3), dtype=bool))
    if not n:
        return mask.copy()
    sizes = np.bincount(cc.ravel()); sizes[0] = 0
    return cc == int(sizes.argmax())


def chamber_prompts(seed, sid, spacing, margin_mm=1.0):
    """Three separated axial core lassos + one near-interface interior negative per opponent.

    Uses array axis 2, not an assumed anatomical orientation. Exact raster masks are returned.
    The seed remains uncertain supervision; erosion reduces, but cannot remove, its errors.
    """
    mask = seed == sid
    if not mask.any():
        return [], []
    dt = ndi.distance_transform_edt(mask, sampling=spacing)
    core = mask & (dt >= margin_mm)
    if not core.any():
        core = dt == dt.max()
    occupied = np.flatnonzero(core.any(axis=(0, 1)))
    planes = sorted(set(int(occupied[round(q * (len(occupied) - 1))]) for q in (.25, .5, .75)))
    lassos = []
    for z in planes:
        coords = np.argwhere(core[:, :, z]); lo = coords.min(0); hi = coords.max(0) + 1
        bbox = [[int(lo[0]), int(hi[0])], [int(lo[1]), int(hi[1])], [z, z + 1]]
        crop = core[lo[0]:hi[0], lo[1]:hi[1], z:z+1].astype(np.uint8)
        lassos.append((crop, bbox))
    distance = ndi.distance_transform_edt(~mask, sampling=spacing)
    negatives = []
    for other in range(1, 8):
        if other == sid or not np.any(seed == other):
            continue
        interior = ndi.distance_transform_edt(seed == other, sampling=spacing)
        eligible = (seed == other) & (interior >= margin_mm)
        if not eligible.any():
            eligible = interior == interior.max()
        cost = np.where(eligible, distance, np.inf)
        p = np.unravel_index(int(cost.argmin()), cost.shape)
        negatives.append({'point': list(map(int, p)), 'opponent': other})
    return lassos, negatives


def directed_seeds(mask, affine, spacing, inset_mm=5.0):
    """Two core seeds directed toward opposite ends of one LCC diameter path.

    Radius is measured at the inset point, not at a truncated endpoint. Both current and
    previous points are actual in-mask centerline voxels. Short/degenerate masks fail explicitly.
    """
    from label_to_prompts import centerline
    mask = largest_component(mask)
    path, _ = centerline(mask, spacing)
    if len(path) < 3:
        raise ValueError('Vessel seed has no usable directed centerline')
    path = np.asarray(path, dtype=int)
    if tuple(path[-1]) < tuple(path[0]):
        path = path[::-1]
    dt = ndi.distance_transform_edt(mask, sampling=spacing)
    path_radii = dt[tuple(path.T)]
    core_radius = .5 * float(path_radii[1:-1].max())
    rows, details = [], []
    for ordered in (path, path[::-1]):
        xyz = nib.affines.apply_affine(affine, ordered)
        arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
        target = min(inset_mm, arc[-1] / 3.)
        k = min(max(1, int(np.searchsorted(arc, target))), len(ordered) - 2)
        initial_k = k
        # A fixed 5-mm inset can still sit on a one-voxel spur. Walk inward to
        # a substantive part of this same path, without inventing a radius or
        # jumping to another component. This relative threshold is a heuristic.
        eligible = np.arange(k, len(ordered)-1)
        radii = dt[tuple(ordered[eligible].T)]
        adequate = eligible[radii >= core_radius]
        k = int(adequate[0] if len(adequate) else eligible[int(radii.argmax())])
        current, previous = ordered[k], ordered[k+1]
        radius = float(dt[tuple(current)])
        if radius <= 0 or not mask[tuple(previous)]:
            raise ValueError('Invalid interior seed')
        ras = nib.affines.apply_affine(affine, [previous, current])
        lps = ras * [-1., -1., 1.]
        rows.append([lps[0].tolist(), lps[1].tolist(), radius])
        details.append({'previous_voxel': previous.tolist(), 'current_voxel': current.tolist(),
                        'radius_mm': radius, 'inset_mm': float(arc[k]),
                        'initial_inset_voxel': ordered[initial_k].tolist(),
                        'initial_radius_mm': float(dt[tuple(ordered[initial_k])]),
                        'core_radius_target_mm': core_radius,
                        'core_target_met': bool(radius >= core_radius),
                        'placement': 'first_inward_path_point_at_half_maximum_path_radius'})
    return rows, details


def guarded_candidate(seed, sid, candidate, spacing, min_retention=.7, max_growth=4.,
                      cleanup=True):
    """Restrict claims to own seed/background. Reject loss/growth after restriction.

    These are QC heuristics, not correctness guarantees. No crossing original class boundaries;
    this explicit conservative policy trades correction of class swaps for an interpretable study.
    """
    original = seed == sid
    raw = candidate.astype(bool)
    if raw.shape != seed.shape:
        raise ValueError('Candidate shape mismatch')
    allowed = raw & ((seed == 0) | original)
    blocked = int(np.count_nonzero(raw & ~((seed == 0) | original)))
    if cleanup:
        cc, n = ndi.label(allowed, structure=np.ones((3, 3, 3), dtype=bool))
        overlaps = np.bincount(cc[original], minlength=n+1); overlaps[0] = 0
        allowed = (cc == int(overlaps.argmax())) if overlaps.any() else np.zeros_like(original)
    count = int(original.sum())
    retained = int(np.count_nonzero(allowed & original)) / max(count, 1)
    growth = int(allowed.sum()) / max(count, 1)
    reasons = []
    if not count:
        reasons.append('absent_seed')
    if not allowed.any():
        reasons.append('empty')
    if retained < min_retention:
        reasons.append('seed_loss')
    if growth > max_growth:
        reasons.append('growth')
    return (original.copy() if reasons else allowed), {
        'raw_voxels': int(raw.sum()), 'blocked_by_seed_class_voxels': blocked,
        'accepted_proposal_voxels': int(allowed.sum()), 'seed_retention': retained,
        'growth_ratio': growth, 'fallback': reasons,
        'seed_ml': count * float(np.prod(spacing)) / 1000.}


def compose(seed, candidates, spacing, min_retention=.7, max_growth=4., cleanup=True, vessel_max_growth=20.):
    """Simultaneous proposals: own seed territory + background, ambiguous additions abstain.

    Non-target labels are preserved exactly. A fallback restores its seed exactly. Background
    claimed by multiple classes stays background; dictionary/order cannot change the result.
    """
    out = seed.copy(); claims = np.zeros(seed.shape, np.uint8)
    claimant = np.zeros(seed.shape, np.uint8); records = {}
    for sid in sorted(candidates):
        accepted, rec = guarded_candidate(seed, sid, candidates[sid], spacing,
                                          min_retention, vessel_max_growth if sid in (6, 7) else max_growth, cleanup)
        records[str(sid)] = rec
        out[seed == sid] = 0
        out[accepted & (seed == sid)] = sid
        new = accepted & (seed == 0)
        claims[new] += 1; claimant[new] = sid
    unique = (seed == 0) & (claims == 1)
    out[unique] = claimant[unique]
    conflicts = (seed == 0) & (claims > 1)
    for sid in candidates:
        rec = records[str(sid)]
        rec['final_voxels'] = int(np.count_nonzero(out == sid))
        rec['added_voxels'] = int(np.count_nonzero((out == sid) & (seed != sid)))
        rec['removed_seed_voxels'] = int(np.count_nonzero((seed == sid) & (out != sid)))
        rec['outcome'] = ('fallback_to_seed' if rec['fallback'] else
                          'unchanged_seed' if not rec['added_voxels'] and not rec['removed_seed_voxels'] else
                          'changed_mask')
        if records[str(sid)]['fallback'] and not np.array_equal(out == sid, seed == sid):
            raise AssertionError('Fallback was not restored exactly')
    return out, {'structures': records, 'ambiguous_background_voxels': int(conflicts.sum())}, conflicts
