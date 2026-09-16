"""Reproduce every fitted constant, validation number, figure and worked
example in the repository, and report any result table that differs from the
copy that was present before the run (in a clean checkout, the committed copy).

    python reproduce.py            # everything; rewrites results, tables and figures
    python reproduce.py --quick    # validation suite only; writes nothing

What is compared: every file in validation/results/ and examples/tables/,
field by field, numeric fields to a relative tolerance of 1e-8 (last-digit
differences between numpy/scipy builds are not a changed result). Files that
appear or disappear are reported as changes. Figures are regenerated but not
compared, and printed output other than those tables is not checked.

Run the full reproduction in a disposable checkout: it overwrites the files
it compares and every figure in figures/out/.

Order matters: the dataset scorers write validation/results/*.csv that the
salt-term generator and two diagnostics read back.
"""
import filecmp
import math
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

STEPS = [
    ('validation suite', ['-m', 'validation.validation']),
    ('Calabrese density scorer', ['validation/calabrese_density_validation.py']),
    ('Yan density scorer', ['validation/yan2011_validation.py']),
    ('volume shifts (asserts the packaged values)', ['fits/fit_pr_vshift.py']),
    ('salinity factor (asserts the packaged values)', ['fits/relative_salt_shift.py']),
    ('H2S coefficient and composition slope', ['fits/murphy_gaines_h2s_refit.py']),
    ('CH4 viscosity coefficients', ['fits/ostermann_ch4_refit.py']),
    ('temperature ceiling', ['validation/temperature_ceiling_test.py']),
    ('mixed-brine viscosity', ['validation/mixed_brine_viscosity_validation.py']),
    ('CaCl2 at pressure', ['validation/cacl2_pressure_viscosity_validation.py']),
    ('Ezrokhi defaults audit', ['validation/ezrokhi_vs_default.py']),
    ('Ezrokhi coefficients', ['fits/ezrokhi_fits.py']),
    ('Ezrokhi pressure cubics', ['fits/ezrokhi_pressure_fit.py']),
    ('worked examples and their tables', ['examples/worked_examples.py', '--tex']),
    ('figures', ['figures/make_figures.py']),
    ('parity figures', ['figures/vphi_freshwater_fit_figure.py']),
]

COMPARED = [os.path.join('validation', 'results'), os.path.join('examples', 'tables')]


def _same_result(a, b, rel=1e-8):
    """Byte-equal, or equal field by field with numeric fields compared to a
    relative tolerance."""
    if filecmp.cmp(a, b, shallow=False):
        return True
    la, lb = open(a).read().splitlines(), open(b).read().splitlines()
    if len(la) != len(lb):
        return False
    for ra, rb in zip(la, lb):
        fa, fb = ra.split(','), rb.split(',')
        if len(fa) != len(fb):
            return False
        for x, y in zip(fa, fb):
            if x == y:
                continue
            try:
                u, v = float(x), float(y)
            except ValueError:
                return False
            if not (math.isclose(u, v, rel_tol=rel, abs_tol=1e-12) or (math.isnan(u) and math.isnan(v))):
                return False
    return True


def main():
    quick = '--quick' in sys.argv
    steps = STEPS[:1] if quick else STEPS
    keep = tempfile.mkdtemp(prefix='brine_props_before_')
    before = {}
    for rel in COMPARED:
        d = os.path.join(ROOT, rel)
        os.makedirs(os.path.join(keep, rel), exist_ok=True)
        for fn in sorted(os.listdir(d)) if os.path.isdir(d) else []:
            if os.path.isfile(os.path.join(d, fn)):
                shutil.copy(os.path.join(d, fn), os.path.join(keep, rel, fn))
                before[os.path.join(rel, fn)] = True
    env = dict(os.environ, PYTHONPATH=ROOT)
    failed = []
    for label, cmd in steps:
        print(f'== {label}')
        r = subprocess.run([PY] + cmd, cwd=ROOT, env=env)
        if r.returncode != 0:
            failed.append(label)
    changed, missing, new = [], [], []
    if not quick:
        for rel in COMPARED:
            d = os.path.join(ROOT, rel)
            after = {fn for fn in os.listdir(d) if os.path.isfile(os.path.join(d, fn))}
            was = {os.path.basename(k) for k in before if k.startswith(rel)}
            missing += [os.path.join(rel, fn) for fn in sorted(was - after)]
            new += [os.path.join(rel, fn) for fn in sorted(after - was)]
            for fn in sorted(was & after):
                if not _same_result(os.path.join(keep, rel, fn), os.path.join(d, fn)):
                    changed.append(os.path.join(rel, fn))
    print('\n' + '=' * 60)
    print('failed steps:', failed or 'none')
    if quick:
        print('quick run: nothing regenerated, nothing compared')
    else:
        print(f'compared {len(before)} files in {" and ".join(COMPARED)}')
        print('results that changed from the pre-run copy:', changed or 'none')
        if new or missing:
            print('files that appeared:', new or 'none')
            print('files that disappeared:', missing or 'none')
    shutil.rmtree(keep, ignore_errors=True)
    return 1 if (failed or changed or new or missing) else 0


if __name__ == '__main__':
    sys.exit(main())
