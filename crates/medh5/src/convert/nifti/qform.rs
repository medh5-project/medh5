//! The qform an affine is written as, as nibabel's `set_qform` derives it.
//!
//! 2.1.0 wrote every export through nibabel, which states the affine twice:
//! as the sform, verbatim, and as a qform --- voxel sizes, a handedness
//! (`qfac`), an offset and a rotation as a unit quaternion.  A rotation has
//! two quaternions, `q` and `-q`; nibabel picks the one whose real part is
//! not negative, and where that part is zero (a half turn: every LPS grid
//! written as RAS is one) the sign is whatever the eigensolver returned.
//! So the quaternion is computed here as nibabel computes it --- the
//! eigenvector of Bar-Itzhack's matrix, by the same LAPACK path NumPy's
//! `eigh` takes (`dsyevd`: `dsytd2`, `dsteqr`, `dorm2r`) --- and a file
//! exported here names the rotation with the sign 2.1.0's did.

/// `[[f64; 3]; 3]`, row-major.
type M3 = [[f64; 3]; 3];

/// The qform fields of a 4x4 affine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Qform {
    /// `pixdim[0]`: `1` for a right-handed rotation, `-1` when the third
    /// axis is flipped to make it one.
    pub qfac: f64,
    /// `pixdim[1..4]`: the affine's column lengths.
    pub zooms: [f64; 3],
    /// `quatern_b`, `quatern_c`, `quatern_d`.
    pub quaternion: [f64; 3],
    /// `qoffset_x`, `qoffset_y`, `qoffset_z`.
    pub offset: [f64; 3],
}

/// What `Nifti1Header.set_qform(affine)` writes; `None` where a column is
/// zero or a value not finite, which leaves no rotation to decompose (numpy's
/// SVD does not converge on the NaN the division makes).
pub fn qform(affine: &[[f64; 4]; 4]) -> Option<Qform> {
    let mut rzs: M3 = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            rzs[r][c] = affine[r][c];
        }
    }
    // `np.sqrt(np.sum(RZS * RZS, axis=0))`
    let mut zooms = [0.0; 3];
    for (c, zoom) in zooms.iter_mut().enumerate() {
        *zoom = (rzs[0][c] * rzs[0][c] + rzs[1][c] * rzs[1][c] + rzs[2][c] * rzs[2][c]).sqrt();
    }
    let mut rotation: M3 = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            rotation[r][c] = rzs[r][c] / zooms[c];
        }
    }
    if rotation.iter().flatten().any(|v| !v.is_finite()) {
        return None;
    }
    let qfac = if det3(&rotation) > 0.0 {
        1.0
    } else {
        for row in rotation.iter_mut() {
            row[2] = -row[2];
        }
        -1.0
    };
    let quat = mat2quat(&nearest_rotation(&rotation));
    Some(Qform {
        qfac,
        zooms,
        quaternion: [quat[1], quat[2], quat[3]],
        offset: [affine[0][3], affine[1][3], affine[2][3]],
    })
}

fn det3(m: &M3) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// The orthogonal matrix nearest `m` (its polar factor), which nibabel
/// takes from an SVD so that a sheared affine still has a quaternion.  A
/// matrix already orthonormal to rounding is its own: the SVD's product
/// moves it by an ulp at most, below what a `float32` field holds.
fn nearest_rotation(m: &M3) -> M3 {
    let gram_error = (0..3)
        .flat_map(|i| (0..3).map(move |j| (i, j)))
        .map(|(i, j)| {
            let mut dot = 0.0;
            for row in m {
                dot += row[i] * row[j];
            }
            (dot - if i == j { 1.0 } else { 0.0 }).abs()
        })
        .fold(0.0, f64::max);
    if gram_error < 1e-12 {
        return *m;
    }
    // Newton's iteration for the polar factor, X <- (X + X^-T) / 2, which
    // converges quadratically for any nonsingular matrix.
    let mut x = *m;
    for _ in 0..100 {
        let Some(inverse) = inverse3(&x) else { return x };
        let mut next = [[0.0; 3]; 3];
        let mut change: f64 = 0.0;
        for r in 0..3 {
            for c in 0..3 {
                next[r][c] = 0.5 * (x[r][c] + inverse[c][r]);
                change = change.max((next[r][c] - x[r][c]).abs());
            }
        }
        x = next;
        if change < 1e-15 {
            break;
        }
    }
    x
}

fn inverse3(m: &M3) -> Option<M3> {
    let det = det3(m);
    if det == 0.0 || !det.is_finite() {
        return None;
    }
    let cofactor = |r0: usize, r1: usize, c0: usize, c1: usize| m[r0][c0] * m[r1][c1] - m[r0][c1] * m[r1][c0];
    let adj = [
        [cofactor(1, 2, 1, 2), -cofactor(0, 2, 1, 2), cofactor(0, 1, 1, 2)],
        [-cofactor(1, 2, 0, 2), cofactor(0, 2, 0, 2), -cofactor(0, 1, 0, 2)],
        [cofactor(1, 2, 0, 1), -cofactor(0, 2, 0, 1), cofactor(0, 1, 0, 1)],
    ];
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r][c] = adj[r][c] / det;
        }
    }
    Some(out)
}

/// `nibabel.quaternions.mat2quat`: the unit eigenvector of Bar-Itzhack's
/// symmetric matrix with the largest eigenvalue, as `[w, x, y, z]`, its
/// real part made non-negative.
fn mat2quat(m: &M3) -> [f64; 4] {
    // `Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = M.flat`: nibabel's
    // names, given to the matrix read row by row.
    let [[qxx, qyx, qzx], [qxy, qyy, qzy], [qxz, qyz, qzz]] = *m;
    let k = [
        [qxx - qyy - qzz, 0.0, 0.0, 0.0],
        [qyx + qxy, qyy - qxx - qzz, 0.0, 0.0],
        [qzx + qxz, qzy + qyz, qzz - qxx - qyy, 0.0],
        [qyz - qzy, qzx - qxz, qxy - qyx, qxx + qyy + qzz],
    ];
    let mut a = [[0.0; 4]; 4];
    for r in 0..4 {
        for c in 0..4 {
            a[r][c] = k[r][c] / 3.0;
        }
    }
    let (values, vectors) = eigh_lower(a);
    let mut best = 0;
    for i in 1..4 {
        // `np.argmax`: the first of equal maxima.
        if values[i] > values[best] {
            best = i;
        }
    }
    // `vecs[[3, 0, 1, 2], argmax]`
    let mut q = [vectors[3][best], vectors[0][best], vectors[1][best], vectors[2][best]];
    if q[0] < 0.0 {
        for v in q.iter_mut() {
            *v = -*v;
        }
    }
    q
}

/// Eigenvalues (ascending) and eigenvectors (columns) of a symmetric 4x4
/// matrix of which only the lower triangle is read, as LAPACK's `dsyevd`
/// computes them for `n = 4`: `dsytd2` reduces it to tridiagonal form,
/// `dsteqr` diagonalises that by implicit QL/QR, and `dorm2r` takes the
/// eigenvectors back.  `a` is indexed `[row][column]`.
#[allow(clippy::needless_range_loop)] // indexed as LAPACK indexes it
fn eigh_lower(mut a: [[f64; 4]; 4]) -> ([f64; 4], [[f64; 4]; 4]) {
    const N: usize = 4;
    let mut d = [0.0; N];
    let mut e = [0.0; N - 1];
    let mut tau = [0.0; N - 1];
    // dsytd2, UPLO = 'L'.
    for i in 0..N - 1 {
        let m = N - i - 1; // the reflector's length
        let (alpha, t) = {
            let mut x: Vec<f64> = (i + 2..N).map(|r| a[r][i]).collect();
            let (beta, t) = dlarfg(a[i + 1][i], &mut x);
            for (r, v) in (i + 2..N).zip(x) {
                a[r][i] = v;
            }
            (beta, t)
        };
        a[i + 1][i] = alpha;
        e[i] = alpha;
        if t != 0.0 {
            a[i + 1][i] = 1.0;
            let v: Vec<f64> = (i + 1..N).map(|r| a[r][i]).collect();
            // x := tau * A * v, A the trailing block (lower triangle).
            let mut w = dsymv_lower(&a, i + 1, t, &v);
            // w := x - 1/2 * tau * (x^T v) * v
            let alpha = -0.5 * t * ddot(&w, &v);
            for (wk, vk) in w.iter_mut().zip(&v) {
                *wk += alpha * vk;
            }
            // A := A - v w^T - w v^T, lower triangle.
            for j in 0..m {
                if v[j] != 0.0 || w[j] != 0.0 {
                    let temp1 = -w[j];
                    let temp2 = -v[j];
                    for r in j..m {
                        a[i + 1 + r][i + 1 + j] = a[i + 1 + r][i + 1 + j] + v[r] * temp1 + w[r] * temp2;
                    }
                }
            }
            a[i + 1][i] = e[i];
        }
        d[i] = a[i][i];
        tau[i] = t;
    }
    d[N - 1] = a[N - 1][N - 1];
    let mut z = [[0.0; N]; N];
    for (r, row) in z.iter_mut().enumerate() {
        row[r] = 1.0;
    }
    dsteqr(&mut d, &mut e, &mut z);
    // dormtr ('L', 'L', 'N') = dorm2r on rows 2..n: Q z, Q = H(1) H(2) H(3),
    // applied H(3) first.
    for i in (0..N - 1).rev() {
        let t = tau[i];
        if t == 0.0 {
            continue;
        }
        // v = (1, a[i+2..n][i]), acting on rows i+1..n of z.
        let mut v = vec![1.0];
        v.extend((i + 2..N).map(|r| a[r][i]));
        for col in 0..N {
            let mut w = 0.0;
            for (k, vk) in v.iter().enumerate() {
                w += z[i + 1 + k][col] * vk;
            }
            let scaled = -t * w;
            for (k, vk) in v.iter().enumerate() {
                z[i + 1 + k][col] += vk * scaled;
            }
        }
    }
    (d, z)
}

/// BLAS `ddot`, summed from zero in order.
fn ddot(x: &[f64], y: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (a, b) in x.iter().zip(y) {
        sum += a * b;
    }
    sum
}

/// `alpha * A * x` for the trailing block of `a` from `start`, A symmetric
/// with its lower triangle stored --- BLAS `dsymv`'s reference loop.
fn dsymv_lower(a: &[[f64; 4]; 4], start: usize, alpha: f64, x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut y = vec![0.0; n];
    for j in 0..n {
        let temp1 = alpha * x[j];
        let mut temp2 = 0.0;
        y[j] += temp1 * a[start + j][start + j];
        for i in j + 1..n {
            y[i] += temp1 * a[start + i][start + j];
            temp2 += a[start + i][start + j] * x[i];
        }
        y[j] += alpha * temp2;
    }
    y
}

/// LAPACK `dlarfg`: the reflector that maps `(alpha, x)` to `(beta, 0)`;
/// `x` becomes the reflector's tail.  Returns `(beta, tau)`.
fn dlarfg(alpha: f64, x: &mut [f64]) -> (f64, f64) {
    if x.is_empty() {
        return (alpha, 0.0);
    }
    let xnorm = dnrm2(x);
    if xnorm == 0.0 {
        return (alpha, 0.0);
    }
    let beta = -dlapy2(alpha, xnorm).copysign(alpha);
    // `beta` below the safe minimum needs rescaling; a rotation's entries
    // never come near it.
    let tau = (beta - alpha) / beta;
    let scale = 1.0 / (alpha - beta);
    for v in x.iter_mut() {
        *v *= scale;
    }
    (beta, tau)
}

/// BLAS `dnrm2` for values far from overflow and underflow.
fn dnrm2(x: &[f64]) -> f64 {
    let mut sum = 0.0;
    for v in x {
        sum += v * v;
    }
    sum.sqrt()
}

/// LAPACK `dlapy2`: `sqrt(x^2 + y^2)` without overflow.
fn dlapy2(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    if y.is_nan() {
        return y;
    }
    let (xa, ya) = (x.abs(), y.abs());
    let w = xa.max(ya);
    let z = xa.min(ya);
    if z == 0.0 || w > f64::MAX {
        w
    } else {
        w * (1.0 + (z / w).powi(2)).sqrt()
    }
}

/// LAPACK `dlartg` (3.10 and later): the plane rotation taking `(f, g)`
/// to `(r, 0)`.  Returns `(c, s, r)`.
fn dlartg(f: f64, g: f64) -> (f64, f64, f64) {
    let safmin = f64::MIN_POSITIVE;
    let safmax = 1.0 / safmin;
    let rtmin = safmin.sqrt();
    let rtmax = (safmax / 2.0).sqrt();
    let (f1, g1) = (f.abs(), g.abs());
    if g == 0.0 {
        (1.0, 0.0, f)
    } else if f == 0.0 {
        (0.0, 1f64.copysign(g), g1)
    } else if f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax {
        let d = (f * f + g * g).sqrt();
        let c = f1 / d;
        let r = d.copysign(f);
        (c, g / r, r)
    } else {
        let u = safmax.min(safmin.max(f1).max(g1));
        let (fs, gs) = (f / u, g / u);
        let d = (fs * fs + gs * gs).sqrt();
        let c = fs.abs() / d;
        let r = d.copysign(f);
        let s = gs / r;
        (c, s, r * u)
    }
}

/// LAPACK `dlaev2`: the eigen-decomposition of `[[a, b], [b, c]]`.
/// Returns `(rt1, rt2, cs1, sn1)`.
fn dlaev2(a: f64, b: f64, c: f64) -> (f64, f64, f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        adf * (1.0 + (ab / adf).powi(2)).sqrt()
    } else if adf < ab {
        ab * (1.0 + (adf / ab).powi(2)).sqrt()
    } else {
        ab * 2f64.sqrt()
    };
    let (rt1, rt2, sgn1) = if sm < 0.0 {
        let rt1 = 0.5 * (sm - rt);
        (rt1, (acmx / rt1) * acmn - (b / rt1) * b, -1)
    } else if sm > 0.0 {
        let rt1 = 0.5 * (sm + rt);
        (rt1, (acmx / rt1) * acmn - (b / rt1) * b, 1)
    } else {
        (0.5 * rt, -0.5 * rt, 1)
    };
    let (cs, sgn2) = if df >= 0.0 { (df + rt, 1) } else { (df - rt, -1) };
    let acs = cs.abs();
    let (mut cs1, mut sn1);
    if acs > ab {
        let ct = -tb / cs;
        sn1 = 1.0 / (1.0 + ct * ct).sqrt();
        cs1 = ct * sn1;
    } else if ab == 0.0 {
        cs1 = 1.0;
        sn1 = 0.0;
    } else {
        let tn = -cs / tb;
        cs1 = 1.0 / (1.0 + tn * tn).sqrt();
        sn1 = tn * cs1;
    }
    if sgn1 == sgn2 {
        let tn = cs1;
        cs1 = -sn1;
        sn1 = tn;
    }
    (rt1, rt2, cs1, sn1)
}

/// LAPACK `dlasr` with `SIDE = 'R'`, `PIVOT = 'V'`: the rotations
/// `(c[j], s[j])` applied to columns `j, j+1` of `z` from column `first`,
/// forward or backward.
fn dlasr_right(z: &mut [[f64; 4]; 4], first: usize, c: &[f64], s: &[f64], forward: bool) {
    let count = c.len();
    let order: Vec<usize> = if forward { (0..count).collect() } else { (0..count).rev().collect() };
    for j in order {
        let (ct, st) = (c[j], s[j]);
        if ct != 1.0 || st != 0.0 {
            for row in z.iter_mut() {
                let temp = row[first + j + 1];
                row[first + j + 1] = ct * temp - st * row[first + j];
                row[first + j] = st * temp + ct * row[first + j];
            }
        }
    }
}

/// LAPACK `dsteqr` with `COMPZ = 'I'` for `n = 4`: the eigenvalues of the
/// symmetric tridiagonal `(d, e)` into `d`, ascending, and `z` (the
/// identity on entry) holding their eigenvectors.
fn dsteqr(d: &mut [f64; 4], e: &mut [f64; 3], z: &mut [[f64; 4]; 4]) {
    const N: usize = 4;
    const MAXIT: usize = 30;
    let eps = f64::EPSILON / 2.0; // DLAMCH('E'): the relative machine precision
    let eps2 = eps * eps;
    let safmin = f64::MIN_POSITIVE;
    let safmax = 1.0 / safmin;
    let ssfmax = safmax.sqrt() / 3.0;
    let ssfmin = safmin.sqrt() / eps2;
    let nmaxit = N * MAXIT;
    let mut jtot = 0;
    let mut work = [0.0; 2 * (N - 1)];
    // 1-based indices, as the Fortran: d(k) is d[k-1], e(k) is e[k-1].
    let mut l1 = 1usize;
    loop {
        if l1 > N {
            break;
        }
        if l1 > 1 {
            e[l1 - 2] = 0.0;
        }
        let mut m = N;
        if l1 < N {
            for mm in l1..N {
                let tst = e[mm - 1].abs();
                if tst == 0.0 {
                    m = mm;
                    break;
                }
                if tst <= (d[mm - 1].abs().sqrt() * d[mm].abs().sqrt()) * eps {
                    e[mm - 1] = 0.0;
                    m = mm;
                    break;
                }
            }
        }
        let mut l = l1;
        let lsv = l;
        let mut lend = m;
        let lendsv = lend;
        l1 = m + 1;
        if lend == l {
            continue;
        }
        // Scale submatrix in rows and columns l to lend.
        let anorm = {
            let mut anorm = d[lend - 1].abs();
            for i in l..lend {
                let sum = d[i - 1].abs();
                if anorm < sum || sum.is_nan() {
                    anorm = sum;
                }
                let sum = e[i - 1].abs();
                if anorm < sum || sum.is_nan() {
                    anorm = sum;
                }
            }
            anorm
        };
        if anorm == 0.0 {
            continue;
        }
        let iscale = if anorm > ssfmax {
            dlascl(anorm, ssfmax, &mut d[l - 1..lend]);
            dlascl(anorm, ssfmax, &mut e[l - 1..lend - 1]);
            1
        } else if anorm < ssfmin {
            dlascl(anorm, ssfmin, &mut d[l - 1..lend]);
            dlascl(anorm, ssfmin, &mut e[l - 1..lend - 1]);
            2
        } else {
            0
        };
        // Choose between QL and QR iteration.
        if d[lend - 1].abs() < d[l - 1].abs() {
            lend = lsv;
            l = lendsv;
        }
        if lend > l {
            // QL iteration: look for a small subdiagonal element.
            loop {
                let mut m = lend;
                if l != lend {
                    for mm in l..lend {
                        let tst = e[mm - 1].abs().powi(2);
                        if tst <= (eps2 * d[mm - 1].abs()) * d[mm].abs() + safmin {
                            m = mm;
                            break;
                        }
                    }
                }
                if m < lend {
                    e[m - 1] = 0.0;
                }
                let mut p = d[l - 1];
                if m == l {
                    // eigenvalue found
                    d[l - 1] = p;
                    l += 1;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if m == l + 1 {
                    let (rt1, rt2, c, s) = dlaev2(d[l - 1], e[l - 1], d[l]);
                    work[l - 1] = c;
                    work[N - 1 + l - 1] = s;
                    let (cs, ss) = ([work[l - 1]], [work[N - 1 + l - 1]]);
                    dlasr_right(z, l - 1, &cs, &ss, false);
                    d[l - 1] = rt1;
                    d[l] = rt2;
                    e[l - 1] = 0.0;
                    l += 2;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;
                // Form shift.
                let mut g = (d[l] - p) / (2.0 * e[l - 1]);
                let mut r = dlapy2(g, 1.0);
                g = d[m - 1] - p + (e[l - 1] / (g + r.copysign(g)));
                let mut s = 1.0;
                let mut c = 1.0;
                p = 0.0;
                for i in (l..m).rev() {
                    let f = s * e[i - 1];
                    let b = c * e[i - 1];
                    let (cc, ss, rr) = dlartg(g, f);
                    c = cc;
                    s = ss;
                    r = rr;
                    if i != m - 1 {
                        e[i] = r;
                    }
                    g = d[i] - p;
                    r = (d[i - 1] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i] = g + p;
                    g = c * r - b;
                    work[i - 1] = c;
                    work[N - 1 + i - 1] = -s;
                }
                let mm = m - l + 1;
                let cs: Vec<f64> = work[l - 1..l - 1 + mm - 1].to_vec();
                let ss: Vec<f64> = work[N - 1 + l - 1..N - 1 + l - 1 + mm - 1].to_vec();
                dlasr_right(z, l - 1, &cs, &ss, false);
                d[l - 1] -= p;
                e[l - 1] = g;
            }
        } else {
            // QR iteration: look for a small superdiagonal element.
            loop {
                let mut m = lend;
                if l != lend {
                    let mut mm = l;
                    while mm > lend {
                        let tst = e[mm - 2].abs().powi(2);
                        if tst <= (eps2 * d[mm - 1].abs()) * d[mm - 2].abs() + safmin {
                            m = mm;
                            break;
                        }
                        mm -= 1;
                    }
                }
                if m > lend {
                    e[m - 2] = 0.0;
                }
                let mut p = d[l - 1];
                if m == l {
                    d[l - 1] = p;
                    if l == lend {
                        break;
                    }
                    l -= 1;
                    continue;
                }
                if m + 1 == l {
                    let (rt1, rt2, c, s) = dlaev2(d[l - 2], e[l - 2], d[l - 1]);
                    work[m - 1] = c;
                    work[N - 1 + m - 1] = s;
                    let (cs, ss) = ([work[m - 1]], [work[N - 1 + m - 1]]);
                    dlasr_right(z, l - 2, &cs, &ss, true);
                    d[l - 2] = rt1;
                    d[l - 1] = rt2;
                    e[l - 2] = 0.0;
                    if l < lend + 2 {
                        break;
                    }
                    l -= 2;
                    continue;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;
                // Form shift.
                let mut g = (d[l - 2] - p) / (2.0 * e[l - 2]);
                let mut r = dlapy2(g, 1.0);
                g = d[m - 1] - p + (e[l - 2] / (g + r.copysign(g)));
                let mut s = 1.0;
                let mut c = 1.0;
                p = 0.0;
                for i in m..l {
                    let f = s * e[i - 1];
                    let b = c * e[i - 1];
                    let (cc, ss, rr) = dlartg(g, f);
                    c = cc;
                    s = ss;
                    r = rr;
                    if i != m {
                        e[i - 2] = r;
                    }
                    g = d[i - 1] - p;
                    r = (d[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    d[i - 1] = g + p;
                    g = c * r - b;
                    work[i - 1] = c;
                    work[N - 1 + i - 1] = s;
                }
                let mm = l - m + 1;
                let cs: Vec<f64> = work[m - 1..m - 1 + mm - 1].to_vec();
                let ss: Vec<f64> = work[N - 1 + m - 1..N - 1 + m - 1 + mm - 1].to_vec();
                dlasr_right(z, m - 1, &cs, &ss, true);
                d[l - 1] -= p;
                e[l - 2] = g;
            }
        }
        // Undo scaling if necessary.
        match iscale {
            1 => {
                dlascl(ssfmax, anorm, &mut d[lsv - 1..lendsv]);
                dlascl(ssfmax, anorm, &mut e[lsv - 1..lendsv - 1]);
            }
            2 => {
                dlascl(ssfmin, anorm, &mut d[lsv - 1..lendsv]);
                dlascl(ssfmin, anorm, &mut e[lsv - 1..lendsv - 1]);
            }
            _ => {}
        }
        if jtot >= nmaxit {
            break;
        }
    }
    // Selection sort, ascending, swapping the vectors along.
    for i in 0..N - 1 {
        let mut k = i;
        let mut p = d[i];
        for (j, value) in d.iter().enumerate().skip(i + 1) {
            if *value < p {
                k = j;
                p = *value;
            }
        }
        if k != i {
            d[k] = d[i];
            d[i] = p;
            for row in z.iter_mut() {
                row.swap(i, k);
            }
        }
    }
}

/// `dlascl('G')` for values far from overflow: multiply by `to / from`.
fn dlascl(from: f64, to: f64, values: &mut [f64]) {
    let factor = to / from;
    for v in values.iter_mut() {
        *v *= factor;
    }
}
