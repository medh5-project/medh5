//! What an annotation's datasets must agree on (spec §7.4, §8, §9).
//!
//! A per-object column holds one row per object, an offset table one row
//! more, and a coordinate array one column per spatial axis.  The readers
//! index every column by row, so a column one row short would be an index
//! past its end --- a panic, not an error --- and a mask shape nobody bounds
//! an allocation of whatever it says, which aborts the process.  The encoders
//! build columns that agree; a stored annotation is held to the same table
//! when it is opened, when a payload is decoded and when a file is validated,
//! so all three report a disagreement under the same code (E405, E408) and
//! none of them reaches the code that would index past it.

use ndarray::{ArrayD, Axis};

use crate::geometry::affine::box_to_slices;
use crate::json::{repr_int_list, repr_int_tuple};

/// One disagreement between an annotation's datasets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Problem {
    /// The §15.2 code.
    pub code: &'static str,
    /// The dataset it is about.
    pub dataset: String,
    pub message: String,
}

impl Problem {
    fn new(code: &'static str, dataset: &str, message: String) -> Problem {
        Problem { code, dataset: dataset.to_string(), message }
    }

    /// The problem as a coded error about annotation `ann_id`.
    pub fn error(&self, ann_id: &str) -> crate::Error {
        crate::Error::coded(self.code, format!("annotation {}: {}", crate::json::repr_str(ann_id), self.message))
    }
}

/// One axis of a dataset's shape.
#[derive(Debug, Clone, Copy)]
enum Dim {
    /// A fixed extent.
    Is(usize),
    /// A count every dataset naming it shares.
    Var(char),
    /// One more than a count: an offset table.
    Plus1(char),
}

use Dim::{Is, Plus1, Var};

const N: Dim = Var('N');
const S: Dim = Var('S');
const K: Dim = Var('K');
const M: Dim = Var('M');
const V: Dim = Var('V');

/// The datasets of each kind with per-object rows, in the order their counts
/// are taken: the first dataset present fixes a count, the rest must agree.
fn columns(kind: &str) -> &'static [(&'static str, &'static [Dim])] {
    match kind {
        "instances" => &[
            ("boxes", &[N, S, Is(2)]),
            ("class_ids", &[N]),
            ("instance_ids", &[N]),
            ("scores", &[N]),
            ("mask_offsets", &[Plus1('N')]),
            ("mask_shapes", &[N, S]),
            ("mask_data", &[Var('B')]),
        ],
        "boxes" => &[
            ("boxes", &[N, S, Is(2)]),
            ("class_ids", &[N]),
            ("instance_ids", &[N]),
            ("scores", &[N]),
            ("attributes", &[N]),
            ("slice_index", &[N]),
        ],
        "obb" => &[
            ("centers", &[N, S]),
            ("sizes", &[N, S]),
            ("rotations", &[N, S, S]),
            ("class_ids", &[N]),
            ("instance_ids", &[N]),
            ("scores", &[N]),
            ("attributes", &[N]),
        ],
        "keypoints" => &[
            ("points", &[N, K, S]),
            ("visibility", &[N, K]),
            ("keypoint_class_ids", &[K]),
            ("class_ids", &[N]),
            ("instance_ids", &[N]),
            ("scores", &[N]),
            ("attributes", &[N]),
        ],
        "points" => {
            &[("points", &[N, S]), ("class_ids", &[N]), ("instance_ids", &[N]), ("names", &[N]), ("weights", &[N])]
        }
        "contours" => &[
            ("contour_offsets", &[Plus1('M')]),
            ("vertices", &[V, S]),
            ("contour_class_ids", &[M]),
            ("contour_plane", &[M, Is(2)]),
            ("contour_role", &[M]),
        ],
        "polylines" => &[
            ("polyline_offsets", &[Plus1('M')]),
            ("vertices", &[V, S]),
            ("polyline_class_ids", &[M]),
            ("polyline_plane", &[M, Is(2)]),
            ("instance_ids", &[M]),
        ],
        "mesh" => &[
            ("vertices", &[V, Is(3)]),
            ("faces", &[Var('F'), Is(3)]),
            ("normals", &[V, Is(3)]),
            ("vertex_class_ids", &[V]),
            ("mesh_offsets", &[Plus1('M')]),
            ("mesh_class_ids", &[M]),
        ],
        "classification" => {
            &[("class_ids", &[K]), ("values", &[K]), ("scope_ids", &[K]), ("schemes", &[K]), ("scheme_values", &[K])]
        }
        _ => &[],
    }
}

/// What a count counts, for messages.
fn noun(kind: &str, symbol: char) -> &'static str {
    match (kind, symbol) {
        ("boxes" | "obb", 'N') => "box(es)",
        ("points", 'N') => "point(s)",
        (_, 'N') => "object(s)",
        ("keypoints", 'K') => "keypoint slot(s)",
        (_, 'K') => "assertion(s)",
        ("contours", 'M') => "polygon(s)",
        ("polylines", 'M') => "polyline(s)",
        (_, 'M') => "mesh(es)",
        (_, 'V') => "vertices",
        (_, 'F') => "faces",
        (_, 'S') => "spatial axes",
        _ => "bytes",
    }
}

/// Where a count came from.
#[derive(Debug, Clone)]
enum Source {
    Dataset(String),
    Grid,
    OneMesh,
}

#[derive(Debug, Clone)]
struct Bound {
    symbol: char,
    value: usize,
    source: Source,
}

fn lookup(bound: &[Bound], symbol: char) -> Option<&Bound> {
    bound.iter().find(|b| b.symbol == symbol)
}

/// A shape with every known count filled in, as Python prints a tuple.
fn render(pattern: &[Dim], bound: &[Bound]) -> String {
    let parts: Vec<String> = pattern
        .iter()
        .map(|d| match d {
            Is(v) => v.to_string(),
            Var(c) => lookup(bound, *c).map(|b| b.value.to_string()).unwrap_or_else(|| c.to_string()),
            Plus1(c) => lookup(bound, *c).map(|b| (b.value + 1).to_string()).unwrap_or_else(|| format!("{c}+1")),
        })
        .collect();
    if parts.len() == 1 {
        format!("({},)", parts[0])
    } else {
        format!("({})", parts.join(", "))
    }
}

/// Why a count is what it is.
fn because(kind: &str, b: &Bound) -> String {
    match &b.source {
        Source::Grid => format!("the grid has {} spatial axes", b.value),
        Source::OneMesh => "there is one mesh where `mesh_offsets` is absent".to_string(),
        Source::Dataset(name) => format!("`{name}` gives {} {}", b.value, noun(kind, b.symbol)),
    }
}

fn check_one(kind: &str, name: &str, shape: &[usize], pattern: &[Dim], bound: &mut Vec<Bound>) -> Option<Problem> {
    let got = repr_int_tuple(shape);
    if shape.len() != pattern.len() {
        return Some(Problem::new(
            "E405",
            name,
            format!("`{name}` has shape {got}; a `{kind}` annotation stores it as {}", render(pattern, bound)),
        ));
    }
    let mut fresh: Vec<Bound> = Vec::new();
    for (dim, extent) in pattern.iter().zip(shape) {
        let reason = match dim {
            Is(v) if extent != v => Some(None),
            Is(_) => None,
            Var(c) => match lookup(bound, *c).or_else(|| lookup(&fresh, *c)) {
                Some(b) if b.value != *extent => Some(Some(b.clone())),
                Some(_) => None,
                None => {
                    fresh.push(Bound { symbol: *c, value: *extent, source: Source::Dataset(name.to_string()) });
                    None
                }
            },
            Plus1(_) if *extent == 0 => {
                return Some(Problem::new(
                    "E405",
                    name,
                    format!("`{name}` is empty; an offset table starts at 0, so it has at least one entry"),
                ))
            }
            Plus1(c) => match lookup(bound, *c).or_else(|| lookup(&fresh, *c)) {
                Some(b) if b.value + 1 != *extent => Some(Some(b.clone())),
                Some(_) => None,
                None => {
                    fresh.push(Bound { symbol: *c, value: extent - 1, source: Source::Dataset(name.to_string()) });
                    None
                }
            },
        };
        if let Some(reason) = reason {
            let mut known = bound.clone();
            known.extend(fresh);
            let want = render(pattern, &known);
            let message = match reason {
                Some(b) => format!("`{name}` has shape {got}, but {}, so it must be {want}", because(kind, &b)),
                None => format!("`{name}` has shape {got}; a `{kind}` annotation stores it as {want}"),
            };
            return Some(Problem::new("E405", name, message));
        }
    }
    bound.extend(fresh);
    None
}

/// Every per-object dataset of a `kind` annotation whose shape disagrees
/// with the others, or with `spatial` --- the spatial rank its coordinates
/// must have, when they are in a grid's index space.
///
/// `shape_of` gives a dataset's shape, `None` when it is absent; an absent
/// dataset is not this check's finding (E410 is).  Kinds without per-object
/// rows have nothing to disagree on.
pub fn column_problems(
    kind: &str,
    shape_of: &dyn Fn(&str) -> Option<Vec<usize>>,
    spatial: Option<usize>,
) -> Vec<Problem> {
    let mut bound = Vec::new();
    if let Some(s) = spatial {
        bound.push(Bound { symbol: 'S', value: s, source: Source::Grid });
    }
    if kind == "mesh" && shape_of("mesh_offsets").is_none() {
        bound.push(Bound { symbol: 'M', value: 1, source: Source::OneMesh });
    }
    let mut out = Vec::new();
    for (name, pattern) in columns(kind) {
        let Some(shape) = shape_of(name) else { continue };
        if let Some(problem) = check_one(kind, name, &shape, pattern, &mut bound) {
            out.push(problem);
        }
    }
    out
}

/// The class table of a `layers` or `bitmask` annotation against its data
/// and its classes (§7.2, §7.3).
///
/// Its shape first (E405): `layer_class_ids` is `(L, K)`, one row per stored
/// layer --- a lone row `(K,)` is one layer --- and `bit_class_ids` is `(C,)`.
/// A `bitmask` stores `ceil(C / 64)` planes (E404, as the validator words
/// it).  `planes` is how many `data` stores, when known.
///
/// Then, given the table's `rows` --- one per layer; `bit_class_ids` is one
/// row --- what it maps (E404): no class twice, and given `declared` (the
/// `class_ids`, when known) every declared class and no other.  `0` pads a
/// layer row and is no class.
pub(crate) fn table_problems(
    kind: &str,
    shape_of: &dyn Fn(&str) -> Option<Vec<usize>>,
    planes: Option<usize>,
    rows: Option<&[Vec<i64>]>,
    declared: Option<&[i64]>,
) -> Vec<Problem> {
    let name = match kind {
        "layers" => "layer_class_ids",
        "bitmask" => "bit_class_ids",
        _ => return Vec::new(),
    };
    let Some(shape) = shape_of(name) else { return Vec::new() };
    let got = repr_int_tuple(&shape);
    if kind == "layers" {
        if shape.len() > 2 {
            let message = format!("`{name}` has shape {got}; a `layers` annotation stores it as (L, K)");
            return vec![Problem::new("E405", name, message)];
        }
        let n_rows = if shape.len() == 2 { shape[0] } else { 1 };
        if let Some(planes) = planes.filter(|p| *p != n_rows) {
            let message = format!(
                "`{name}` has {n_rows} row(s) for the {planes} layer(s) `data` stores; the table has one row of class \
                 ids per layer (§7.2)"
            );
            return vec![Problem::new("E405", name, message)];
        }
    } else {
        if shape.len() != 1 {
            let message = format!("`{name}` has shape {got}; a `bitmask` annotation stores it as (C,)");
            return vec![Problem::new("E405", name, message)];
        }
        let expected = shape[0].div_ceil(64).max(1);
        if let Some(planes) = planes.filter(|p| *p != expected) {
            let message = format!("{planes} bitplanes for {} classes; expected {expected}", shape[0]);
            return vec![Problem::new("E404", "data", message)];
        }
    }
    let Some(rows) = rows else { return Vec::new() };
    let mut seen: std::collections::BTreeMap<i64, usize> = std::collections::BTreeMap::new();
    let flat = rows.iter().enumerate().flat_map(|(r, row)| row.iter().enumerate().map(move |(k, c)| (r, k, *c)));
    for (row, column, class_id) in flat {
        if kind == "layers" && class_id == 0 {
            continue;
        }
        let place = if kind == "layers" { row } else { column };
        if let Some(first) = seen.insert(class_id, place) {
            let message = if kind == "layers" {
                format!("class {class_id} appears in layers {first} and {place}")
            } else {
                format!("class {class_id} has bits {first} and {place}; a class has one bit (§7.3)")
            };
            return vec![Problem::new("E404", name, message)];
        }
    }
    let mut out = Vec::new();
    let Some(declared) = declared else { return out };
    let mut missing: Vec<i64> = declared.iter().copied().filter(|c| !seen.contains_key(c)).collect();
    missing.sort_unstable();
    missing.dedup();
    if !missing.is_empty() {
        let where_ = if kind == "layers" { "are in no layer of" } else { "have no bit in" };
        let message = format!("class_ids {} {where_} `{name}`; every class is stored once", repr_int_list(&missing));
        out.push(Problem::new("E404", name, message));
    }
    let undeclared: Vec<i64> = seen.keys().copied().filter(|c| !declared.contains(c)).collect();
    if !undeclared.is_empty() {
        let message = format!(
            "`{name}` holds class id(s) {}, which `class_ids` does not declare; it declares every class the \
             annotation can express (§6.2)",
            repr_int_list(&undeclared)
        );
        out.push(Problem::new("E404", name, message));
    }
    out
}

/// Whether a `kind` annotation's coordinates have to match the spatial rank
/// of its grid: every voxel kind's objects and every geometric kind in
/// `index` space (§8.1) --- `mesh` excepted, whose vertices are 3-D (§8.7).
pub fn ranked(kind: &str, space: Option<&str>) -> bool {
    match kind {
        "instances" => true,
        "boxes" | "obb" | "keypoints" | "points" | "contours" | "polylines" => space.unwrap_or("index") == "index",
        _ => false,
    }
}

/// The stored masks of an `instances` annotation against its boxes and its
/// grid (§7.4): for each object a crop whose shape is its box's extent and no
/// larger than the grid (E405), and offsets that delimit exactly the bytes
/// that crop packs into, inside `mask_data` (E408) --- offsets that fall, that
/// run past the data, or that give an object more or fewer bytes than its
/// packed crop.  A reader that trusted a short span padded the missing bits
/// with `false`, and returned part of a mask as the whole of it.
///
/// The columns are assumed to agree in shape ([`column_problems`]).  An
/// object whose box has `lo > hi` is skipped: its extent is undefined, and
/// that is E406's finding.  Without a `grid` the size bound is not checked.
pub fn mask_problems(
    boxes: &ArrayD<f64>,
    offsets: &[i64],
    shapes: &ArrayD<i64>,
    mask_bytes: usize,
    grid: Option<&[usize]>,
) -> Vec<Problem> {
    if offsets.first().is_some_and(|o| *o < 0) || offsets.windows(2).any(|w| w[1] < w[0]) {
        return vec![Problem::new("E408", "mask_offsets", "`mask_offsets` are not monotonic".to_string())];
    }
    let end = offsets.last().copied().unwrap_or(0);
    if end as u64 > mask_bytes as u64 {
        return vec![Problem::new(
            "E408",
            "mask_offsets",
            format!("`mask_offsets` end at byte {end}, past the {mask_bytes} bytes of `mask_data`"),
        )];
    }
    let voxels = grid.map(|g| g.iter().try_fold(1u128, |a, b| a.checked_mul(*b as u128)).unwrap_or(u128::MAX));
    let mut out = Vec::new();
    let rows = |a: &[usize]| a.first().copied().unwrap_or(0);
    let n = offsets.len().saturating_sub(1).min(rows(boxes.shape())).min(rows(shapes.shape()));
    for i in 0..n {
        let bbox: Vec<f64> = boxes.index_axis(Axis(0), i).iter().copied().collect();
        let Ok(slices) = box_to_slices(&bbox, None) else { continue };
        // Saturating: a finite bound past the integers is the end of them, and
        // a box from one end to the other overflowed the subtraction.
        let extent: Vec<i64> = slices.iter().map(|(a, b)| b.saturating_sub(*a)).collect();
        let row: Vec<i64> = shapes.index_axis(Axis(0), i).iter().copied().collect();
        if row != extent {
            out.push(Problem::new(
                "E405",
                "mask_shapes",
                format!(
                    "object {i}'s mask is {} voxels, and its box spans {}; a mask is cropped to its box",
                    repr_int_tuple(&row),
                    repr_int_tuple(&extent)
                ),
            ));
            continue;
        }
        let size = extent.iter().try_fold(1u128, |a, b| a.checked_mul(*b as u128)).unwrap_or(u128::MAX);
        if let Some(voxels) = voxels.filter(|v| size > *v) {
            out.push(Problem::new(
                "E405",
                "mask_shapes",
                format!(
                    "object {i}'s mask is {} voxels, more than the {voxels} of the grid it is cut from",
                    repr_int_tuple(&row)
                ),
            ));
            continue;
        }
        let (start, stop) = (offsets[i], offsets[i + 1]);
        let needed = size.div_ceil(8);
        if (stop - start) as u128 != needed {
            out.push(Problem::new(
                "E408",
                "mask_offsets",
                format!(
                    "`mask_offsets` give object {i} bytes {start}..{stop} of `mask_data`, {} byte(s), and its {} mask \
                     packs into {needed}",
                    stop - start,
                    repr_int_tuple(&row)
                ),
            ));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::IxDyn;
    use std::collections::HashMap;

    fn shapes(entries: &[(&str, &[usize])]) -> HashMap<String, Vec<usize>> {
        entries.iter().map(|(k, v)| (k.to_string(), v.to_vec())).collect()
    }

    fn problems(kind: &str, entries: &[(&str, &[usize])], spatial: Option<usize>) -> Vec<Problem> {
        let table = shapes(entries);
        column_problems(kind, &|name| table.get(name).cloned(), spatial)
    }

    #[test]
    fn s8_2_every_box_column_has_one_row_per_box() {
        let good = [("boxes", &[2, 3, 2][..]), ("class_ids", &[2]), ("scores", &[2]), ("slice_index", &[2])];
        assert!(problems("boxes", &good, Some(3)).is_empty());
        let short = problems("boxes", &[("boxes", &[2, 3, 2]), ("class_ids", &[1])], Some(3));
        assert_eq!(short.len(), 1);
        assert_eq!((short[0].code, short[0].dataset.as_str()), ("E405", "class_ids"));
        assert!(short[0].message.contains("must be (2,)"), "{}", short[0].message);
        // A scalar dataset, and a last axis that is not [lo, hi].
        assert_eq!(problems("boxes", &[("boxes", &[]), ("class_ids", &[2])], None).len(), 1);
        assert_eq!(problems("boxes", &[("boxes", &[2, 3, 3]), ("class_ids", &[2])], None)[0].dataset, "boxes");
    }

    #[test]
    fn s8_1_index_coordinates_have_the_grid_rank() {
        let found = problems("boxes", &[("boxes", &[1, 2, 2]), ("class_ids", &[1])], Some(3));
        assert!(found[0].message.contains("the grid has 3 spatial axes"), "{}", found[0].message);
        assert!(problems("points", &[("points", &[4, 2])], None).is_empty());
    }

    #[test]
    fn s7_4_instance_offsets_have_one_entry_more_than_objects() {
        let good = [
            ("boxes", &[2, 3, 2][..]),
            ("class_ids", &[2]),
            ("instance_ids", &[2]),
            ("mask_offsets", &[3]),
            ("mask_shapes", &[2, 3]),
            ("mask_data", &[9]),
        ];
        assert!(problems("instances", &good, Some(3)).is_empty());
        let mut bad = good;
        bad[3] = ("mask_offsets", &[2]);
        bad[4] = ("mask_shapes", &[1, 3]);
        let found = problems("instances", &bad, Some(3));
        assert_eq!(found.iter().map(|p| p.dataset.as_str()).collect::<Vec<_>>(), ["mask_offsets", "mask_shapes"]);
        assert_eq!(problems("instances", &[("mask_offsets", &[0])], None)[0].code, "E405");
    }

    #[test]
    fn s8_3_s8_4_s8_6_s8_7_s9_shapes() {
        assert_eq!(
            problems("obb", &[("centers", &[1, 3]), ("sizes", &[3, 2]), ("rotations", &[1, 2, 2])], None).len(),
            2
        );
        assert_eq!(
            problems(
                "keypoints",
                &[("points", &[1, 2, 3]), ("visibility", &[1, 3]), ("keypoint_class_ids", &[2])],
                None
            )
            .len(),
            1
        );
        assert_eq!(
            problems(
                "contours",
                &[("contour_offsets", &[3]), ("vertices", &[7, 3]), ("contour_class_ids", &[1])],
                None
            )
            .len(),
            1
        );
        assert!(
            problems("mesh", &[("vertices", &[4, 3]), ("faces", &[2, 3]), ("mesh_class_ids", &[1])], None).is_empty()
        );
        assert_eq!(problems("mesh", &[("vertices", &[4, 3]), ("mesh_class_ids", &[2])], None).len(), 1);
        assert_eq!(problems("classification", &[("class_ids", &[2]), ("values", &[1])], None).len(), 1);
    }

    fn column(shape: &[usize], values: Vec<i64>) -> ArrayD<i64> {
        ArrayD::from_shape_vec(IxDyn(shape), values).unwrap()
    }

    #[test]
    fn s7_4_a_mask_is_its_box_and_no_larger_than_its_grid() {
        // Two objects: a 2x3x3 box and a 2x2x2 box, at voxel edges.
        let boxes =
            ArrayD::from_shape_vec(IxDyn(&[2, 3, 2]), vec![0.5, 2.5, 0.5, 3.5, 0.5, 3.5, 3.5, 5.5, 3.5, 5.5, 3.5, 5.5])
                .unwrap();
        let shapes = column(&[2, 3], vec![2, 3, 3, 2, 2, 2]);
        assert!(mask_problems(&boxes, &[0, 3, 4], &shapes, 4, Some(&[8, 8, 8])).is_empty());
        // Offsets that fall, or reach past the data.
        assert_eq!(mask_problems(&boxes, &[0, 4, 3], &shapes, 4, Some(&[8, 8, 8]))[0].code, "E408");
        assert_eq!(mask_problems(&boxes, &[0, 3, 5], &shapes, 4, Some(&[8, 8, 8]))[0].code, "E408");
        // A crop that is not its box's extent.
        let wrong = column(&[2, 3], vec![2, 3, 3, 2, 2, 3]);
        assert_eq!(mask_problems(&boxes, &[0, 3, 4], &wrong, 4, Some(&[8, 8, 8]))[0].dataset, "mask_shapes");
        // Spans that never fall but are not the packed crops: one short, then
        // one long; and one long, then one short.
        for offsets in [[0, 2, 4], [0, 4, 4]] {
            let found = mask_problems(&boxes, &offsets, &shapes, 4, Some(&[8, 8, 8]));
            assert_eq!((found[0].code, found[0].dataset.as_str()), ("E408", "mask_offsets"), "{offsets:?}");
        }
        // A crop the size of no grid: refused before anything allocates it.
        let huge =
            ArrayD::from_shape_vec(IxDyn(&[1, 3, 2]), vec![-0.5, 1048575.5, -0.5, 1048575.5, -0.5, 1023.5]).unwrap();
        let found =
            mask_problems(&huge, &[0, 1], &column(&[1, 3], vec![1 << 20, 1 << 20, 1 << 10]), 1, Some(&[8, 8, 8]));
        assert!(found[0].message.contains("more than the 512"), "{}", found[0].message);
    }

    #[test]
    fn s7_4_a_box_past_the_integers_is_a_finding_not_a_panic() {
        // Finite, so not E406; its slices are the ends of the i64 range, and
        // their difference overflowed (found by the open_validate fuzzer).
        let boxes = ArrayD::from_shape_vec(IxDyn(&[1, 1, 2]), vec![-1e300, 1e300]).unwrap();
        let found = mask_problems(&boxes, &[0, 1], &column(&[1, 1], vec![8]), 1, Some(&[8]));
        assert_eq!((found[0].code, found[0].dataset.as_str()), ("E405", "mask_shapes"));
    }
}
