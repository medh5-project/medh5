//! Measuring agreement between two annotations (§11.2).
//!
//! Three decisions are deliberate.  **Only shared, examined classes are
//! scored**: a class one rater never looked at (§11.3) contributes no
//! measurement and comes back under `skipped`.  **Empty-on-both is not a
//! disagreement**: Dice is undefined there, reported as `None` and excluded
//! from the mean.  **Instances match by id first** within one sample: where
//! both sides carry `instance_id` (§7.4) the correspondence is stated; greedy
//! IoU matching is the fallback, and the result says which was used.  An id
//! names an object within its own sample, so two files' objects always match
//! by IoU.
//!
//! Compared across two files, a class id is a number within its own label
//! set: an id the two label sets give different meanings is refused (C102)
//! rather than scored.  Two vocabularies are compared by class key
//! ([`ClassMap::Key`]) or by an explicit map of ids ([`ClassMap::Explicit`]),
//! through [`compare_with`].
//!
//! A record is keyed the way §11.2 keys it, `per_class` by class **id**, and
//! its `against` names an annotation of the recording sample (§11.2): a
//! comparison across two samples has none, and is recorded only through
//! `to_record_against`.

use std::collections::BTreeSet;

use indexmap::IndexMap;
use ndarray::{Array2, ArrayD, Axis, Zip};
use serde_json::{json, Map, Value};

use crate::annotations::{Annotation, Instance};
use crate::array::Slice;
use crate::curation::quality::Agreement;
use crate::curation::tracking::carries_instance_ids;
use crate::h5::ops;
use crate::json::{num, py_float, repr, repr_int_list, repr_int_tuple, repr_str};
use crate::labels::{ClassKey, LabelClass, LabelSet};
use crate::numeric::mean;
use crate::{Error, Result};

pub const DEFAULT_IOU: f64 = 0.5;

/// Kinds whose objects carry an axis-aligned box that IoU can be taken over.
pub const OBJECT_KINDS: [&str; 2] = ["instances", "boxes"];

fn count_true<'a>(it: impl Iterator<Item = &'a bool>) -> u64 {
    it.filter(|v| **v).count() as u64
}

/// Sørensen--Dice, or `None` when both masks are empty.
pub fn dice(a: &ndarray::ArrayD<bool>, b: &ndarray::ArrayD<bool>) -> Option<f64> {
    dice_view(a.view(), b.view())
}

/// [`dice`] of two views of one shape, which may be broadcast.
pub fn dice_view(a: ndarray::ArrayViewD<'_, bool>, b: ndarray::ArrayViewD<'_, bool>) -> Option<f64> {
    let counts = pair_counts(a, b);
    dice_of(counts.both, counts.in_a + counts.in_b)
}

/// Intersection over union, or `None` when both masks are empty.
pub fn iou(a: &ndarray::ArrayD<bool>, b: &ndarray::ArrayD<bool>) -> Option<f64> {
    iou_view(a.view(), b.view())
}

/// [`iou`] of two views of one shape, which may be broadcast.
pub fn iou_view(a: ndarray::ArrayViewD<'_, bool>, b: ndarray::ArrayViewD<'_, bool>) -> Option<f64> {
    let counts = pair_counts(a, b);
    iou_of(counts.both, counts.union)
}

/// What Dice and IoU are taken from.
struct PairCounts {
    in_a: u64,
    in_b: u64,
    both: u64,
    union: u64,
}

/// [`PairCounts`] in one pass.  Masks of two shapes are taken element by
/// element in order as far as the shorter goes, as a `zip` of their
/// iterators took them, for `both` and `union`.
fn pair_counts(a: ndarray::ArrayViewD<'_, bool>, b: ndarray::ArrayViewD<'_, bool>) -> PairCounts {
    if a.shape() != b.shape() {
        let pairs = || a.iter().zip(b.iter());
        return PairCounts {
            in_a: count_true(a.iter()),
            in_b: count_true(b.iter()),
            both: pairs().filter(|(x, y)| **x && **y).count() as u64,
            union: pairs().filter(|(x, y)| **x || **y).count() as u64,
        };
    }
    let (in_a, in_b, both) = ndarray::Zip::from(&a)
        .and(&b)
        .fold((0, 0, 0), |(x, y, z), &p, &q| (x + u64::from(p), y + u64::from(q), z + u64::from(p && q)));
    PairCounts { in_a, in_b, both, union: in_a + in_b - both }
}

/// Dice from voxel counts: `2 |A ∩ B| / (|A| + |B|)`.
fn dice_of(both: u64, total: u64) -> Option<f64> {
    (total > 0).then(|| 2.0 * both as f64 / total as f64)
}

/// IoU from voxel counts: `|A ∩ B| / |A ∪ B|`.
fn iou_of(both: u64, union: u64) -> Option<f64> {
    (union > 0).then(|| both as f64 / union as f64)
}

/// `np.prod` of a float64 run: the identity times each element in turn.
fn product(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(1.0, |acc, v| acc * v)
}

/// IoU of two `(S, 2)` boxes in the same space; NaN where there is none
/// (see [`box_iou_f64`]).
pub fn box_iou(a: &Array2<f32>, b: &Array2<f32>) -> f64 {
    box_iou_checked(a, b).unwrap_or(f64::NAN)
}

/// [`box_iou`] of float64 boxes, as a caller gives them rather than a file
/// stores them.
///
/// NaN where the two are not boxes of one space, rather than a score: boxes
/// of different numbers of axes --- a 2-D box against a 3-D one scored about
/// 10, its area over a volume --- or of none, which scored 1, and a bound that
/// is not finite or a `lo` above its `hi`, whose extent no product measures.
/// Agreement and the frontends refuse those instead.
pub fn box_iou_f64(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    box_iou_f64_checked(a, b).unwrap_or(f64::NAN)
}

/// [`box_iou`], refusing what it answers with NaN.
pub(crate) fn box_iou_checked(a: &Array2<f32>, b: &Array2<f32>) -> Result<f64> {
    box_iou_f64_checked(&a.mapv(f64::from), &b.mapv(f64::from))
}

/// [`box_iou_f64`], refusing what it answers with NaN: boxes of different
/// numbers of axes or of none (`ValueError`), and a bound that is not finite
/// or a `lo` above its `hi` (E406).
///
/// Public for the workspace's frontends, not as API: 2.0.x keeps the 2.0.1
/// signatures above.
#[doc(hidden)]
pub fn box_iou_f64_checked(a: &Array2<f64>, b: &Array2<f64>) -> Result<f64> {
    check_box(a)?;
    check_box(b)?;
    if a.nrows() != b.nrows() {
        return Err(Error::Value(format!(
            "boxes of {} and {} axes have no IoU: they are not in one space",
            a.nrows(),
            b.nrows()
        )));
    }
    let overlap = product((0..a.nrows()).map(|i| {
        let lo = a[[i, 0]].max(b[[i, 0]]);
        let hi = a[[i, 1]].min(b[[i, 1]]);
        let d = hi - lo;
        if d < 0.0 {
            0.0
        } else {
            d
        }
    }));
    if overlap == 0.0 {
        return Ok(0.0);
    }
    let volume = |m: &Array2<f64>| product(m.outer_iter().map(|r| r[1] - r[0]));
    let union = volume(a) + volume(b) - overlap;
    Ok(if union > 0.0 { overlap / union } else { 0.0 })
}

/// Refuse what is not a box: no axis, or an axis that is not a finite
/// `[lo, hi]` with `lo <= hi` (E406).
fn check_box(m: &Array2<f64>) -> Result<()> {
    if m.ncols() != 2 || m.nrows() == 0 {
        return Err(Error::Value(format!(
            "a box is an (S, 2) array with at least one axis, not shape {}",
            repr_int_tuple(m.shape())
        )));
    }
    for (axis, row) in m.outer_iter().enumerate() {
        let (lo, hi) = (row[0], row[1]);
        if !(lo.is_finite() && hi.is_finite() && lo <= hi) {
            return Err(Error::coded(
                "E406",
                format!(
                    "a box's axis {axis} spans [{}, {}]; each axis is a finite [lo, hi] with lo <= hi",
                    py_float(lo),
                    py_float(hi)
                ),
            ));
        }
    }
    Ok(())
}

/// Per-class agreement between two voxel annotations.
#[derive(Debug, Clone, PartialEq)]
pub struct VoxelAgreement {
    pub metric: String,
    pub per_class: IndexMap<String, f64>,
    /// Classes not scored: absent from one side's coverage, or empty in both.
    pub skipped: Vec<String>,
    pub against: Option<String>,
    /// `per_class` key -> class id, which is what a record is keyed by.
    pub class_ids: IndexMap<String, i64>,
}

impl VoxelAgreement {
    /// Mean over the classes that were actually comparable; `None` when none was.
    pub fn value(&self) -> Option<f64> {
        mean(&self.per_class.values().copied().collect::<Vec<_>>())
    }

    /// The [`Agreement`] a `quality` record stores (§11.2).  Refused for a
    /// comparison across two samples, which names no annotation of the
    /// recording one: [`VoxelAgreement::to_record_against`] names it.
    pub fn to_record(&self) -> Result<Agreement> {
        let against = recorded_against(self.against.as_deref())?;
        self.to_record_against(&against)
    }

    /// [`VoxelAgreement::to_record`] against `against`: the path, from the
    /// recording sample's root, of the annotation of that sample that holds
    /// what was compared against (the other rater's, copied in).
    pub fn to_record_against(&self, against: &str) -> Result<Agreement> {
        let value = measured(self.value(), &self.skipped)?;
        let mut per_class = IndexMap::new();
        for (key, score) in &self.per_class {
            per_class.insert(self.class_id(key)?.to_string(), *score);
        }
        Ok(Agreement { metric: self.metric.clone(), value, against: Some(checked_against(against)?), per_class })
    }

    fn class_id(&self, key: &str) -> Result<i64> {
        if let Some(id) = self.class_ids.get(key) {
            return Ok(*id);
        }
        if !key.is_empty() && key.chars().all(|c| c.is_ascii_digit()) {
            if let Ok(id) = key.parse() {
                return Ok(id);
            }
        }
        Err(Error::invalid(format!(
            "per-class score {} names no class id; a `quality.agreement` record is keyed by class id (§11.2)",
            repr_str(key)
        )))
    }

    /// The report, which states an undefined comparison rather than refusing.
    pub fn to_json(&self) -> Value {
        let mut out = Map::new();
        out.insert("metric".into(), json!(self.metric));
        out.insert("value".into(), self.value().map(num).unwrap_or(Value::Null));
        if let Some(against) = &self.against {
            out.insert("against".into(), json!(against));
        }
        out.insert(
            "per_class".into(),
            Value::Object(self.per_class.iter().map(|(k, v)| (k.clone(), num(*v))).collect()),
        );
        out.insert("skipped".into(), json!(self.skipped));
        out.insert("compared".into(), json!(self.per_class.len()));
        Value::Object(out)
    }
}

/// Object-level agreement: what matched, what did not, and how.
#[derive(Debug, Clone, PartialEq)]
pub struct InstanceAgreement {
    /// `(index in a, index in b, IoU)` for every matched pair.
    pub matched: Vec<(usize, usize, f64)>,
    pub only_in_a: Vec<usize>,
    pub only_in_b: Vec<usize>,
    pub matched_by: String,
    /// The IoU at or above which IoU matching pairs two objects.  Objects
    /// matched by `instance_id` are paired whatever their IoU, so the report
    /// of such a match gives no threshold (`null`).
    pub threshold: f64,
    pub against: Option<String>,
    /// `(instance_id, class in a, class in b)` --- matched but classed apart.
    pub class_mismatches: Vec<(u64, i64, i64)>,
    /// Classes whose objects were left out: not examined by both sides.
    pub skipped: Vec<String>,
}

impl InstanceAgreement {
    /// F1 over objects; `None` when neither side has an object to compare.
    pub fn value(&self) -> Option<f64> {
        let tp = self.matched.len();
        if tp + self.only_in_a.len() + self.only_in_b.len() == 0 {
            return None;
        }
        if tp == 0 {
            return Some(0.0);
        }
        let precision = tp as f64 / (tp + self.only_in_b.len()) as f64;
        let recall = tp as f64 / (tp + self.only_in_a.len()) as f64;
        Some(2.0 * precision * recall / (precision + recall))
    }

    /// Mean IoU of the matched pairs; `None` when nothing matched.
    pub fn mean_iou(&self) -> Option<f64> {
        mean(&self.matched.iter().map(|m| m.2).collect::<Vec<_>>())
    }

    /// The [`Agreement`] a `quality` record stores (§11.2), without
    /// `per_class`.  Refused for a comparison across two samples, as
    /// [`VoxelAgreement::to_record`] is.
    pub fn to_record(&self) -> Result<Agreement> {
        let against = recorded_against(self.against.as_deref())?;
        self.to_record_against(&against)
    }

    /// [`InstanceAgreement::to_record`] against `against`, an annotation of
    /// the recording sample.
    pub fn to_record_against(&self, against: &str) -> Result<Agreement> {
        Ok(Agreement {
            metric: "object_f1".into(),
            value: measured(self.value(), &self.skipped)?,
            against: Some(checked_against(against)?),
            per_class: IndexMap::new(),
        })
    }

    pub fn to_json(&self) -> Value {
        json!({
            "metric": "object_f1",
            "value": self.value().map(num),
            "mean_iou": self.mean_iou().map(num),
            "matched": self.matched.iter().map(|(i, j, v)| json!([i, j, num(*v)])).collect::<Vec<_>>(),
            "only_in_a": self.only_in_a,
            "only_in_b": self.only_in_b,
            "matched_by": self.matched_by,
            "threshold": if self.matched_by == "instance_id" { Value::Null } else { num(self.threshold) },
            "class_mismatches": self.class_mismatches.iter().map(|(i, a, b)| json!([i, a, b])).collect::<Vec<_>>(),
            "skipped": self.skipped,
            "against": self.against,
        })
    }
}

/// Either comparison, as [`compare`] chooses it.
#[derive(Debug, Clone, PartialEq)]
pub enum Comparison {
    Voxel(VoxelAgreement),
    Instance(InstanceAgreement),
}

impl Comparison {
    pub fn value(&self) -> Option<f64> {
        match self {
            Comparison::Voxel(v) => v.value(),
            Comparison::Instance(i) => i.value(),
        }
    }

    pub fn to_record(&self) -> Result<Agreement> {
        match self {
            Comparison::Voxel(v) => v.to_record(),
            Comparison::Instance(i) => i.to_record(),
        }
    }

    pub fn to_record_against(&self, against: &str) -> Result<Agreement> {
        match self {
            Comparison::Voxel(v) => v.to_record_against(against),
            Comparison::Instance(i) => i.to_record_against(against),
        }
    }

    fn skipped_mut(&mut self) -> &mut Vec<String> {
        match self {
            Comparison::Voxel(v) => &mut v.skipped,
            Comparison::Instance(i) => &mut i.skipped,
        }
    }

    pub fn to_json(&self) -> Value {
        match self {
            Comparison::Voxel(v) => v.to_json(),
            Comparison::Instance(i) => i.to_json(),
        }
    }
}

/// A value worth recording, or a refusal that says why there is none.
fn measured(value: Option<f64>, skipped: &[String]) -> Result<f64> {
    value.ok_or_else(|| {
        let reason = if skipped.is_empty() { String::new() } else { format!(" (not scored: {})", skipped.join(", ")) };
        Error::invalid(format!(
            "nothing was comparable, so there is no agreement to record{reason}; agreement on an empty comparison \
             is undefined, and recording 0 would report a disagreement nobody measured"
        ))
    })
}

fn repr_opt(value: Option<&str>) -> String {
    repr(&json!(value))
}

/// The `against` a record takes from a comparison, or the refusal of one
/// across two samples (§11.2).
fn recorded_against(against: Option<&str>) -> Result<String> {
    against.map(str::to_string).ok_or_else(|| {
        Error::invalid(
            "this comparison is across two samples, so no annotation of the recording sample holds what it was \
             compared against; an agreement's `against` names an annotation of its own sample (§11.2), so copy the \
             other annotation in and give its path as `against`",
        )
    })
}

/// An `against` path as a record stores it: `annotations/<id>`, from the
/// sample root (§11.2).
fn checked_against(against: &str) -> Result<String> {
    let path = against.trim_start_matches('/');
    match path.strip_prefix("annotations/") {
        Some(id) if !id.is_empty() && !id.contains('/') => Ok(path.to_string()),
        _ => Err(Error::invalid(format!(
            "an agreement's `against` is the path of an annotation of the recording sample, \
             'annotations/<id>' (§11.2), not {}",
            repr_str(against)
        ))),
    }
}

/// Which class of `b` each class of `a` is compared with.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ClassMap {
    /// One id names one class on both sides (one vocabulary); ids the two
    /// label sets give different meanings are refused (C102).
    #[default]
    Id,
    /// Classes of one key are compared, across two inline label sets; a key
    /// `b` lacks is skipped, and one key with two ontology codes is C102.
    Key,
    /// `a`'s class id -> `b`'s, injective; classes it does not name are not
    /// compared.
    Explicit(IndexMap<i64, i64>),
}

/// How objects are paired.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatchBy {
    /// By `instance_id` within one sample where both sides carry ids and
    /// share one, else by IoU.
    #[default]
    Auto,
    /// By IoU, whatever ids the two share.
    Iou,
    /// By `instance_id`, within one sample only: an id names an object of
    /// its own sample (§7.4).
    InstanceId,
}

impl MatchBy {
    pub fn as_str(self) -> &'static str {
        match self {
            MatchBy::Auto => "auto",
            MatchBy::Iou => "iou",
            MatchBy::InstanceId => "instance_id",
        }
    }
}

/// What [`compare_with`] takes; `Default` is [`compare`]'s defaults.
#[non_exhaustive]
#[derive(Debug, Clone, Default)]
pub struct CompareOptions {
    pub metric: Option<String>,
    pub threshold: Option<f64>,
    pub classes: Option<Vec<ClassKey>>,
    pub map_classes: ClassMap,
    pub match_by: MatchBy,
}

/// A comparison, and the classes it compared: `a`'s key -> `(id in a, id in b)`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct Compared {
    pub comparison: Comparison,
    pub pairs: IndexMap<String, (i64, i64)>,
}

struct Pair {
    /// The ids `classes` named, as `a` resolves them; `None` for all.
    asked: Option<Vec<i64>>,
    /// `(id in a, id in b)` for each class both sides examined.
    pairs: Vec<(i64, i64)>,
    skipped: Vec<String>,
}

impl Pair {
    /// `a`'s class -> `b`'s, for the compared classes.
    fn to_b(&self) -> std::collections::HashMap<i64, i64> {
        self.pairs.iter().copied().collect()
    }

    /// `a`'s key -> `(id in a, id in b)`.
    fn named(&self, a: &Annotation) -> IndexMap<String, (i64, i64)> {
        self.pairs.iter().map(|(x, y)| (a.class_key(*x), (*x, *y))).collect()
    }
}

/// Classes both sides committed to finding, in a stable order (§11.3), each
/// with the class of `b` it is compared with.
fn classes_to_compare(a: &Annotation, b: &Annotation, classes: Option<&[ClassKey]>, map: &ClassMap) -> Result<Pair> {
    let mut skipped = Vec::new();
    let (asked, candidates): (Option<Vec<i64>>, Vec<(i64, i64)>) = match map {
        ClassMap::Id => {
            let asked: Option<Vec<i64>> = match classes {
                Some(keys) => Some(keys.iter().map(|k| resolve_on_both(a, b, k)).collect::<Result<_>>()?),
                None => None,
            };
            let wanted: Vec<i64> = match &asked {
                Some(ids) => ids.clone(),
                None => sorted_ids(a.class_ids().iter().chain(b.class_ids())),
            };
            (asked, wanted.into_iter().map(|c| (c, c)).collect())
        }
        ClassMap::Key => {
            let (left, right) = inline_label_sets(a, b)?;
            let asked: Option<Vec<i64>> = match classes {
                Some(keys) => Some(keys.iter().map(|k| a.resolve_class(k)).collect::<Result<_>>()?),
                None => None,
            };
            let wanted = asked.clone().unwrap_or_else(|| sorted_ids(a.class_ids().iter()));
            let mut pairs = Vec::new();
            for id in wanted {
                let Some(class) = left.by_id(id) else {
                    skipped.push(format!("{id} (not in {})", vocabulary_of(a)));
                    continue;
                };
                match right.by_key(&class.key) {
                    None => skipped.push(format!("{} (not in {})", class.key, vocabulary_of(b))),
                    Some(other) => {
                        check_codes(class, other, a, b)?;
                        pairs.push((id, other.id));
                    }
                }
            }
            if asked.is_none() {
                for id in sorted_ids(b.class_ids().iter()) {
                    if let Some(class) = right.by_id(id) {
                        if left.by_key(&class.key).is_none() {
                            skipped.push(format!("{} (not in {})", class.key, vocabulary_of(a)));
                        }
                    }
                }
            }
            (asked, pairs)
        }
        ClassMap::Explicit(explicit) => {
            check_injective(explicit)?;
            let asked: Option<Vec<i64>> = match classes {
                Some(keys) => Some(keys.iter().map(|k| a.resolve_class(k)).collect::<Result<_>>()?),
                None => None,
            };
            let pairs = match &asked {
                None => explicit.iter().map(|(x, y)| (*x, *y)).collect(),
                Some(ids) => {
                    let mut pairs = Vec::new();
                    for id in ids {
                        match explicit.get(id) {
                            Some(other) => pairs.push((*id, *other)),
                            None => skipped.push(format!("{} (not in the class map)", a.class_key(*id))),
                        }
                    }
                    pairs
                }
            };
            (asked, pairs)
        }
    };
    let mut out = Pair { asked, pairs: Vec::new(), skipped };
    for (x, y) in candidates {
        if a.is_annotated(&ClassKey::Id(x))? && b.is_annotated(&ClassKey::Id(y))? {
            out.pairs.push((x, y));
        } else if x == y {
            out.skipped.push(format!("{} (not examined by both)", key_of(a, b, x)));
        } else {
            out.skipped.push(format!("{} (not examined by both)", a.class_key(x)));
        }
    }
    Ok(out)
}

fn sorted_ids<'a>(ids: impl Iterator<Item = &'a i64>) -> Vec<i64> {
    let mut all: Vec<i64> = ids.copied().collect();
    all.sort_unstable();
    all.dedup();
    all
}

/// Both sides' label sets, which comparing by key reads: inline ones, since
/// a label set by reference carries no classes to read keys from.
fn inline_label_sets<'a>(a: &'a Annotation, b: &'a Annotation) -> Result<(&'a LabelSet, &'a LabelSet)> {
    let inline = |ann: &'a Annotation| -> Result<&'a LabelSet> {
        match ann.label_set() {
            Some(set) if set.form == "inline" => Ok(set),
            Some(set) => Err(Error::invalid(format!(
                "comparing classes by key reads both label sets, and {} is a label set by reference ({}); \
                 compare by id, or give an explicit class map",
                vocabulary_of(ann),
                repr_opt(set.uri.as_deref())
            ))),
            None => Err(Error::invalid(format!(
                "comparing classes by key reads both label sets, and {} has none; compare by id, or give an \
                 explicit class map",
                vocabulary_of(ann)
            ))),
        }
    };
    Ok((inline(a)?, inline(b)?))
}

/// Refuse one key whose two classes name two codes in one ontology (C102):
/// matched by key, they would be scored as one class though their codes say
/// they are two.  Either side's code in an ontology both use must be on the
/// other side too, so the answer does not depend on which is `a`.
fn check_codes(left: &LabelClass, right: &LabelClass, a: &Annotation, b: &Annotation) -> Result<()> {
    let holds =
        |class: &LabelClass, system: &str, code: &str| class.codes.iter().any(|c| c.system == system && c.code == code);
    for x in &left.codes {
        for y in right.codes.iter().filter(|y| y.system == x.system) {
            if !holds(left, &y.system, &y.code) || !holds(right, &x.system, &x.code) {
                return Err(Error::coded(
                    "C102",
                    format!(
                        "class {} is {} {} in {} and {} {} in {}; one key with two codes is two classes, so it is \
                         not compared by key",
                        repr_str(&left.key),
                        x.system,
                        repr_str(&x.code),
                        vocabulary_of(a),
                        y.system,
                        repr_str(&y.code),
                        vocabulary_of(b)
                    ),
                ));
            }
        }
    }
    Ok(())
}

/// Refuse a class map that sends two classes of `a` to one of `b`, which
/// would count that class twice.
fn check_injective(map: &IndexMap<i64, i64>) -> Result<()> {
    let mut seen: std::collections::HashMap<i64, i64> = std::collections::HashMap::new();
    for (x, y) in map {
        if let Some(first) = seen.insert(*y, *x) {
            return Err(Error::invalid(format!(
                "the class map sends classes {first} and {x} to class {y}; a class map is one to one, or the \
                 class it sends two classes to is counted twice"
            )));
        }
    }
    Ok(())
}

/// A class id's key in the label set of a side that uses it: an id only `b`
/// uses was named by `a`'s label set, which across two files may give it
/// another meaning.
fn key_of(a: &Annotation, b: &Annotation, class_id: i64) -> String {
    let uses = |ann: &Annotation| ann.class_ids().contains(&class_id) || ann.annotated_class_ids().contains(&class_id);
    if uses(a) || !uses(b) {
        a.class_key(class_id)
    } else {
        b.class_key(class_id)
    }
}

/// A class key as the one id both annotations give it.  A name resolves
/// through `a`'s label set, and through `b`'s as well where that holds it,
/// and is refused (C102) where the two give it two ids: "liver" is a number
/// only within its own label set.  A name `b`'s label set lacks leaves its
/// id to [`check_meanings`].
fn resolve_on_both(a: &Annotation, b: &Annotation, key: &ClassKey) -> Result<i64> {
    let id = a.resolve_class(key)?;
    if let (ClassKey::Key(name), Some(set)) = (key, b.label_set()) {
        let Some(other) = set.by_key(name).map(|c| c.id) else { return Ok(id) };
        if other != id {
            return Err(Error::coded(
                "C102",
                format!(
                    "class {} is id {id} in {} and id {other} in {}; agreement compares one class id on both \
                     sides, and these are two",
                    repr_str(name),
                    vocabulary_of(a),
                    vocabulary_of(b)
                ),
            ));
        }
    }
    Ok(id)
}

/// Refuse class ids, of those both sides use, that the two annotations'
/// label sets give different meanings (C102, the rule `dataset check`
/// applies to a cohort).
///
/// A class id is a number within its own label set: compared across two
/// files, "liver" in one was scored against "spleen" in the other, at Dice 1.
/// One label set --- one sample's, or one vocabulary in two files --- means
/// the same throughout, and two versions of a vocabulary that agree on a
/// class compare it.  Where either side has no label set, an id is only a
/// number, and is compared as one.
fn check_meanings(a: &Annotation, b: &Annotation, ids: &BTreeSet<i64>) -> Result<()> {
    let (Some(left), Some(right)) = (a.label_set(), b.label_set()) else { return Ok(()) };
    if std::ptr::eq(left, right) || left.sha256() == right.sha256() {
        return Ok(());
    }
    let key = |set: &LabelSet, id: i64| set.by_id(id).map(|c| repr_str(&c.key));
    let clashing: Vec<(i64, Option<String>, Option<String>)> =
        ids.iter().map(|id| (*id, key(left, *id), key(right, *id))).filter(|(_, x, y)| x.is_none() || x != y).collect();
    if clashing.is_empty() {
        return Ok(());
    }
    let undefined = || "undefined".to_string();
    let what: Vec<String> = clashing
        .iter()
        .map(|(id, x, y)| {
            format!(
                "{id} is {} in the first and {} in the second",
                x.clone().unwrap_or_else(undefined),
                y.clone().unwrap_or_else(undefined)
            )
        })
        .collect();
    let ids: Vec<i64> = clashing.iter().map(|c| c.0).collect();
    Err(Error::coded(
        "C102",
        format!(
            "class id(s) {} mean different things in {} and {}: {}; scored across the two, one class is measured \
             against another",
            repr_int_list(&ids),
            vocabulary_of(a),
            vocabulary_of(b),
            what.join("; ")
        ),
    ))
}

/// `annotation 'seg' of 'a.medh5' (label set 'v' 1.0.0)`: what gives the
/// annotation's class ids their meaning, and where.  Two raters' files
/// often call their annotations alike.
fn vocabulary_of(ann: &Annotation) -> String {
    let set = match ann.label_set() {
        Some(set) => format!("label set {} {}", repr_str(&set.id), set.version),
        None => "no label set".into(),
    };
    format!("annotation {} of {} ({set})", repr_str(&ann.ann_id), repr_str(&ann.group.filename()))
}

/// Per-class Dice or IoU between two voxel annotations on the same grid.
pub fn compare_voxel(
    a: &Annotation,
    b: &Annotation,
    metric: &str,
    classes: Option<&[ClassKey]>,
) -> Result<VoxelAgreement> {
    Ok(voxel_with(a, b, metric, classes, &ClassMap::Id)?.0)
}

/// [`compare_voxel`] under a class map, with the pairs it compared.
fn voxel_with(
    a: &Annotation,
    b: &Annotation,
    metric: &str,
    classes: Option<&[ClassKey]>,
    map: &ClassMap,
) -> Result<(VoxelAgreement, Pair)> {
    if metric != "dice" && metric != "iou" {
        return Err(Error::invalid(format!("unknown agreement metric {}", repr_str(metric))));
    }
    if !on_one_grid(a, b) {
        return Err(Error::coded(
            "E101",
            format!(
                "annotations {} and {} are on different grids ({} vs {}){}; resample one onto the other's \
                 grid before comparing (`medh5 resample PATH --image ID --onto GRID` carries the \
                 annotations with the images)",
                repr_str(&a.ann_id),
                repr_str(&b.ann_id),
                repr_opt(a.grid_id()),
                repr_opt(b.grid_id()),
                if one_sample(a, b) { "" } else { ACROSS_SAMPLES }
            ),
        ));
    }
    let pair = classes_to_compare(a, b, classes, map)?;
    if *map == ClassMap::Id {
        check_meanings(a, b, &shared_ids(a, &[], b, &[], pair.asked.as_deref()))?;
    }
    // A voxel either annotation ignores (§7.7) is evidence neither for nor
    // against agreement, so it is left out of both: counted, it scored a rater
    // against voxels the other declared unexamined.
    let ignored = match (a.ignore_region()?, b.ignore_region()?) {
        (Some(mut left), Some(right)) => {
            same_shape(&left, &right, a, b)?;
            left.zip_mut_with(&right, |l, r| *l |= *r);
            Some(left)
        }
        (left, right) => left.or(right),
    };
    let counts = class_counts(a, b, &pair.pairs, ignored.as_ref())?;
    let mut per_class = IndexMap::new();
    let mut ids = IndexMap::new();
    let mut skipped = pair.skipped.clone();
    for ((class_id, _), (both, left, right)) in pair.pairs.iter().copied().zip(counts) {
        let score = if metric == "dice" { dice_of(both, left + right) } else { iou_of(both, left + right - both) };
        let key = a.class_key(class_id);
        match score {
            None if ignored.is_some() => skipped.push(format!("{key} (empty in both outside the ignore region)")),
            None => skipped.push(format!("{key} (empty in both)")),
            Some(s) => {
                per_class.insert(key.clone(), s);
                ids.insert(key, class_id);
            }
        }
    }
    let found = VoxelAgreement { metric: metric.into(), per_class, skipped, against: against_of(a, b), class_ids: ids };
    Ok((found, pair))
}

/// What a comparison is against: `b`'s path where it is an annotation of
/// `a`'s sample, else nothing --- recorded in `a`, a path of `b`'s sample
/// names an annotation of `a`'s, or none (§11.2).
fn against_of(a: &Annotation, b: &Annotation) -> Option<String> {
    one_sample(a, b).then(|| format!("annotations/{}", b.ann_id))
}

/// Said of two grids in two samples that do not count one lattice.
const ACROSS_SAMPLES: &str = " --- in two samples, whose grid ids are each their own, grids are one only when they \
                              share a declared frame and are one lattice in it";

/// Whether two annotations were read from one sample, whose grid ids name
/// one set of grids: through one handle, or two on one sample root of one
/// file.  HDF5 shares one open file between two handles, so they report one
/// file number (an amended file is a new inode, and another file); a
/// collection holds many samples in one file, so the root is compared too.
fn one_sample(a: &Annotation, b: &Annotation) -> bool {
    std::ptr::eq(a.grids(), b.grids()) || matches!((sample_root(a), sample_root(b)), (Some(x), Some(y)) if x == y)
}

/// The file number and root path of the sample an annotation was read from:
/// `<root>/annotations/<id>`.
fn sample_root(ann: &Annotation) -> Option<(std::os::raw::c_ulong, String)> {
    let (fileno, _) = ops::object_id_of(&ann.group)?;
    let path = ann.group.name();
    let root = path.rsplit_once('/')?.0.rsplit_once('/')?.0;
    Some((fileno, root.to_string()))
}

/// Whether two annotations count voxels of one grid: within one sample, one
/// grid id; across two, grids physically comparable (§3.3 rule 4: one
/// declared `frame_uid`, one `coord_system`) that are one lattice --- shape,
/// axes, spacing, origin and direction --- in one unit.
///
/// A grid id is a sample's own name.  Two samples each calling a grid `g`
/// were compared voxel for voxel though one was shifted 100 mm, and their
/// ignore regions merged though their shapes differed, which panicked.
fn on_one_grid(a: &Annotation, b: &Annotation) -> bool {
    if one_sample(a, b) {
        return a.grid_id() == b.grid_id();
    }
    let (Ok(ga), Ok(gb)) = (a.grid(), b.grid()) else { return false };
    ga.comparable_with(gb) && ga.is_congruent(gb, 1e-6) && ga.units == gb.units
}

/// Refuse two voxel arrays that do not cover one grid, rather than let an
/// elementwise merge of them panic (a sibling ignore mask on another grid,
/// say).
fn same_shape(x: &ArrayD<bool>, y: &ArrayD<bool>, a: &Annotation, b: &Annotation) -> Result<()> {
    if x.shape() == y.shape() {
        return Ok(());
    }
    Err(Error::coded(
        "E405",
        format!(
            "annotations {} and {} cover voxels of shapes {} and {}; voxel agreement compares one grid's voxels",
            repr_str(&a.ann_id),
            repr_str(&b.ann_id),
            repr_int_tuple(x.shape()),
            repr_int_tuple(y.shape())
        ),
    ))
}

/// Class-plane voxels [`class_counts`] holds in memory at once, per side.
const SLAB_VOXELS: usize = 1 << 26;

/// `(|A ∩ B|, |A|, |B|)` for each of `classes`, over the voxels neither side
/// ignores.
///
/// Classes are read together, through the reader contract (`dense`), in
/// slabs of whole stored chunks: read one class at a time, every label map or
/// layer was decoded once for each class it holds, and a 100-class
/// comparison read each volume 100 times.  The planes held at once stay
/// under [`SLAB_VOXELS`] a side; where a slab of every class would not, the
/// classes are read in groups.
fn class_counts(
    a: &Annotation,
    b: &Annotation,
    classes: &[(i64, i64)],
    ignored: Option<&ArrayD<bool>>,
) -> Result<Vec<(u64, u64, u64)>> {
    counts_in_slabs(a, b, classes, ignored, SLAB_VOXELS)
}

/// [`class_counts`] holding at most `budget` class-plane voxels a side, or
/// one class's plane of one chunk where that is more.
fn counts_in_slabs(
    a: &Annotation,
    b: &Annotation,
    classes: &[(i64, i64)],
    ignored: Option<&ArrayD<bool>>,
    budget: usize,
) -> Result<Vec<(u64, u64, u64)>> {
    let mut counts = vec![(0, 0, 0); classes.len()];
    let shape = a.spatial_shape()?;
    let other = b.spatial_shape()?;
    if shape != other {
        return Err(Error::coded(
            "E405",
            format!(
                "annotations {} and {} cover voxels of shapes {} and {}; voxel agreement compares one grid's voxels",
                repr_str(&a.ann_id),
                repr_str(&b.ann_id),
                repr_int_tuple(&shape),
                repr_int_tuple(&other)
            ),
        ));
    }
    if let Some(region) = ignored {
        if region.shape() != shape.as_slice() {
            return Err(Error::coded(
                "E405",
                format!(
                    "annotations {} and {}: an ignore region of shape {} on a grid of shape {}; voxel agreement \
                     compares one grid's voxels",
                    repr_str(&a.ann_id),
                    repr_str(&b.ann_id),
                    repr_int_tuple(region.shape()),
                    repr_int_tuple(&shape)
                ),
            ));
        }
    }
    let Some((&depth, rest)) = shape.split_first() else { return Ok(counts) };
    if classes.is_empty() || depth == 0 {
        return Ok(counts);
    }
    let plane = rest.iter().product::<usize>().max(1);
    let chunk = match chunk_rows(a).max(chunk_rows(b)) {
        0 => depth,
        rows => rows.min(depth),
    };
    let slab = chunk.saturating_mul(plane);
    let (rows, group) = match classes.len().saturating_mul(slab) {
        all if all <= budget => (chunk.saturating_mul(budget / all), classes.len()),
        _ => (chunk, (budget / slab).max(1)),
    };
    for start in (0..depth).step_by(rows) {
        let stop = (start + rows).min(depth);
        let mut roi = vec![Slice::new(start as i64, stop as i64)];
        roi.extend(rest.iter().map(|_| Slice::full()));
        let kept = ignored.map(|region| region.slice_axis(Axis(0), ndarray::Slice::from(start..stop)));
        for (ids, tallies) in classes.chunks(group).zip(counts.chunks_mut(group)) {
            let keys_a: Vec<ClassKey> = ids.iter().map(|c| ClassKey::Id(c.0)).collect();
            let keys_b: Vec<ClassKey> = ids.iter().map(|c| ClassKey::Id(c.1)).collect();
            let left = a.dense(Some(&keys_a), Some(&roi))?;
            let right = b.dense(Some(&keys_b), Some(&roi))?;
            same_shape(&left, &right, a, b)?;
            for (i, tally) in tallies.iter_mut().enumerate() {
                let (x, y) = (left.index_axis(Axis(0), i), right.index_axis(Axis(0), i));
                let mut add = |p: bool, q: bool| {
                    tally.0 += u64::from(p && q);
                    tally.1 += u64::from(p);
                    tally.2 += u64::from(q);
                };
                match &kept {
                    Some(skip) => Zip::from(&x).and(&y).and(skip).for_each(|p, q, s| {
                        if !*s {
                            add(*p, *q);
                        }
                    }),
                    None => Zip::from(&x).and(&y).for_each(|p, q| add(*p, *q)),
                }
            }
        }
    }
    Ok(counts)
}

/// The extent of `ann`'s stored chunks along the first spatial axis, or 0
/// where it stores none to read (an `instances` annotation, contiguous data).
fn chunk_rows(ann: &Annotation) -> usize {
    let Ok(Some(data)) = ann.optional_dataset("data") else { return 0 };
    let stacked = usize::from(matches!(ann.kind(), "layers" | "bitmask" | "probmap"));
    data.chunk().and_then(|c| c.get(stacked).copied()).unwrap_or(0)
}

fn space_of(ann: &Annotation) -> String {
    ann.header.space.clone().filter(|s| !s.is_empty()).unwrap_or_else(|| "index".into())
}

fn frame_of(ann: &Annotation) -> Option<String> {
    if let Some(f) = &ann.header.frame_uid {
        return Some(f.clone());
    }
    ann.grid().ok().and_then(|g| g.frame_uid.clone())
}

/// Refuse two coordinate systems that cannot be compared number for number.
fn check_same_space(a: &Annotation, b: &Annotation) -> Result<()> {
    let (space_a, space_b) = (space_of(a), space_of(b));
    if space_a != space_b {
        return Err(Error::coded(
            "E414",
            format!(
                "annotations {} and {} store boxes in different spaces ({} vs {}); convert one before comparing",
                repr_str(&a.ann_id),
                repr_str(&b.ann_id),
                repr_str(&space_a),
                repr_str(&space_b)
            ),
        ));
    }
    if space_a == "world" {
        if a.grid_id().is_some() && on_one_grid(a, b) {
            return Ok(());
        }
        let (frame_a, frame_b) = (frame_of(a), frame_of(b));
        if frame_a.is_some() && frame_a == frame_b {
            return Ok(());
        }
        return Err(Error::coded(
            "E414",
            format!(
                "annotations {} and {} are in frames {} and {}; a transform is required to relate them",
                repr_str(&a.ann_id),
                repr_str(&b.ann_id),
                repr_opt(frame_a.as_deref()),
                repr_opt(frame_b.as_deref())
            ),
        ));
    }
    if !on_one_grid(a, b) {
        return Err(Error::coded(
            "E101",
            format!(
                "annotations {} and {} are on different grids ({} vs {}){}; their index coordinates count \
                 different voxels, so resample one onto the other's grid before comparing (`medh5 \
                 resample PATH --image ID --onto GRID` carries the annotations with the images)",
                repr_str(&a.ann_id),
                repr_str(&b.ann_id),
                repr_opt(a.grid_id()),
                repr_opt(b.grid_id()),
                if one_sample(a, b) { "" } else { ACROSS_SAMPLES }
            ),
        ));
    }
    Ok(())
}

/// Object-level agreement between two instance-carrying annotations.
///
/// **A miss counts only where the other side looked** (§11.3): an unmatched
/// object whose class the other annotation never examined is dropped and its
/// class listed under `skipped`.  Two objects match at an IoU of at least
/// `threshold`, which lies in (0, 1].
pub fn compare_instances(
    a: &Annotation,
    b: &Annotation,
    threshold: f64,
    classes: Option<&[ClassKey]>,
) -> Result<InstanceAgreement> {
    Ok(instances_with(a, b, threshold, classes, &ClassMap::Id, MatchBy::Auto)?.0)
}

/// [`compare_instances`] under a class map and a way of pairing objects,
/// with the class pairs it compared.
fn instances_with(
    a: &Annotation,
    b: &Annotation,
    threshold: f64,
    classes: Option<&[ClassKey]>,
    map: &ClassMap,
    match_by: MatchBy,
) -> Result<(InstanceAgreement, Pair)> {
    // As `medh5 agree --threshold` holds it: NaN matched no object, and 0 or
    // less every pair, objects that do not touch among them.
    let in_range = threshold > 0.0 && threshold <= 1.0;
    if !in_range {
        return Err(Error::invalid(format!(
            "threshold {} is no IoU threshold: two objects match at an IoU in (0, 1]",
            py_float(threshold)
        )));
    }
    for ann in [a, b] {
        if !OBJECT_KINDS.contains(&ann.kind()) {
            return Err(Error::invalid(format!(
                "annotation {} is {}; object agreement needs objects with boxes ({}). Compare voxel annotations \
                 with `compare_voxel`",
                repr_str(&ann.ann_id),
                repr_str(ann.kind()),
                OBJECT_KINDS.join(", ")
            )));
        }
    }
    check_same_space(a, b)?;
    check_axes(a, b)?;
    if match_by == MatchBy::InstanceId && !one_sample(a, b) {
        return Err(Error::invalid(format!(
            "annotations {} and {} are in two samples, and an instance_id names an object of its own sample \
             (§7.4): two files numbering from 1 share ids, not objects; match by IoU",
            repr_str(&a.ann_id),
            repr_str(&b.ann_id)
        )));
    }
    let pair = classes_to_compare(a, b, classes, map)?;
    let mut left = a.instances()?;
    let mut right = b.instances()?;
    in_the_world_of(a, b, &mut right)?;
    if *map == ClassMap::Id {
        if let Some(asked) = &pair.asked {
            left.retain(|o| asked.contains(&o.class_id));
            right.retain(|o| asked.contains(&o.class_id));
        }
        check_meanings(a, b, &shared_ids(a, &left, b, &right, pair.asked.as_deref()))?;
    } else {
        // Only the mapped classes are compared, each of b's objects under
        // the class of a it maps from.
        let to_a: std::collections::HashMap<i64, i64> = pair.pairs.iter().map(|(x, y)| (*y, *x)).collect();
        left.retain(|o| pair.pairs.iter().any(|(x, _)| *x == o.class_id));
        right.retain(|o| to_a.contains_key(&o.class_id));
        for object in &mut right {
            object.class_id = to_a[&object.class_id];
        }
    }
    check_boxes(a, &left)?;
    check_boxes(b, &right)?;
    // `instance_id` names an object within its own sample (§7.4), so only
    // there does a shared id state a correspondence: two raters' files that
    // each number from 1 paired disjoint lesions as matches, at an F1 of 1.
    let ids_b: std::collections::HashSet<u64> = right.iter().map(|o| o.instance_id).collect();
    let shared: std::collections::HashSet<u64> =
        left.iter().map(|o| o.instance_id).filter(|i| ids_b.contains(i)).collect();
    let identified = carries_instance_ids(a)? && carries_instance_ids(b)?;
    let by_id = match match_by {
        MatchBy::Iou => false,
        MatchBy::InstanceId if !identified => {
            return Err(Error::invalid(format!(
                "matching by instance_id needs ids on both sides, and {} carries none",
                repr_str(if carries_instance_ids(a)? { &b.ann_id } else { &a.ann_id })
            )));
        }
        MatchBy::InstanceId => true,
        MatchBy::Auto => one_sample(a, b) && !shared.is_empty() && identified,
    };
    let against = against_of(a, b);
    let mut result = if by_id {
        match_by_id(&left, &right, &shared, threshold, against)?
    } else {
        match_by_iou(&left, &right, threshold, against)?
    };
    let class_a: std::collections::HashMap<usize, i64> = left.iter().map(|o| (o.index, o.class_id)).collect();
    let class_b: std::collections::HashMap<usize, i64> = right.iter().map(|o| (o.index, o.class_id)).collect();
    // Coverage is each side's own: a's class, as b names it, in b's.
    let to_b = pair.to_b();
    let in_b = |c: i64| if *map == ClassMap::Id { c } else { to_b.get(&c).copied().unwrap_or(c) };
    let looked_a = a.annotated_class_ids();
    let looked_b = b.annotated_class_ids();
    result.only_in_a.retain(|i| looked_b.contains(&in_b(class_a[i])));
    result.only_in_b.retain(|j| looked_a.contains(&class_b[j]));
    result.skipped = pair.skipped.clone();
    Ok((result, pair))
}

/// Refuse boxes of different numbers of spatial axes (E414).  A grid's
/// `direction` is S×S (§3.2), so a 2-D grid's world is a plane of its own,
/// not a slice of a volume's: scored as one, a 2-D box against a 3-D one
/// came out at an IoU of about 10, and as a match.
fn check_axes(a: &Annotation, b: &Annotation) -> Result<()> {
    let (left, right) = (axes_of(a)?, axes_of(b)?);
    if left == right {
        return Ok(());
    }
    Err(Error::coded(
        "E414",
        format!(
            "annotations {} and {} hold boxes of {left} and {right} spatial axes, which are not in one space",
            repr_str(&a.ann_id),
            repr_str(&b.ann_id)
        ),
    ))
}

/// The spatial axes of an object annotation's `(N, S, 2)` boxes.
fn axes_of(ann: &Annotation) -> Result<usize> {
    Ok(ann.dataset_shape("boxes")?.get(1).copied().unwrap_or(0))
}

/// Refuse an object whose box IoU cannot measure (E406), naming it: a bound
/// that is not finite or a `lo` above its `hi`, which scored as an empty box
/// or as a box with the volume of its inversion.
fn check_boxes(ann: &Annotation, objects: &[Instance]) -> Result<()> {
    for object in objects {
        check_box(&object.bbox.mapv(f64::from)).map_err(|e| {
            Error::coded(
                "E406",
                format!("annotation {}: object {}: {}", repr_str(&ann.ann_id), object.index, e.message()),
            )
        })?;
    }
    Ok(())
}

/// The class ids both sides use, of those asked for: what a comparison may
/// read on both, so what must mean one thing on both.  A side uses the ids
/// it can express (`class_ids`), those it examined, and those its objects
/// carry.
fn shared_ids(
    a: &Annotation,
    left: &[Instance],
    b: &Annotation,
    right: &[Instance],
    asked: Option<&[i64]>,
) -> BTreeSet<i64> {
    let used = |ann: &Annotation, objects: &[Instance]| -> BTreeSet<i64> {
        ann.class_ids()
            .iter()
            .chain(ann.annotated_class_ids())
            .copied()
            .chain(objects.iter().map(|o| o.class_id))
            .filter(|c| asked.is_none_or(|ids| ids.contains(c)))
            .collect()
    };
    used(a, left).intersection(&used(b, right)).copied().collect()
}

/// `b`'s world boxes, in `a`'s units.
///
/// A shared frame says two annotations' world is one place, not that their
/// numbers are in one unit: the same boxes in metres and in millimetres scored
/// an F1 of 0, and boxes in LPS against RAS an IoU of 1 where they were
/// disjoint --- which E414 now refuses, since §3.3
/// rule 4 compares world coordinates in one `coord_system` only.  An
/// annotation that names no grid states neither, and is taken as it is.
fn in_the_world_of(a: &Annotation, b: &Annotation, objects: &mut [Instance]) -> Result<()> {
    if space_of(a) != "world" || on_one_grid(a, b) {
        return Ok(());
    }
    let (Some(to), Some(from)) = (a.world_grid(), b.world_grid()) else { return Ok(()) };
    let scale = from.world_scale_into(to).map_err(|e| {
        Error::coded(
            "E414",
            format!("annotations {} and {}: {}", repr_str(&a.ann_id), repr_str(&b.ann_id), e.message()),
        )
    })?;
    if scale != 1.0 {
        for object in objects {
            object.bbox.mapv_inplace(|v| (f64::from(v) * scale) as f32);
        }
    }
    Ok(())
}

fn match_by_id(
    left: &[Instance],
    right: &[Instance],
    shared: &std::collections::HashSet<u64>,
    threshold: f64,
    against: Option<String>,
) -> Result<InstanceAgreement> {
    let by_id_b: std::collections::HashMap<u64, &Instance> = right.iter().map(|o| (o.instance_id, o)).collect();
    let mut matched = Vec::new();
    let mut mismatches = Vec::new();
    for obj in left.iter().filter(|o| shared.contains(&o.instance_id)) {
        let other = by_id_b[&obj.instance_id];
        matched.push((obj.index, other.index, box_iou_checked(&obj.bbox, &other.bbox)?));
        if obj.class_id != other.class_id {
            mismatches.push((obj.instance_id, obj.class_id, other.class_id));
        }
    }
    Ok(InstanceAgreement {
        matched,
        only_in_a: left.iter().filter(|o| !shared.contains(&o.instance_id)).map(|o| o.index).collect(),
        only_in_b: right.iter().filter(|o| !shared.contains(&o.instance_id)).map(|o| o.index).collect(),
        matched_by: "instance_id".into(),
        threshold,
        against,
        class_mismatches: mismatches,
        skipped: Vec::new(),
    })
}

/// Greedy highest-IoU-first matching, one object to at most one object.
fn match_by_iou(
    left: &[Instance],
    right: &[Instance],
    threshold: f64,
    against: Option<String>,
) -> Result<InstanceAgreement> {
    let mut candidates: Vec<(f64, usize, usize)> = Vec::new();
    for (i, obj) in left.iter().enumerate() {
        for (j, other) in right.iter().enumerate() {
            if obj.class_id != other.class_id {
                continue;
            }
            let overlap = box_iou_checked(&obj.bbox, &other.bbox)?;
            if overlap >= threshold {
                candidates.push((overlap, i, j));
            }
        }
    }
    candidates.sort_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)).then(x.2.cmp(&y.2)));
    let mut used_a = std::collections::HashSet::new();
    let mut used_b = std::collections::HashSet::new();
    let mut matched = Vec::new();
    for (overlap, i, j) in candidates {
        if used_a.contains(&i) || used_b.contains(&j) {
            continue;
        }
        used_a.insert(i);
        used_b.insert(j);
        matched.push((left[i].index, right[j].index, overlap));
    }
    matched.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.cmp(&y.1)).then(x.2.total_cmp(&y.2)));
    Ok(InstanceAgreement {
        matched,
        only_in_a: left.iter().enumerate().filter(|(i, _)| !used_a.contains(i)).map(|(_, o)| o.index).collect(),
        only_in_b: right.iter().enumerate().filter(|(j, _)| !used_b.contains(j)).map(|(_, o)| o.index).collect(),
        matched_by: "iou".into(),
        threshold,
        against,
        class_mismatches: Vec::new(),
        skipped: Vec::new(),
    })
}

/// Compare two annotations, choosing the comparison their kinds support.
///
/// Two object-carrying annotations are compared object by object at an IoU
/// `threshold`; two voxel annotations class by class, by `metric`.  An
/// argument the chosen comparison cannot use is refused rather than ignored.
pub fn compare(
    a: &Annotation,
    b: &Annotation,
    metric: Option<&str>,
    threshold: Option<f64>,
    classes: Option<&[ClassKey]>,
) -> Result<Comparison> {
    let options = CompareOptions {
        metric: metric.map(str::to_string),
        threshold,
        classes: classes.map(<[ClassKey]>::to_vec),
        ..CompareOptions::default()
    };
    Ok(compare_with(a, b, &options)?.comparison)
}

/// [`compare`] with every option: which class of `b` each class of `a` is
/// compared with ([`ClassMap`]), and how objects are paired ([`MatchBy`]).
/// The result names the class pairs it compared.  A skipped class is listed
/// once.
pub fn compare_with(a: &Annotation, b: &Annotation, options: &CompareOptions) -> Result<Compared> {
    let classes = options.classes.as_deref();
    let (comparison, pair) = if OBJECT_KINDS.contains(&a.kind()) && OBJECT_KINDS.contains(&b.kind()) {
        if let Some(m) = &options.metric {
            return Err(Error::invalid(format!(
                "metric {} scores voxels; {} and {} are compared object by object, as F1 at an IoU threshold",
                repr_str(m),
                repr_str(&a.ann_id),
                repr_str(&b.ann_id)
            )));
        }
        let threshold = options.threshold.unwrap_or(DEFAULT_IOU);
        let (found, pair) = instances_with(a, b, threshold, classes, &options.map_classes, options.match_by)?;
        (Comparison::Instance(found), pair)
    } else if a.is_voxel() && b.is_voxel() {
        if options.threshold.is_some() || options.match_by != MatchBy::Auto {
            return Err(Error::invalid(format!(
                "{} matches objects; {} and {} are compared voxel by voxel",
                if options.threshold.is_some() { "threshold" } else { "match_by" },
                repr_str(&a.ann_id),
                repr_str(&b.ann_id)
            )));
        }
        let metric = options.metric.as_deref().unwrap_or("dice");
        let (found, pair) = voxel_with(a, b, metric, classes, &options.map_classes)?;
        (Comparison::Voxel(found), pair)
    } else {
        return Err(Error::invalid(format!(
            "cannot compare {} with {}: transcode them to a common kind first",
            repr_str(a.kind()),
            repr_str(b.kind())
        )));
    };
    Ok(compared(comparison, &pair, a))
}

/// [`compare_voxel`] with [`compare_with`]'s options (`metric`, `classes`,
/// `map_classes`); a threshold or a way of pairing objects is refused.
pub fn compare_voxel_with(a: &Annotation, b: &Annotation, options: &CompareOptions) -> Result<Compared> {
    if options.threshold.is_some() || options.match_by != MatchBy::Auto {
        return Err(Error::invalid("a threshold and match_by pair objects; voxel agreement compares voxels"));
    }
    let metric = options.metric.as_deref().unwrap_or("dice");
    let (found, pair) = voxel_with(a, b, metric, options.classes.as_deref(), &options.map_classes)?;
    Ok(compared(Comparison::Voxel(found), &pair, a))
}

/// [`compare_instances`] with [`compare_with`]'s options (`threshold`,
/// `classes`, `map_classes`, `match_by`); a metric is refused.
pub fn compare_instances_with(a: &Annotation, b: &Annotation, options: &CompareOptions) -> Result<Compared> {
    if let Some(metric) = &options.metric {
        return Err(Error::invalid(format!("metric {} scores voxels; objects are compared as F1", repr_str(metric))));
    }
    let threshold = options.threshold.unwrap_or(DEFAULT_IOU);
    let (found, pair) =
        instances_with(a, b, threshold, options.classes.as_deref(), &options.map_classes, options.match_by)?;
    Ok(compared(Comparison::Instance(found), &pair, a))
}

/// A comparison with its class pairs, each skipped class listed once.
fn compared(mut comparison: Comparison, pair: &Pair, a: &Annotation) -> Compared {
    let skipped = comparison.skipped_mut();
    let mut seen = BTreeSet::new();
    skipped.retain(|s| seen.insert(s.clone()));
    Compared { comparison, pairs: pair.named(a) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn box_iou_is_zero_without_overlap_and_one_for_identity() {
        let a: Array2<f32> = array![[0.0, 2.0], [0.0, 2.0]];
        let b: Array2<f32> = array![[2.0, 4.0], [0.0, 2.0]];
        assert_eq!(box_iou(&a, &b), 0.0);
        assert_eq!(box_iou(&a, &a), 1.0);
        let c: Array2<f32> = array![[1.0, 3.0], [0.0, 2.0]];
        assert_eq!(box_iou(&a, &c), 2.0 / 6.0);
    }

    #[test]
    fn s8_1_box_iou_refuses_what_is_not_two_boxes_of_one_space() {
        let plane: Array2<f64> = array![[0.0, 1.0], [0.0, 1.0]];
        let volume: Array2<f64> = array![[0.0, 1.0], [0.0, 1.0], [0.0, 0.1]];
        // The public form keeps its 2.0.1 signature and answers NaN for
        // exactly what the checked form refuses.
        let both = |a: &Array2<f64>, b: &Array2<f64>| {
            let checked = box_iou_f64_checked(a, b);
            let public = box_iou_f64(a, b);
            match &checked {
                Ok(v) => assert_eq!(public, *v),
                Err(_) => assert!(public.is_nan()),
            }
            checked
        };
        // An area over a volume scored about 10, in both orders.
        assert!(both(&plane, &volume).is_err());
        assert!(both(&volume, &plane).is_err());
        // No axis at all scored 1.
        let none = Array2::<f64>::zeros((0, 2));
        assert!(both(&none, &none).is_err());
        for bad in [[f64::NAN, 1.0], [0.0, f64::INFINITY], [f64::NEG_INFINITY, 0.0], [1.0, 0.0]] {
            let broken = array![[bad[0], bad[1]], [0.0, 1.0]];
            assert_eq!(both(&broken, &plane).unwrap_err().code(), Some("E406"), "{bad:?}");
            assert_eq!(both(&plane, &broken).unwrap_err().code(), Some("E406"), "{bad:?}");
        }
        // Boxes of one space still score, a flat one as no overlap.
        let flat: Array2<f64> = array![[0.0, 0.0], [0.0, 1.0]];
        assert_eq!(both(&flat, &flat).unwrap(), 0.0);
        assert_eq!(both(&volume, &volume).unwrap(), 1.0);
    }

    /// Counted in slabs of chunks and in groups of classes, however small the
    /// budget, two annotations agree voxel for voxel with their classes read
    /// one at a time over the whole grid.
    #[test]
    fn s11_2_counts_in_slabs_are_the_counts_of_whole_planes() {
        use crate::array::NdArray;
        use crate::labels::LabelClass;
        use crate::rng::Rng;
        use crate::sample::open_sample;
        use crate::sample::writer::{create, GridOptions, ImageOptions};
        use crate::sample::writer_annotations::{SegmentationOptions, SegmentationSource};
        use ndarray::IxDyn;

        // Above the size a dataset is stored whole in (64 KiB), chunked in
        // three rows as the grid asks.
        let (shape, classes) = ([16_usize, 72, 64], 5_i64);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("raters.medh5");
        let mut rng = Rng::new(11);
        let mut labelling = || ArrayD::from_shape_simple_fn(IxDyn(&shape), || rng.integer(0, classes + 1).unwrap());
        let (first, second) = (labelling(), labelling());
        let masks = |labels: &ArrayD<i64>| -> Vec<(ClassKey, ArrayD<bool>)> {
            (1..=classes).map(|c| (ClassKey::Id(c), labels.mapv(|v| v == c))).collect()
        };
        let mut w = create(&path, Some("raters"), None, "portable", &[]).unwrap();
        let named = (1..=classes).map(|c| LabelClass::new(c, format!("c{c}"), format!("C{c}")).unwrap()).collect();
        w.label_set(LabelSet::new("v", named, "1.0.0", vec![], vec![], "inline", None, None).unwrap());
        let grid = GridOptions { chunk_hint: Some(vec![3, 72, 64]), ..Default::default() };
        w.add_grid("g", &[16, 72, 64], &[1.0, 1.0, 1.0], grid).unwrap();
        let image = NdArray::from_vec(&shape, vec![0_i16; shape.iter().product()]).unwrap();
        w.add_image("CT", &image, "g", "CT", ImageOptions::default()).unwrap();
        let plane_at = |axis: usize, at: usize| ArrayD::from_shape_fn(IxDyn(&shape), move |i| i[axis] == at);
        for (name, labels, encoding, ignore) in [
            ("a", &first, "labelmap", Some(&plane_at(0, 4) | &plane_at(2, 0))),
            ("b", &second, "layers", None),
            ("c", &second, "bitmask", Some(plane_at(1, 3))),
            ("d", &second, "labelmap", None),
        ] {
            let options = SegmentationOptions { encoding: Some(encoding.into()), ignore, ..Default::default() };
            w.add_segmentation(name, "g", SegmentationSource::Masks(masks(labels)), options).unwrap();
        }
        w.commit(true).unwrap();
        let sample = open_sample(&path).unwrap();
        assert_eq!(chunk_rows(sample.annotation("a").unwrap()), 3, "a label map is read a chunk of rows at a time");
        let slab = 3 * 72 * 64;
        let ids: Vec<i64> = (1..=classes).collect();
        for (x, y) in [("a", "d"), ("a", "b"), ("a", "c"), ("c", "b"), ("d", "c")] {
            let (a, b) = (sample.annotation(x).unwrap(), sample.annotation(y).unwrap());
            let ignored = match (a.ignore_region().unwrap(), b.ignore_region().unwrap()) {
                (Some(l), Some(r)) => Some(&l | &r),
                (l, r) => l.or(r),
            };
            let skip: Vec<bool> = match &ignored {
                Some(region) => region.iter().copied().collect(),
                None => vec![false; shape.iter().product()],
            };
            let whole: Vec<(u64, u64, u64)> = ids
                .iter()
                .map(|c| {
                    let plane = |ann: &Annotation| ann.dense(Some(&[ClassKey::Id(*c)]), None).unwrap();
                    let (l, r) = (plane(a), plane(b));
                    let (mut both, mut left, mut right) = (0, 0, 0);
                    for ((p, q), _) in l.iter().zip(r.iter()).zip(&skip).filter(|(_, s)| !**s) {
                        both += u64::from(*p && *q);
                        left += u64::from(*p);
                        right += u64::from(*q);
                    }
                    (both, left, right)
                })
                .collect();
            assert!(whole.iter().all(|(both, _, _)| *both > 0), "{x} and {y} overlap in every class");
            // Less than a chunk's rows of one class: a chunk at a time, one
            // class at a time.  Then two classes, four, all of them, and
            // slabs of several chunks, to the whole grid at once.
            for budget in [1, slab - 1, slab, 2 * slab, 4 * slab, 5 * slab, 10 * slab, SLAB_VOXELS] {
                let pairs: Vec<(i64, i64)> = ids.iter().map(|c| (*c, *c)).collect();
                let counted = counts_in_slabs(a, b, &pairs, ignored.as_ref(), budget).unwrap();
                assert_eq!(counted, whole, "{x} vs {y} in a budget of {budget}");
            }
        }
    }
}
