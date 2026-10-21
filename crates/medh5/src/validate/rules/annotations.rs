//! §6--§7 --- annotation headers, references and the voxel encodings (E4xx).

use std::collections::{BTreeMap, BTreeSet, HashMap};

use super::geometric::{bad_boxes, bad_boxes_message, check_classification, check_geometric};
use super::{
    dtype_name, grid_spatial, has_member, int_set, kinds_known, loc, read_f64, read_i64, str_attr, strs_attr,
    sub_dataset, Context,
};
use crate::annotations::header::{is_kind_of, kind_since, RESERVED_KINDS, TASKS, VOXEL_KINDS};
use crate::annotations::layout::{self, mask_problems};
use crate::annotations::payload::{contains_value, SLAB_BYTES};
use crate::annotations::select::greedy_colour;
use crate::array::{Index, Slice};
use crate::h5::attrs;
use crate::h5::data;
use crate::h5::ops::{self, Node};
use crate::json::{py_float, repr_int_list, repr_int_tuple, repr_list, repr_str};
use crate::labels::{BACKGROUND_ID, CLOSURES, IGNORE_ID};
use crate::validate::Diagnostic;
use crate::Result;

/// Extra layers beyond the greedy optimum before W908 fires.
pub const W908_TOLERANCE: usize = 2;

/// The voxel kinds whose reads `closure = "implicit"` widens (§5.4, W918).
const IMPLICIT_VOXEL_KINDS: [&str; 5] = ["labelmap", "layers", "bitmask", "instances", "probmap"];

/// Datasets each kind requires.
pub fn required_datasets(kind: &str) -> &'static [&'static str] {
    match kind {
        "labelmap" | "probmap" | "mask" => &["data"],
        "layers" => &["data", "layer_class_ids"],
        "bitmask" => &["data", "bit_class_ids"],
        "instances" => &["boxes", "class_ids", "instance_ids"],
        "boxes" => &["boxes", "class_ids"],
        "obb" => &["centers", "sizes", "rotations", "class_ids"],
        "keypoints" => &["points", "keypoint_class_ids", "class_ids"],
        "points" => &["points"],
        "contours" => &["vertices", "contour_offsets", "contour_class_ids"],
        "polylines" => &["vertices", "polyline_offsets", "polyline_class_ids"],
        "mesh" => &["vertices", "faces"],
        "classification" => &["class_ids", "values"],
        _ => &[],
    }
}

/// The dtypes a voxel kind's `data` may have.
pub fn allowed_dtypes(kind: &str) -> Option<&'static [&'static str]> {
    Some(match kind {
        "labelmap" | "layers" => &["uint8", "uint16"],
        "bitmask" => &["uint64"],
        "probmap" => &["float16", "float32"],
        "mask" => &["bool", "uint8"],
        _ => return None,
    })
}

/// Per-kind dtype rules for the §8/§9 datasets.
pub fn geometric_dtypes(kind: &str) -> &'static [(&'static str, &'static [&'static str])] {
    match kind {
        "boxes" => &[("boxes", &["float32", "float64"]), ("class_ids", &["uint16"])],
        "obb" => &[
            ("centers", &["float32", "float64"]),
            ("sizes", &["float32", "float64"]),
            ("rotations", &["float32", "float64"]),
        ],
        "keypoints" => &[("points", &["float32", "float64"]), ("visibility", &["uint8"])],
        "points" => &[("points", &["float32", "float64"])],
        "contours" => &[("vertices", &["float32", "float64"]), ("contour_offsets", &["int64"])],
        "polylines" => &[
            ("vertices", &["float32", "float64"]),
            ("polyline_offsets", &["int64"]),
            ("polyline_class_ids", &["uint16"]),
            ("polyline_plane", &["int32"]),
        ],
        "mesh" => &[("vertices", &["float32", "float64"]), ("faces", &["int32", "int64"])],
        "classification" => &[("class_ids", &["uint16"]), ("values", &["float32", "float64"])],
        _ => &[],
    }
}

/// Structural: every annotation against §6--§9 --- its header (E401, E403,
/// E412), what it names (E101, E402, E409, E413, E414), the reserved ids
/// (E303), its datasets against its kind and grid (E405, E408, E410, E411)
/// and a stacked encoding's chunks (W902), each kind's invariants (E404,
/// E406, E407, W908, W909), and partial coverage without an ignore region
/// (W904).  A kind or value only a later minor defines is W913 in its files
/// (1.1 §2.2).
///
/// An annotation that cannot be read is its own finding, and the others are
/// still checked.  Without a document there is nothing to resolve its
/// timepoints against: that is `/meta`'s finding (E004, E005), not an E409
/// for every annotation naming one.
pub fn check_annotations(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let grids = ops::child_group(&ctx.root, "grids");
    let declared_tps: Option<BTreeSet<String>> =
        ctx.document.as_ref().map(|d| d.timepoints.ids().into_iter().collect());
    let label_set = ctx.document.as_ref().and_then(|d| d.label_set.clone());
    let sample = Sample { grids: grids.as_ref(), timepoints: declared_tps.as_ref(), label_set: label_set.as_ref() };
    for (name, group) in ctx.children("annotations")? {
        match check_annotation(ctx, &name, &group, &sample) {
            Ok(found) => out.extend(found),
            Err(e) => out.push(ctx.failed(format!("/annotations/{name}"), &e)),
        }
    }
    Ok(out)
}

/// What an annotation is checked against: the grids, and what the document
/// declares when it could be read.
struct Sample<'a> {
    grids: Option<&'a hdf5::Group>,
    timepoints: Option<&'a BTreeSet<String>>,
    label_set: Option<&'a crate::labels::LabelSet>,
}

/// One annotation, against §6--§9.
fn check_annotation(ctx: &Context, name: &str, group: &Node, sample: &Sample<'_>) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let grids = sample.grids;
    let location = format!("/annotations/{name}");
    let a = loc(group);
    let Some(kind) = str_attr(a, "kind")? else {
        out.push(ctx.err("E412", location, "missing required attribute `kind`"));
        return Ok(out);
    };
    if RESERVED_KINDS.contains(&kind.as_str()) {
        out.push(ctx.unknown(
            "E401",
            location,
            format!("kind {} is reserved by spec §16 and must not appear in a 1.0 or 1.1 file", repr_str(&kind)),
        ));
        return Ok(out);
    }
    if !is_kind_of(&kind, ctx.version.as_deref()) {
        // A kind a later version defines is no kind a file of this one holds
        // (1.2 §2): a writer of it writes that version.
        let message = match (kind_since(&kind), ctx.version.as_deref()) {
            (Some(since), Some(version)) => format!(
                "annotation kind {} is defined by MEDH5 {since}, and this file declares {version}; a sample holding \
                 it is written at {since} or later",
                repr_str(&kind)
            ),
            _ => format!("unknown annotation kind {}", repr_str(&kind)),
        };
        out.push(ctx.unknown("E401", location, message));
        return Ok(out);
    }
    // What a reader would otherwise assume: a `closure` it took as
    // `explicit`, a `task` it guessed from the kind.
    let mut missing: Vec<&str> = ["task", "closure"].into_iter().filter(|k| !attrs::has(a, k)).collect();
    if kind != "mask" && !attrs::has(a, "class_ids") {
        missing.push("class_ids");
    }
    if !missing.is_empty() {
        out.push(ctx.err(
            "E412",
            location.clone(),
            format!("missing required attribute(s) {}; §6.2 requires them of every annotation", repr_list(&missing)),
        ));
    }
    if let Some(task) = str_attr(a, "task")? {
        if !TASKS.contains(&task.as_str()) {
            out.push(ctx.unknown("E412", location.clone(), format!("unknown task {}", repr_str(&task))));
        }
    }
    if let Some(closure) = str_attr(a, "closure")? {
        if !CLOSURES.contains(&closure.as_str()) {
            out.push(ctx.unknown("E412", location.clone(), format!("unknown closure {}", repr_str(&closure))));
        }
        // §5.4: writers SHOULD use `explicit` for voxel annotations, and under
        // `implicit` every read of an ancestor includes its descendants' voxels.
        if closure == "implicit"
            && IMPLICIT_VOXEL_KINDS.contains(&kind.as_str())
            && !ctx.errors_only
            && ctx.level != "structural"
        {
            out.push(ctx.err(
                "W918",
                location.clone(),
                "a voxel annotation declares closure='implicit': readers include every `is_a` descendant's voxels \
                 in each ancestor's reads (§5.4); writers SHOULD use 'explicit' for voxel annotations",
            ));
        }
    }
    if !attrs::has(a, "annotated_class_ids") && kind != "mask" {
        out.push(ctx.err("E412", location.clone(), "missing `annotated_class_ids`; the coverage contract is required"));
    }
    let class_ids = int_set(a, "class_ids")?;
    let annotated = int_set(a, "annotated_class_ids")?;
    // Without `class_ids` that is the finding (E412), not every annotated id.
    if !annotated.is_subset(&class_ids) && kind != "mask" && attrs::has(a, "class_ids") {
        let extra: Vec<i64> = annotated.difference(&class_ids).copied().collect();
        out.push(ctx.err(
            "E403",
            location.clone(),
            format!("annotated_class_ids {} are not in class_ids", repr_int_list(&extra)),
        ));
    }
    if let Some(ls) = sample.label_set.filter(|l| l.form == "inline") {
        let missing = ls.missing(class_ids.iter().copied());
        if !missing.is_empty() {
            out.push(ctx.err(
                "E402",
                location.clone(),
                format!("class ids {} are not in label set {}", repr_int_list(&missing), repr_str(&ls.id)),
            ));
        }
    }
    let reserved: Vec<i64> = class_ids.iter().copied().filter(|c| *c == BACKGROUND_ID || *c == IGNORE_ID).collect();
    if !reserved.is_empty() {
        out.push(ctx.err(
            "E303",
            location.clone(),
            format!("class_ids uses reserved id(s) {}", repr_int_list(&reserved)),
        ));
    }
    out.extend(check_ignore_id(ctx, &location, a, &class_ids)?);
    if let (true, Some(declared)) = (attrs::has(a, "timepoints"), sample.timepoints) {
        for tp in strs_attr(a, "timepoints")? {
            if !declared.contains(&tp) {
                out.push(ctx.err("E409", location.clone(), format!("undeclared timepoint {}", repr_str(&tp))));
            }
        }
    }
    let grid_id = str_attr(a, "grid")?;
    // A grid is a group (§3.2): a dataset or a dangling link of its name is
    // no grid to read.
    let on_grid = |gid: &str| grids.is_some_and(|g| ops::is_group(g, gid));
    if VOXEL_KINDS.contains(&kind.as_str()) {
        match &grid_id {
            None => out.push(ctx.err("E412", location.clone(), format!("kind {} requires a `grid`", repr_str(&kind)))),
            Some(gid) if !on_grid(gid) => out.push(ctx.err(
                "E101",
                location.clone(),
                format!("names grid {}, which does not exist", repr_str(gid)),
            )),
            _ => {}
        }
    }
    for required in required_datasets(&kind) {
        if !has_member(group, required) {
            out.push(ctx.err(
                "E410",
                location.clone(),
                format!("kind {} requires dataset {}", repr_str(&kind), repr_str(required)),
            ));
        }
    }
    if let (Some(ds), Some(allowed)) = (sub_dataset(group, "data"), allowed_dtypes(&kind)) {
        let found = dtype_name(&ds);
        if !allowed.contains(&found.as_str()) {
            out.push(ctx.err(
                "E411",
                format!("{location}/data"),
                format!(
                    "dtype {found} is not permitted for kind {} (expected one of {})",
                    repr_str(&kind),
                    repr_list(allowed)
                ),
            ));
        }
    }
    // The grid's spatial axes, where it has them and they are known: what
    // index coordinates and instance masks are measured against.
    let spatial: Option<Vec<usize>> = match (&grid_id, grids) {
        (Some(gid), Some(g)) if on_grid(gid) && kinds_known(g, gid)? => grid_spatial(g, gid)
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.iter().map(|v| (*v).max(0) as usize).collect()),
        _ => None,
    };
    let space = str_attr(a, "space")?;
    let rank = spatial.as_ref().filter(|_| layout::ranked(&kind, space.as_deref())).map(Vec::len);
    let columns = layout::column_problems(&kind, &|n| sub_dataset(group, n).map(|d| d.shape()), rank);
    // Rules that index one column by another's rows run only where the
    // columns agree; the disagreement is the finding.
    let consistent = columns.is_empty();
    for problem in columns {
        out.push(ctx.err(problem.code, format!("{location}/{}", problem.dataset), problem.message));
    }
    out.extend(check_voxel_shape(ctx, name, group, &kind, grid_id.as_deref(), grids)?);
    out.extend(check_encoding_invariants(ctx, name, group, &kind, consistent, spatial.as_deref())?);
    out.extend(check_dataset_dtypes(ctx, name, group, &kind));
    out.extend(check_object_classes(ctx, name, group, &kind, &class_ids)?);
    out.extend(check_geometric(ctx, name, group, &kind, grid_id.as_deref(), grids, consistent)?);
    out.extend(check_classification(ctx, name, group, &kind, consistent)?);
    if !ctx.errors_only
        && kind != "mask"
        && annotated.is_subset(&class_ids)
        && annotated.len() < class_ids.len()
        && !has_ignore(group, &kind)?
    {
        out.push(ctx.err(
            "W904",
            location,
            format!(
                "{} class(es) are encodable but not annotated, and there is no ignore region; `0` cannot be read as a verified negative for them",
                class_ids.difference(&annotated).count()
            ),
        ));
    }
    Ok(out)
}

/// A stored `ignore_id` (§6.2, §7.7) is a `uint16` no class uses, and not
/// background: `0` made every background voxel read as ignored, and an id a
/// class holds made that class's voxels ignored --- either way the region
/// where `0` is a verified negative was not the one the file means (E404).
fn check_ignore_id(
    ctx: &Context,
    location: &str,
    a: &hdf5::Location,
    class_ids: &BTreeSet<i64>,
) -> Result<Vec<Diagnostic>> {
    let Some(value) = attrs::read(a, "ignore_id")? else { return Ok(Vec::new()) };
    let message = match value.as_i64() {
        None => "`ignore_id` is not an integer; it is a uint16 (§6.2)".to_string(),
        Some(id) if !(1..=i64::from(u16::MAX)).contains(&id) => format!(
            "`ignore_id` {id} is outside 1..65535: it is stored as a uint16 beside the class ids, and 0 is \
             background, so every unlabelled voxel would read as ignored"
        ),
        Some(id) if class_ids.contains(&id) => {
            format!("`ignore_id` {id} is also a class id; one value cannot mark both the class and the ignore region")
        }
        Some(_) => return Ok(Vec::new()),
    };
    Ok(vec![ctx.err("E404", location, message)])
}

fn has_ignore(group: &Node, kind: &str) -> Result<bool> {
    if attrs::has(loc(group), "ignore_mask") {
        return Ok(true);
    }
    in_band_ignore(group, kind)
}

fn in_band_ignore(group: &Node, kind: &str) -> Result<bool> {
    if kind != "labelmap" && kind != "layers" {
        return Ok(false);
    }
    let Some(ds) = sub_dataset(group, "data") else { return Ok(false) };
    let ignore_id = attrs::get_i64(loc(group), "ignore_id")?.unwrap_or(IGNORE_ID);
    contains_value(&ds, ignore_id)
}

fn annotation_id(reference: &str) -> &str {
    reference.strip_prefix("annotations/").unwrap_or(reference)
}

/// Every cross-object reference §4.4 and §6.2 name resolves (E413).
pub fn check_references(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let groups: BTreeMap<String, Node> = ctx.children("annotations")?.into_iter().collect();
    for (name, group) in &groups {
        let location = format!("/annotations/{name}");
        let a = loc(group);
        if let Some(target) = str_attr(a, "ignore_mask")? {
            out.extend(check_mask_reference(ctx, &location, "ignore_mask", &target, a, &groups)?);
        }
        if attrs::has(a, "derived_from") {
            for reference in strs_attr(a, "derived_from")? {
                if !groups.contains_key(annotation_id(&reference)) {
                    out.push(ctx.err(
                        "E413",
                        location.clone(),
                        format!("`derived_from` names annotation {}, which does not exist", repr_str(&reference)),
                    ));
                }
            }
        }
    }
    for (name, node) in ctx.children("images")? {
        if let Some(target) = str_attr(loc(&node), "valid_mask")? {
            out.extend(check_mask_reference(
                ctx,
                &format!("/images/{name}"),
                "valid_mask",
                &target,
                loc(&node),
                &groups,
            )?);
        }
    }
    Ok(out)
}

fn check_mask_reference(
    ctx: &Context,
    location: &str,
    attr: &str,
    target: &str,
    owner: &hdf5::Location,
    groups: &BTreeMap<String, Node>,
) -> Result<Vec<Diagnostic>> {
    let Some(other) = groups.get(target) else {
        return Ok(vec![ctx.err(
            "E413",
            location,
            format!("`{attr}` names annotation {}, which does not exist", repr_str(target)),
        )]);
    };
    let kind = str_attr(loc(other), "kind")?;
    if kind.as_deref() != Some("mask") {
        return Ok(vec![ctx.err(
            "E413",
            location,
            format!(
                "`{attr}` names {}, whose kind is {}; §4.4 and §7.7 require a `mask` annotation",
                repr_str(target),
                kind.as_deref().map(repr_str).unwrap_or_else(|| "None".into())
            ),
        )]);
    }
    let mine = str_attr(owner, "grid")?;
    let theirs = str_attr(loc(other), "grid")?;
    if let (Some(m), Some(t)) = (mine, theirs) {
        if m != t {
            return Ok(vec![ctx.err(
                "E413",
                location,
                format!(
                    "`{attr}` names {} on grid {}, but this object is on grid {}; a mask delimits the voxels of the grid it shares",
                    repr_str(target),
                    repr_str(&t),
                    repr_str(&m)
                ),
            )]);
        }
    }
    Ok(Vec::new())
}

fn check_voxel_shape(
    ctx: &Context,
    name: &str,
    group: &Node,
    kind: &str,
    grid_id: Option<&str>,
    grids: Option<&hdf5::Group>,
) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let (Some(gid), Some(grids)) = (grid_id, grids) else { return Ok(out) };
    if !VOXEL_KINDS.contains(&kind) || !ops::is_group(grids, gid) {
        return Ok(out);
    }
    let Some(ds) = sub_dataset(group, "data") else { return Ok(out) };
    let shape: Vec<i64> = ds.shape().iter().map(|v| *v as i64).collect();
    let stacked = matches!(kind, "layers" | "bitmask" | "probmap");
    let tail: Vec<i64> = if stacked { shape.get(1..).unwrap_or(&[]).to_vec() } else { shape.clone() };
    // A grid naming an axis kind this validator does not know has no spatial
    // shape to compare with: its own finding stands, and the rest is judged.
    if kinds_known(grids, gid)? {
        let spatial = grid_spatial(grids, gid)?;
        if tail != spatial {
            out.push(ctx.err(
                "E405",
                format!("/annotations/{name}/data"),
                format!(
                    "spatial shape {} != grid {} spatial shape {}",
                    repr_int_tuple(&tail),
                    repr_str(gid),
                    repr_int_tuple(&spatial)
                ),
            ));
        }
    }
    // A probmap's leading axis is its classes (§7.5): a plane per class id.
    // Absent or unreadable `class_ids` are another rule's finding.
    let declared = attrs::get_i64s(loc(group), "class_ids").ok().flatten();
    if let (true, Some(ids)) = (kind == "probmap", declared) {
        let classes = ids.len();
        if shape.first().copied() != Some(classes as i64) {
            out.push(ctx.err(
                "E405",
                format!("/annotations/{name}/data"),
                format!(
                    "{} planes for {classes} class id(s); a probmap stores one plane per class, in `class_ids` order",
                    shape.first().copied().unwrap_or(0)
                ),
            ));
        }
    }
    if stacked {
        if let Some(chunks) = data::chunks(&ds) {
            if chunks.first().copied() != Some(1) {
                out.push(ctx.err(
                    "W902",
                    format!("/annotations/{name}/data"),
                    format!(
                        "chunk shape {} spans the stacked axis; the spec requires (1, *spatial_chunk) so one plane reads without the others",
                        repr_int_tuple(&chunks)
                    ),
                ));
            }
        }
    }
    Ok(out)
}

fn check_encoding_invariants(
    ctx: &Context,
    name: &str,
    group: &Node,
    kind: &str,
    consistent: bool,
    spatial: Option<&[usize]>,
) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let location = format!("/annotations/{name}");
    let declared = int_set(loc(group), "class_ids")?;
    if kind == "layers" {
        if let Some(table_ds) = sub_dataset(group, "layer_class_ids") {
            let table = read_i64(&table_ds)?;
            let rows: Vec<Vec<i64>> = if table.ndim() >= 2 {
                table.outer_iter().map(|r| r.iter().copied().collect()).collect()
            } else {
                vec![table.iter().copied().collect()]
            };
            // A row per layer (§7.2): a class listed in a row with no plane is
            // read from none, and a plane with no row holds classes nothing
            // names --- a voxel of either read as a verified negative.
            if let Some(planes) = sub_dataset(group, "data").and_then(|d| d.shape().first().copied()) {
                if rows.len() != planes {
                    out.push(ctx.err(
                        "E405",
                        format!("{location}/layer_class_ids"),
                        format!(
                            "`layer_class_ids` has {} row(s) for the {planes} layer(s) of `data`; §7.2 gives each \
                             layer one row",
                            rows.len()
                        ),
                    ));
                }
            }
            let mut seen: BTreeMap<i64, usize> = BTreeMap::new();
            for (layer, row) in rows.iter().enumerate() {
                for class_id in row {
                    if *class_id == 0 {
                        continue;
                    }
                    if let Some(first) = seen.get(class_id) {
                        out.push(ctx.err(
                            "E404",
                            location.clone(),
                            format!("class {class_id} appears in layers {first} and {layer}; every class must be in exactly one layer"),
                        ));
                    }
                    seen.insert(*class_id, layer);
                }
            }
            let missing: Vec<i64> = declared.iter().copied().filter(|c| !seen.contains_key(c)).collect();
            if !missing.is_empty() {
                out.push(ctx.err(
                    "E404",
                    location.clone(),
                    format!("class_ids {} are not assigned to any layer", repr_int_list(&missing)),
                ));
            }
            out.extend(check_layer_optimality(ctx, name, group, rows.len(), declared.len())?);
        }
    }
    if kind == "labelmap" {
        if let Some(ds) = sub_dataset(group, "data") {
            if dtype_name(&ds) == "uint16"
                && !declared.is_empty()
                && declared.iter().max().copied().unwrap_or(0) <= 254
                && !in_band_ignore(group, kind)?
            {
                out.push(ctx.err(
                    "E411",
                    format!("{location}/data"),
                    "stored as uint16 although max(class_ids) is at most 254 and no ignore voxel is present; §7.1 requires uint8",
                ));
            }
        }
    }
    if kind == "probmap" {
        if let Some(value) = attrs::get_f64(loc(group), "threshold")? {
            if !(0.0..=1.0).contains(&value) || value.is_nan() {
                out.push(ctx.err(
                    "E404",
                    location.clone(),
                    format!(
                        "`threshold` {} is outside [0, 1]; it is the probability at or above which a voxel contains the class (§7.5)",
                        py_float(value)
                    ),
                ));
            }
        }
    }
    if kind == "bitmask" {
        if let Some(bits) = sub_dataset(group, "bit_class_ids") {
            out.extend(check_bit_positions(ctx, &location, &bits, &declared)?);
        }
        if let (Some(bits), Some(ds)) = (sub_dataset(group, "bit_class_ids"), sub_dataset(group, "data")) {
            let n_classes = bits.shape().iter().product::<usize>();
            let expected = n_classes.div_ceil(64).max(1);
            let planes = ds.shape().first().copied().unwrap_or(0);
            if planes != expected {
                out.push(ctx.err(
                    "E404",
                    location.clone(),
                    format!("{planes} bitplanes for {n_classes} classes; expected {expected}"),
                ));
            }
        }
    }
    if kind == "instances" {
        out.extend(check_instances(ctx, name, group, consistent, spatial)?);
    }
    Ok(out)
}

/// `bit_class_ids` (§7.3) is `(C,)`, one bit position per class id: each
/// id once, and every id of `class_ids`.  A class given two positions reads
/// as the union of two bits' voxels, and a declared class given none reads as
/// absent everywhere --- a negative nobody looked for.  An id the attribute
/// does not declare is [`check_object_classes`]'s finding.
fn check_bit_positions(
    ctx: &Context,
    location: &str,
    bits: &hdf5::Dataset,
    declared: &BTreeSet<i64>,
) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let shape = bits.shape();
    if shape.len() != 1 || shape[0] != declared.len() {
        out.push(ctx.err(
            "E405",
            format!("{location}/bit_class_ids"),
            format!(
                "`bit_class_ids` has shape {}; a `bitmask` stores it as ({},), one position per class id (§7.3)",
                repr_int_tuple(&shape),
                declared.len()
            ),
        ));
    }
    // Unreadable values are another rule's finding (E411).
    let Ok(values) = read_i64(bits) else { return Ok(out) };
    let mut seen = BTreeSet::new();
    let mut repeated: Vec<i64> = values.iter().copied().filter(|c| !seen.insert(*c)).collect();
    repeated.sort_unstable();
    repeated.dedup();
    if !repeated.is_empty() {
        out.push(ctx.err(
            "E404",
            format!("{location}/bit_class_ids"),
            format!("class id(s) {} hold more than one bit position; each class has one bit", repr_int_list(&repeated)),
        ));
    }
    let unplaced: Vec<i64> = declared.iter().copied().filter(|c| !seen.contains(c)).collect();
    if !unplaced.is_empty() {
        out.push(ctx.err(
            "E404",
            format!("{location}/bit_class_ids"),
            format!(
                "class id(s) {} of `class_ids` have no bit position, so no voxel can hold them",
                repr_int_list(&unplaced)
            ),
        ));
    }
    Ok(out)
}

fn check_layer_optimality(
    ctx: &Context,
    name: &str,
    group: &Node,
    n_layers: usize,
    n_classes: usize,
) -> Result<Vec<Diagnostic>> {
    // A layer-data read, so not at `structural`; every level above runs it,
    // as each level runs everything the ones before it do.
    if ctx.errors_only || ctx.level == "structural" || n_classes == 0 {
        return Ok(Vec::new());
    }
    let (Some(table_ds), Some(ds)) = (sub_dataset(group, "layer_class_ids"), sub_dataset(group, "data")) else {
        return Ok(Vec::new());
    };
    let mut classes: Vec<i64> = read_i64(&table_ds)?.iter().copied().filter(|v| *v != 0).collect();
    classes.sort_unstable();
    classes.dedup();
    let ignore_id = attrs::get_i64(loc(group), "ignore_id")?.unwrap_or(IGNORE_ID);
    let colouring = greedy_colour(&classes, &overlap_edges(&ds, ignore_id)?);
    let optimal = colouring.values().max().map(|m| m + 1).unwrap_or(0);
    if n_layers > optimal + W908_TOLERANCE {
        let cut = 100.0 * (1.0 - optimal as f64 / n_layers as f64);
        return Ok(vec![ctx.err(
            "W908",
            format!("/annotations/{name}"),
            format!(
                "{n_layers} layers where a greedy colouring of the overlap graph needs {optimal}; transcoding would cut the label volume by {}%",
                format_g_fixed0(cut)
            ),
        )]);
    }
    Ok(Vec::new())
}

/// `format(value, ".0f")` with Python's round-half-even.
fn format_g_fixed0(value: f64) -> String {
    format!("{:.0}", value.round_ties_even())
}

/// The class overlap graph of a `layers` dataset, read in bounded slabs.
///
/// Within one layer classes never share a voxel, so an edge is a pair of
/// values that co-occur at a voxel in two different layers.  Pairing the
/// layers slab by slab answers that directly, in memory bounded by the slab
/// rather than by classes times voxels.
fn overlap_edges(ds: &hdf5::Dataset, ignore_id: i64) -> Result<BTreeSet<(i64, i64)>> {
    overlap_edges_within(ds, ignore_id, SLAB_BYTES)
}

/// [`overlap_edges`] reading at most about `budget` bytes at a time.
fn overlap_edges_within(ds: &hdf5::Dataset, ignore_id: i64, budget: usize) -> Result<BTreeSet<(i64, i64)>> {
    let shape = ds.shape();
    let mut edges = BTreeSet::new();
    // Not `product() == 0`: four layers on a grid of 2**62 voxels wrapped to
    // exactly that, and the check passed what it never read.
    if shape.len() < 2 || shape[0] < 2 || shape.contains(&0) {
        return Ok(edges);
    }
    let n_layers = shape[0];
    let rows = shape[1];
    let itemsize = data::dtype(ds)?.itemsize();
    // Saturating: a row past `usize` is a row over any budget.
    let per_row = shape[2..].iter().fold(n_layers.saturating_mul(itemsize), |n, e| n.saturating_mul(*e));
    let step = (budget / per_row.max(1)).clamp(1, rows);
    let mut start = 0;
    while start < rows {
        let block = data::read_region(
            ds,
            &[Index::Slice(Slice::full()), Index::Slice(Slice::new(start as i64, (start + step) as i64))],
        )?
        .cast::<i64>();
        let n = block.len() / n_layers;
        let flat: Vec<i64> = block.iter().copied().collect();
        for i in 0..n_layers {
            for j in i + 1..n_layers {
                for k in 0..n {
                    let (a, b) = (flat[i * n + k], flat[j * n + k]);
                    if a != 0 && a != ignore_id && b != 0 && b != ignore_id {
                        edges.insert(if a < b { (a, b) } else { (b, a) });
                    }
                }
            }
        }
        start += step;
    }
    Ok(edges)
}

fn check_dataset_dtypes(ctx: &Context, name: &str, group: &Node, kind: &str) -> Vec<Diagnostic> {
    let mut out = Vec::new();
    for (dataset, allowed) in geometric_dtypes(kind) {
        let Some(ds) = sub_dataset(group, dataset) else { continue };
        let found = dtype_name(&ds);
        if !allowed.contains(&found.as_str()) {
            out.push(ctx.err(
                "E411",
                format!("/annotations/{name}/{dataset}"),
                format!(
                    "dtype {found} is not permitted for {}.{dataset} (expected one of {})",
                    repr_str(kind),
                    repr_list(allowed)
                ),
            ));
        }
    }
    out
}

/// `consistent` is whether the object columns agree in shape; `spatial` the
/// grid's spatial shape, which no object's mask may outgrow.
/// The datasets whose values are class ids: the per-object and per-element
/// classes, and the tables `layers` and `bitmask` index by class.
fn class_columns(kind: &str) -> &'static [&'static str] {
    match kind {
        "instances" | "boxes" | "obb" | "points" | "classification" => &["class_ids"],
        "keypoints" => &["class_ids", "keypoint_class_ids"],
        "contours" => &["contour_class_ids"],
        "polylines" => &["polyline_class_ids"],
        "mesh" => &["vertex_class_ids", "mesh_class_ids"],
        "layers" => &["layer_class_ids"],
        "bitmask" => &["bit_class_ids"],
        _ => &[],
    }
}

/// Every class the data can encode is in the `class_ids` attribute (§6.2,
/// E404).
///
/// Coverage is read from the attribute, never from the data: a class the
/// data holds and the attribute omits is answered for by `contains()` and
/// missing from every question about what the annotation can express.
fn check_object_classes(
    ctx: &Context,
    name: &str,
    group: &Node,
    kind: &str,
    declared: &BTreeSet<i64>,
) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    for column in class_columns(kind) {
        // An unreadable column is another rule's finding (E411).
        let Some(values) = sub_dataset(group, column).and_then(|ds| read_i64(&ds).ok()) else { continue };
        // `layer_class_ids` pads its rows with 0, which is no class.
        let padding = |c: &i64| kind == "layers" && *c == BACKGROUND_ID;
        let mut undeclared: Vec<i64> =
            values.iter().copied().filter(|c| !declared.contains(c) && !padding(c)).collect();
        undeclared.sort_unstable();
        undeclared.dedup();
        if !undeclared.is_empty() {
            out.push(ctx.err(
                "E404",
                format!("/annotations/{name}/{column}"),
                format!(
                    "class id(s) {} in `{column}` are not in the `class_ids` attribute, which declares every class \
                     the annotation can express (§6.2)",
                    repr_int_list(&undeclared)
                ),
            ));
        }
    }
    Ok(out)
}

fn check_instances(
    ctx: &Context,
    name: &str,
    group: &Node,
    consistent: bool,
    spatial: Option<&[usize]>,
) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let location = format!("/annotations/{name}");
    let boxes = match sub_dataset(group, "boxes") {
        Some(ds) => Some(read_f64(&ds)?),
        None => None,
    };
    if let Some(boxes) = boxes.as_ref().filter(|b| b.ndim() == 3) {
        let bad = bad_boxes(boxes);
        if bad > 0 {
            out.push(ctx.err("E406", location.clone(), bad_boxes_message(bad)));
        }
    }
    let offsets: Option<Vec<i64>> = match sub_dataset(group, "mask_offsets") {
        Some(ds) => Some(read_i64(&ds)?.iter().copied().collect()),
        None => None,
    };
    let stored = (sub_dataset(group, "mask_shapes"), sub_dataset(group, "mask_data"));
    if let (true, Some(boxes), Some(offsets), (Some(shapes), Some(bytes))) = (consistent, &boxes, &offsets, stored) {
        // What a reader holds the masks to before it decodes one (§7.4).
        let shapes = read_i64(&shapes)?;
        let stored = bytes.shape().iter().product();
        for problem in mask_problems(boxes, offsets, &shapes, stored, spatial) {
            out.push(ctx.err(problem.code, format!("{location}/{}", problem.dataset), problem.message));
        }
    } else if offsets.as_ref().is_some_and(|o| o.windows(2).any(|w| w[1] < w[0])) {
        out.push(ctx.err("E408", format!("{location}/mask_offsets"), "`mask_offsets` are not monotonic"));
    }
    let ids: Option<Vec<u64>> = match sub_dataset(group, "instance_ids") {
        Some(ds) => Some(data::read(&ds)?.cast::<u64>().iter().copied().collect()),
        None => None,
    };
    if let Some(ids) = &ids {
        let mut counts: BTreeMap<u64, usize> = BTreeMap::new();
        for i in ids {
            *counts.entry(*i).or_default() += 1;
        }
        let shared: Vec<u64> = counts.into_iter().filter(|(_, n)| *n > 1).map(|(i, _)| i).collect();
        if !shared.is_empty() {
            out.push(ctx.err(
                "E404",
                location.clone(),
                format!(
                    "instance id(s) {} name more than one object; two distinct objects MUST NOT share one (§7.4)",
                    repr_int_list(&shared)
                ),
            ));
        }
    }
    // A row count the columns disagree on is already E405; pairing them
    // past the shorter one has no answer.
    if let (true, Some(ids), Some(cds)) = (consistent, &ids, sub_dataset(group, "class_ids")) {
        let classes: Vec<i64> = read_i64(&cds)?.iter().copied().collect();
        let mut table: HashMap<u64, i64> = HashMap::new();
        let mut conflicting = BTreeSet::new();
        for (i, c) in ids.iter().zip(&classes) {
            if *table.entry(*i).or_insert(*c) != *c {
                conflicting.insert(*i);
            }
        }
        if !conflicting.is_empty() {
            let listed: Vec<u64> = conflicting.into_iter().collect();
            out.push(ctx.err(
                "W909",
                location,
                format!(
                    "instance id(s) {} carry more than one class id; this is almost always a tracking error rather than a reclassification",
                    repr_int_list(&listed)
                ),
            ));
        }
    }
    Ok(out)
}

/// `instance_id` is sample-scoped, so the check has to be too (§7.4).
pub fn check_instance_identity(ctx: &mut Context) -> Result<Vec<Diagnostic>> {
    let mut out = Vec::new();
    let mut table: BTreeMap<u64, BTreeMap<i64, Vec<String>>> = BTreeMap::new();
    for (name, group) in ctx.children("annotations")? {
        // A polyline's class is its `polyline_class_ids` (1.2 §3).
        let classes = sub_dataset(&group, "class_ids").or_else(|| sub_dataset(&group, "polyline_class_ids"));
        let (Some(ids_ds), Some(cls_ds)) = (sub_dataset(&group, "instance_ids"), classes) else {
            continue;
        };
        let ids: Vec<u64> = data::read(&ids_ds)?.cast::<u64>().iter().copied().collect();
        let classes: Vec<i64> = read_i64(&cls_ds)?.iter().copied().collect();
        if ids.len() != classes.len() {
            continue;
        }
        for (i, c) in ids.iter().zip(&classes) {
            table.entry(*i).or_default().entry(*c).or_default().push(name.clone());
        }
    }
    for (instance_id, by_class) in table {
        if by_class.len() < 2 {
            continue;
        }
        let where_: BTreeSet<&String> = by_class.values().flatten().collect();
        if where_.len() < 2 {
            continue;
        }
        let detail = by_class
            .iter()
            .map(|(c, names)| {
                let unique: BTreeSet<&String> = names.iter().collect();
                format!("class {c} in {}", repr_list(&unique.into_iter().collect::<Vec<_>>()))
            })
            .collect::<Vec<_>>()
            .join(", ");
        out.push(ctx.err(
            "W909",
            "/annotations",
            format!(
                "instance id {instance_id} carries several class ids across annotations ({detail}); the longitudinal join treats these as one object"
            ),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{s, Array4};

    #[test]
    fn s7_2_the_overlap_graph_is_read_in_slabs() {
        // Two layers of 8 rows of 4x4 uint16: 64 bytes a row, so a 64-byte
        // budget reads one row per slab.
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("layers.h5"))).unwrap();
        let mut data = Array4::<u16>::zeros((2, 8, 4, 4));
        data.slice_mut(s![0, 1..6, .., ..]).fill(1); // liver
        data.slice_mut(s![0, 7, .., ..]).fill(3); // spleen, in the liver's layer
        data.slice_mut(s![1, 5, 1..3, 1..3]).fill(2); // lesion, inside the liver
        data[[1, 3, 0, 0]] = 65535; // ignore over the liver: not an overlap
        let ds = file.new_dataset_builder().with_data(&data).create("data").unwrap();
        let whole = overlap_edges(&ds, 65535).unwrap();
        assert_eq!(whole, BTreeSet::from([(1, 2)]));
        assert_eq!(overlap_edges_within(&ds, 65535, 64).unwrap(), whole);
        assert_eq!(overlap_edges_within(&ds, 65535, 1).unwrap(), whole);
    }

    /// Four layers of `2 x 2**62` voxels are `2**65`, which wrapped to 0:
    /// the check took the dataset for empty and passed what it never read.
    /// Its rows are past any read, so the check reports that it cannot read.
    #[test]
    fn s7_2_a_dataset_past_the_integers_is_not_taken_for_empty() {
        let dir = tempfile::tempdir().unwrap();
        let file = crate::h5::file::marked(|| hdf5::File::create(dir.path().join("layers.h5"))).unwrap();
        let ds = crate::h5::unwritten(&file, "data", crate::array::DType::U8, &[4, 2, 1 << 62]);
        let err = overlap_edges(&ds, 65535).unwrap_err();
        assert!(err.to_string().contains("cannot be read"), "{err}");
    }
}
