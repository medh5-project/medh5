//! [`Sample`]: a read-only, lazy, timepoint-aware view of one sample root.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

use indexmap::IndexMap;
use ndarray::{ArrayD, IxDyn};
use serde_json::{json, Map, Value};

use super::image::{Image, SPEC_IMAGE_ATTRS};
use super::projection;
use crate::annotations::header::SPEC_ANNOTATION_ATTRS;
use crate::annotations::{Annotation, Grids};
use crate::array::Slice;
use crate::clinical::{Clinical, Documents};
use crate::curation::identity::{Cohort, Identity};
use crate::curation::timeline::Timeline;
use crate::document::{SampleDocument, META_DATASET};
use crate::geometry::grid::{read_grid, read_grids, Grid, SPEC_GRID_ATTRS};
use crate::h5::attrs::{self, AttrValue};
use crate::h5::{data, file, ops};
use crate::integrity::{compute_content_id, stale_index_entries, verify_root, AttrNameMap, VerifyResult};
use crate::json::{repr_list, repr_str};
use crate::labels::LabelSet;
use crate::storage::index::{read_indices, SamplingIndex};
use crate::transforms::model::{open_transforms, transform_groups, Transform, TransformHeader, SPEC_TRANSFORM_ATTRS};
use crate::transforms::resolve::{frames_of_timepoint, resolve_between};
use crate::{Error, Result, FORMAT_VERSION};

/// The conformance profiles a sample may declare (1.0 §1.3; `clinical`, 1.1 §1).
pub const PROFILES: [&str; 10] =
    ["core", "seg", "det", "cls", "reg", "curation", "multiscale", "training", "longitudinal", "clinical"];

/// Root attributes covered by `content_id`; `created` and `generator` are
/// excluded so byte-identical samples written apart share an address (§13.2).
pub const ROOT_DIGEST_ATTRS: [&str; 3] = ["medh5_version", "medh5_kind", "medh5_profiles"];

/// Where frame-of-reference UIDs are named.
pub const FRAME_ATTRS: [(&str, &[&str]); 3] =
    [("grids", &["frame_uid"]), ("annotations", &["frame_uid"]), ("transforms", &["from_frame", "to_frame"])];

/// How [`Sample::transform_between_as`] reads a key.  The three namespaces
/// are separate (§2.3), so a grid may share its id with a timepoint.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KeyKind {
    /// A timepoint id: the frames of every grid of that visit.
    Timepoint,
    /// A grid id: the grid's own frame.
    Grid,
    /// A frame-of-reference uid.
    Frame,
}

impl KeyKind {
    /// `"timepoint"`, `"grid"` or `"frame"`.
    pub fn as_str(self) -> &'static str {
        match self {
            KeyKind::Timepoint => "timepoint",
            KeyKind::Grid => "grid",
            KeyKind::Frame => "frame",
        }
    }

    /// The kind [`KeyKind::as_str`] spells `name`; a `ValueError` otherwise.
    pub fn parse(name: &str) -> Result<KeyKind> {
        match name {
            "timepoint" => Ok(KeyKind::Timepoint),
            "grid" => Ok(KeyKind::Grid),
            "frame" => Ok(KeyKind::Frame),
            other => Err(Error::Value(format!("by must be 'timepoint', 'grid' or 'frame', not {}", repr_str(other)))),
        }
    }
}

/// The `KeyError` a collection raises for a missing name.
pub fn missing_key(what: &str, key: &str, available: impl IntoIterator<Item = String>) -> Error {
    let mut names: Vec<String> = available.into_iter().collect();
    names.sort();
    Error::Key(format!("no {what} {}; available: {}", repr_str(key), repr_list(&names)))
}

/// A failed lookup of `key` in `listing`: the W913 of an object a later
/// minor's file holds that the projection leaves out, else the `KeyError`
/// naming the available ones.
fn lookup_failure<'a>(
    left_out: &IndexMap<String, Error>,
    what: &str,
    key: &str,
    available: impl IntoIterator<Item = &'a String>,
) -> Error {
    match left_out.get(key) {
        Some(left_out) => left_out.clone(),
        None => missing_key(what, key, available.into_iter().cloned()),
    }
}

/// The grids an image lies on: its own, and a pyramid's levels'.
fn image_grids(location: &hdf5::Location) -> Result<Vec<String>> {
    let mut out: Vec<String> = attrs::get_str(location, "grid")?.into_iter().collect();
    out.extend(attrs::get_strs(location, "grid_levels")?.unwrap_or_default());
    Ok(out)
}

/// What `cell` holds, read by `init` on first use, once `root` is known to
/// be open: a closed file answers a lookup as an empty one, which read as a
/// sample without its groups (E008) or its document --- and a value read
/// before the close is no answer from a file that is gone.
fn cached<'a, T>(root: &hdf5::Group, cell: &'a OnceLock<Result<T>>, init: impl FnOnce() -> Result<T>) -> Result<&'a T> {
    crate::h5::alive(root)?;
    match cell.get_or_init(init) {
        Ok(v) => Ok(v),
        Err(e) => Err(e.clone()),
    }
}

/// One group's objects as this engine reads them: those it can, by id, and
/// those a later minor's file holds that its projection leaves out
/// ([`projection`]), each with the W913 a lookup of it raises.
#[derive(Debug)]
struct Listing<T> {
    items: T,
    left_out: IndexMap<String, Error>,
}

impl<T> Listing<T> {
    fn all(items: T) -> Self {
        Listing { items, left_out: IndexMap::new() }
    }
}

/// An object a later minor's projection leaves out: the W913 a lookup of it
/// raises, and the value it holds where that is why, read up to which
/// whatever else is wrong with it is the error it was
/// ([`projection::read_up_to`]).
#[derive(Debug)]
struct Aside {
    error: Error,
    value: Option<projection::Later>,
}

impl Aside {
    /// Left out for the value it holds.
    fn holds(what: &str, id: &str, value: projection::Later, version: &str) -> Self {
        Aside { error: projection::holds(what, id, &value, version), value: Some(value) }
    }

    /// Left out with an object it depends on, and not read.
    fn depends(error: Error) -> Self {
        Aside { error, value: None }
    }
}

/// A sample's annotations, by id.
type Annotations = IndexMap<String, Arc<Annotation>>;
/// A sample's transforms, by id.
type Transforms = IndexMap<String, Transform>;

/// A read-only view of one sample root.
///
/// Every collection is read once per handle: a `Sample` is read-only, and
/// `amend` is copy-on-write --- it replaces the inode, so an open handle never
/// sees an edit.
#[derive(Debug)]
pub struct Sample {
    pub path: Option<PathBuf>,
    pub root: hdf5::Group,
    handle: Option<hdf5::File>,
    document: OnceLock<Result<SampleDocument>>,
    grids: OnceLock<Result<Listing<Arc<Grids>>>>,
    images: OnceLock<Result<Listing<IndexMap<String, Image>>>>,
    annotations: OnceLock<Result<Listing<Annotations>>>,
    transforms: OnceLock<Result<Listing<Arc<Transforms>>>>,
    index: OnceLock<Result<Listing<IndexMap<String, SamplingIndex>>>>,
    fresh: OnceLock<Result<BTreeSet<String>>>,
    clinical: OnceLock<Result<Option<Arc<Clinical>>>>,
    documents: OnceLock<Result<Option<Arc<Documents>>>>,
    resolved: Mutex<HashMap<(String, String), Option<Transform>>>,
}

impl Sample {
    /// A view of a root inside an open file (a standalone sample, or a
    /// collection member).
    pub fn from_root(root: hdf5::Group, handle: Option<hdf5::File>, path: Option<PathBuf>) -> Sample {
        Sample {
            path,
            root,
            handle,
            document: OnceLock::new(),
            grids: OnceLock::new(),
            images: OnceLock::new(),
            annotations: OnceLock::new(),
            transforms: OnceLock::new(),
            index: OnceLock::new(),
            fresh: OnceLock::new(),
            clinical: OnceLock::new(),
            documents: OnceLock::new(),
            resolved: Mutex::new(HashMap::new()),
        }
    }

    /// The open file, when this view owns one.
    pub fn handle(&self) -> Option<&hdf5::File> {
        self.handle.as_ref()
    }

    /// Whether the file this view reads is open: a closed sample's, or a
    /// member's of a closed collection, is not.
    pub fn is_open(&self) -> bool {
        self.root.is_valid()
    }

    /// Release the file, when this view owns it.  Further reads through this
    /// view fail.
    pub fn close(&mut self) {
        if let Some(file) = self.handle.take() {
            let _ = crate::h5::file::close_everything(file, false);
        }
    }

    /// Close the file and everything opened through it, when this view owns
    /// the file --- a member of an open collection does not; its collection
    /// does ([`Collection::get`](crate::collection::Collection::get)).  An
    /// object read from the sample and still held becomes invalid rather than
    /// keeping the file open and locked.
    pub fn close_file(&self) -> Result<()> {
        match &self.handle {
            Some(file) => crate::h5::file::close_everything(file.clone(), false),
            None => Ok(()),
        }
    }

    /// Python's `repr()`.
    pub fn repr(&self) -> Result<String> {
        Ok(format!(
            "Sample({}, {} timepoints, {} images, {} annotations)",
            repr_str(&self.identity()?.sample_id),
            self.timepoints()?.len(),
            self.images()?.len(),
            self.annotations()?.len()
        ))
    }

    // -- document -------------------------------------------------------------

    /// `/meta`, parsed once: a later minor's as its supported projection
    /// ([`read_document`]).
    pub fn document(&self) -> Result<&SampleDocument> {
        cached(&self.root, &self.document, || read_document(&self.root))
    }

    pub fn identity(&self) -> Result<&Identity> {
        Ok(&self.document()?.identity)
    }

    pub fn cohort(&self) -> Result<&Cohort> {
        Ok(&self.document()?.cohort)
    }

    pub fn timepoints(&self) -> Result<&Timeline> {
        Ok(&self.document()?.timepoints)
    }

    pub fn label_set(&self) -> Result<Option<&LabelSet>> {
        Ok(self.document()?.label_set.as_ref())
    }

    pub fn version(&self) -> Result<String> {
        let v = attrs::require(&self.root, "medh5_version", "E001")?;
        Ok(v.as_str().unwrap_or_else(|| attrs::stringify_value(&v)))
    }

    pub fn kind(&self) -> Result<String> {
        Ok(attrs::get_str(&self.root, "medh5_kind")?.unwrap_or_else(|| "sample".into()))
    }

    /// The declared profiles; `{"core"}` when none are declared.
    pub fn profiles(&self) -> Result<BTreeSet<String>> {
        Ok(match attrs::get_strs(&self.root, "medh5_profiles")? {
            None => BTreeSet::from(["core".to_string()]),
            Some(v) => v.into_iter().collect(),
        })
    }

    pub fn content_id(&self) -> Result<Option<String>> {
        attrs::get_str(&self.root, "content_id")
    }

    /// What this engine can do with the sample: [`Full`] for 1.0 and 1.1;
    /// [`Projection`] for a higher minor, which reads only what this engine
    /// knows and is never amended (1.1 §2).
    ///
    /// [`Full`]: crate::version::Support::Full
    /// [`Projection`]: crate::version::Support::Projection
    pub fn support(&self) -> Result<crate::version::Support> {
        Ok(crate::version::support(&self.version()?))
    }

    /// The `clinical` profile's records (1.1 §3--§7), when the sample
    /// declares the profile; `None` otherwise.  Events and links are read
    /// now, document text when asked for.
    pub fn clinical(&self) -> Result<Option<&Arc<Clinical>>> {
        let found = cached(&self.root, &self.clinical, || {
            if !self.profiles()?.contains(crate::clinical::PROFILE) {
                return Ok(None);
            }
            let projection = crate::version::is_projection(&self.version()?);
            Ok(Clinical::open(&self.root, projection)?.map(Arc::new))
        })?;
        Ok(found.as_ref())
    }

    /// The `clinical` profile's documents alone --- metadata and text
    /// offsets, not the events or the links --- when the sample declares the
    /// profile: what reading one report needs (1.1 §6).
    pub fn documents(&self) -> Result<Option<&Arc<Documents>>> {
        let found = cached(&self.root, &self.documents, || {
            if !self.profiles()?.contains(crate::clinical::PROFILE) {
                return Ok(None);
            }
            let projection = crate::version::is_projection(&self.version()?);
            Ok(Documents::open(&self.root, projection)?.map(Arc::new))
        })?;
        Ok(found.as_ref())
    }

    // -- objects -----------------------------------------------------------------

    /// The version a later minor's file declares, when this engine reads the
    /// file as a projection (1.1 §2.2); `None` for a version it implements.
    fn later_minor(&self) -> Option<String> {
        attrs::get_str(&self.root, "medh5_version").ok().flatten().filter(|v| crate::version::is_projection(v))
    }

    /// The objects a later minor's file holds that this engine's projection
    /// leaves out (1.1 §2.2), by path (`grids/g`), each with the W913 a lookup
    /// of it raises: one holding a value only that minor defines, and what
    /// depends on one.  Empty, and nothing read, for a file of a version this
    /// engine implements.
    pub fn left_out(&self) -> Result<IndexMap<String, Error>> {
        let mut out = IndexMap::new();
        if self.later_minor().is_none() {
            return Ok(out);
        }
        let groups = [
            ("grids", &self.grid_listing()?.left_out),
            ("images", &self.image_listing()?.left_out),
            ("annotations", &self.annotation_listing()?.left_out),
            ("transforms", &self.transform_listing()?.left_out),
            ("index", &self.index_listing()?.left_out),
        ];
        for (group, left_out) in groups {
            out.extend(left_out.iter().map(|(name, e)| (format!("{group}/{name}"), e.clone())));
        }
        Ok(out)
    }

    fn grid_listing(&self) -> Result<&Listing<Arc<Grids>>> {
        cached(&self.root, &self.grids, || {
            let Listing { items: stored, left_out } = self.read_grid_listing()?;
            let declared = match self.timepoints() {
                Ok(t) => t.ids(),
                Err(e) if e.is_medh5() => return Ok(Listing { items: Arc::new(stored), left_out }),
                Err(e) => return Err(e),
            };
            if declared.len() != 1 {
                return Ok(Listing { items: Arc::new(stored), left_out });
            }
            let only = declared[0].clone();
            let items = stored
                .into_iter()
                .map(|(gid, g)| {
                    let g = if g.timepoint.is_none() { g.with_timepoint(Some(only.clone())) } else { g };
                    (gid, g)
                })
                .collect();
            Ok(Listing { items: Arc::new(items), left_out })
        })
    }

    /// The stored grids, a later minor's left out where they hold an axis kind
    /// only it defines.
    fn read_grid_listing(&self) -> Result<Listing<Grids>> {
        let Some(version) = self.later_minor() else {
            return Ok(Listing::all(read_grids(&self.root)?.into_iter().collect()));
        };
        let Some(node) = ops::child_group(&self.root, "grids") else {
            return Err(Error::coded("E008", "sample has no `grids` group"));
        };
        let mut out = Listing::all(Grids::new());
        for name in ops::members(&node)? {
            let Some(group) = ops::child_group(&node, &name) else { continue };
            let read = || read_grid(&group, Some(&name));
            match projection::grid_value(&group)? {
                Some(value) => {
                    projection::read_up_to(&value, read)?;
                    out.left_out.insert(name.clone(), projection::holds("grid", &name, &value, &version));
                }
                None => {
                    out.items.insert(name.clone(), read()?);
                }
            }
        }
        Ok(out)
    }

    /// The sample's grids, with §3.7's implicit timepoint resolved: with
    /// exactly one declared timepoint, a grid that names none belongs to it.
    /// A later minor's file lists those its projection reads ([`left_out`]).
    ///
    /// [`left_out`]: Sample::left_out
    pub fn grids(&self) -> Result<&Arc<Grids>> {
        Ok(&self.grid_listing()?.items)
    }

    /// A grid by id: `KeyError` naming the available ones, or W913 for one a
    /// later minor's file holds that this engine's projection leaves out.
    pub fn grid(&self, grid_id: &str) -> Result<&Grid> {
        let listing = self.grid_listing()?;
        listing
            .items
            .get(grid_id)
            .ok_or_else(|| lookup_failure(&listing.left_out, "grid", grid_id, listing.items.keys()))
    }

    /// `grids/ref` when present, else the grid of the first image (§3.2).
    pub fn reference_grid(&self) -> Result<Grid> {
        let listing = self.grid_listing()?;
        if let Some(left_out) = listing.left_out.get("ref") {
            return Err(left_out.clone());
        }
        let grids = &listing.items;
        if let Some(g) = grids.get("ref") {
            return Ok(g.clone());
        }
        let images = self.images()?;
        if let Some((_, first)) = images.first() {
            return Ok(first.grid()?.clone());
        }
        if grids.is_empty() {
            return Err(Error::coded(
                "E111",
                format!(
                    "{} declares no grids, so it has no reference grid",
                    self.path.as_ref().map(|p| p.to_string_lossy().into_owned()).unwrap_or_else(|| "<memory>".into())
                ),
            ));
        }
        let mut names: Vec<&String> = grids.keys().collect();
        names.sort();
        Ok(grids[names[0].as_str()].clone())
    }

    fn image_listing(&self) -> Result<&Listing<IndexMap<String, Image>>> {
        cached(&self.root, &self.images, || {
            let grids = self.grid_listing()?;
            let mut out = Listing::all(IndexMap::new());
            let Some(node) = ops::child_group(&self.root, "images") else {
                return Ok(out);
            };
            for name in ops::members(&node)? {
                let on = if grids.left_out.is_empty() {
                    Vec::new()
                } else if let Some(group) = ops::child_group(&node, &name) {
                    image_grids(&group)?
                } else if let Some(dataset) = ops::child_dataset(&node, &name) {
                    image_grids(&dataset)?
                } else {
                    Vec::new()
                };
                match on.iter().find_map(|gid| grids.left_out.get(gid).map(|e| (gid, e))) {
                    Some((gid, cause)) => {
                        let left_out = projection::depends("image", &name, "lies on", "grid", gid, cause);
                        out.left_out.insert(name, left_out);
                    }
                    None => {
                        let image = Image::open(&name, &node, grids.items.clone())?;
                        out.items.insert(name, image);
                    }
                }
            }
            Ok(out)
        })
    }

    /// The sample's images, by id (name order).  A later minor's file lists
    /// those its projection reads ([`left_out`]).
    ///
    /// [`left_out`]: Sample::left_out
    pub fn images(&self) -> Result<&IndexMap<String, Image>> {
        Ok(&self.image_listing()?.items)
    }

    /// One image: `KeyError` naming the available ones, or W913 for one a
    /// later minor's file holds that this engine's projection leaves out.
    pub fn image(&self, image_id: &str) -> Result<&Image> {
        let listing = self.image_listing()?;
        listing
            .items
            .get(image_id)
            .ok_or_else(|| lookup_failure(&listing.left_out, "image", image_id, listing.items.keys()))
    }

    fn annotation_listing(&self) -> Result<&Listing<Annotations>> {
        cached(&self.root, &self.annotations, || {
            let grids = self.grid_listing()?;
            let label_set = self.label_set()?.cloned().map(Arc::new);
            let mut out = Listing::all(IndexMap::new());
            let Some(node) = ops::child_group(&self.root, "annotations") else {
                return Ok(out);
            };
            let aside = self.annotations_left_out(&node)?;
            for name in ops::members(&node)? {
                let Some(group) = ops::child_group(&node, &name) else { continue };
                let open = || Annotation::open(&name, group.clone(), grids.items.clone(), label_set.clone());
                match aside.get(&name) {
                    None => {
                        out.items.insert(name.clone(), Arc::new(open()?));
                    }
                    Some(Aside { error, value }) => {
                        if let Some(value) = value {
                            projection::read_up_to(value, open)?;
                        }
                        out.left_out.insert(name.clone(), error.clone());
                    }
                }
            }
            Ok(out)
        })
    }

    /// The annotations under `node` a later minor's projection leaves out:
    /// one lying on a grid left out, or holding a value only that minor
    /// defines.  Decided from their attributes, so what indexes an annotation
    /// is told apart without the document the annotations are read with.
    fn annotations_left_out(&self, node: &hdf5::Group) -> Result<IndexMap<String, Aside>> {
        let mut out = IndexMap::new();
        let Some(version) = self.later_minor() else { return Ok(out) };
        let grids = &self.grid_listing()?.left_out;
        for name in ops::members(node)? {
            let Some(group) = ops::child_group(node, &name) else { continue };
            let gid = attrs::get_str(&group, "grid")?;
            let aside = if let Some((gid, cause)) = gid.as_ref().and_then(|g| grids.get(g).map(|e| (g, e))) {
                Aside::depends(projection::depends("annotation", &name, "lies on", "grid", gid, cause))
            } else if let Some(value) = projection::annotation_value(&group)? {
                Aside::holds("annotation", &name, value, &version)
            } else {
                continue;
            };
            out.insert(name, aside);
        }
        Ok(out)
    }

    /// The sample's annotations, by id (name order).  A later minor's file
    /// lists those its projection reads ([`left_out`]).
    ///
    /// [`left_out`]: Sample::left_out
    pub fn annotations(&self) -> Result<&IndexMap<String, Arc<Annotation>>> {
        Ok(&self.annotation_listing()?.items)
    }

    /// One annotation: `KeyError` naming the available ones, or W913 for one
    /// a later minor's file holds that this engine's projection leaves out.
    pub fn annotation(&self, ann_id: &str) -> Result<&Arc<Annotation>> {
        let listing = self.annotation_listing()?;
        listing
            .items
            .get(ann_id)
            .ok_or_else(|| lookup_failure(&listing.left_out, "annotation", ann_id, listing.items.keys()))
    }

    /// The §7.7 ignore region of a voxel annotation, under any encoding: in
    /// band (`labelmap`, `layers`) and/or the `mask` named by
    /// `header.ignore_mask`.  All `false` where none is declared.
    pub fn ignore_region(&self, ann_id: &str, roi: Option<&[Slice]>) -> Result<ArrayD<bool>> {
        let annotation = self.annotation(ann_id)?;
        if !annotation.is_voxel() {
            return Err(Error::invalid(format!(
                "annotation {} is a {}, not a voxel annotation; the §7.7 ignore region is defined on voxels",
                repr_str(ann_id),
                repr_str(annotation.kind())
            )));
        }
        let window = annotation.window(roi)?;
        let shape: Vec<usize> = window.iter().map(|(a, b)| b.saturating_sub(*a)).collect();
        let mut region = ArrayD::from_elem(IxDyn(&shape), false);
        // In band only where the data can hold the ignore id: a training
        // item reads this per patch, and most label volumes hold none.
        if matches!(annotation.kind(), "labelmap" | "layers") && annotation.may_encode_ignore()? {
            let slices = window_slices(&window);
            region.zip_mut_with(&annotation.ignore_mask(Some(&slices))?, |r, v| *r |= *v);
        }
        if let Some(reference) = annotation.header.ignore_mask.clone() {
            let mask = self.referenced_mask(ann_id, "ignore_mask", &reference, &window_slices(&window))?;
            region.zip_mut_with(&mask, |r, v| *r |= *v);
        }
        Ok(region)
    }

    /// Where an image holds data (§4.4): its `valid_mask`, or everywhere.
    ///
    /// `roi` covers the spatial axes, resolved as an annotation's window is:
    /// a negative bound counts from the end and every bound is clipped to its
    /// axis, and a slice with a step other than 1 is refused (`ValueError`),
    /// so the region lines up with the image patch it masks.
    pub fn valid_region(&self, image_id: &str, roi: Option<&[Slice]>) -> Result<ArrayD<bool>> {
        let image = self.image(image_id)?;
        let shape = image.grid()?.spatial_shape();
        let window: Vec<(usize, usize)> = match roi {
            None => shape.iter().map(|n| (0, *n)).collect(),
            Some(roi) => {
                if roi.len() != shape.len() {
                    return Err(Error::invalid(format!(
                        "roi has {} axes; grid has {} spatial axes",
                        roi.len(),
                        shape.len()
                    )));
                }
                roi.iter()
                    .zip(&shape)
                    .map(|(s, n)| {
                        let (a, b, step) = s.resolve(*n)?;
                        if step != 1 {
                            return Err(Error::Value(format!(
                                "image {}: a region of interest is read whole, so a slice with step {step} is \
                                 refused; read the region, then slice the result with the step",
                                repr_str(image_id)
                            )));
                        }
                        Ok((a, b))
                    })
                    .collect::<Result<_>>()?
            }
        };
        match image.valid_mask()? {
            None => {
                let dims: Vec<usize> = window.iter().map(|(a, b)| b - a).collect();
                Ok(ArrayD::from_elem(IxDyn(&dims), true))
            }
            Some(reference) => self.referenced_mask(image_id, "valid_mask", &reference, &window_slices(&window)),
        }
    }

    fn referenced_mask(&self, owner: &str, attr: &str, reference: &str, window: &[Slice]) -> Result<ArrayD<bool>> {
        let name = annotation_id(reference);
        let listing = self.annotation_listing()?;
        if let Some(left_out) = listing.left_out.get(name) {
            return Err(left_out.clone());
        }
        let target = listing.items.get(name).filter(|a| a.kind() == "mask");
        match target {
            None => Err(Error::coded(
                "E413",
                format!(
                    "{}: {attr} names {}, which is not a `mask` annotation in this file",
                    repr_str(owner),
                    repr_str(name)
                ),
            )),
            Some(t) => t.read_mask(Some(window)),
        }
    }

    fn index_listing(&self) -> Result<&Listing<IndexMap<String, SamplingIndex>>> {
        cached(&self.root, &self.index, || {
            let stored = read_indices(&self.root)?;
            let annotations = match ops::child_group(&self.root, "annotations") {
                Some(node) if self.later_minor().is_some() => self.annotations_left_out(&node)?,
                _ => return Ok(Listing::all(stored)),
            };
            // An entry indexes its annotation's voxels.
            let mut out = Listing::all(IndexMap::new());
            for (name, entry) in stored {
                match annotations.get(&name) {
                    Some(Aside { error: cause, .. }) => {
                        let left_out = projection::depends("index entry", &name, "indexes", "annotation", &name, cause);
                        out.left_out.insert(name, left_out);
                    }
                    None => {
                        out.items.insert(name, entry);
                    }
                }
            }
            Ok(out)
        })
    }

    /// The sampling index entries, by annotation id.  A later minor's file
    /// lists those its projection reads ([`left_out`]).
    ///
    /// [`left_out`]: Sample::left_out
    pub fn index(&self) -> Result<&IndexMap<String, SamplingIndex>> {
        Ok(&self.index_listing()?.items)
    }

    /// One index entry: `KeyError`, or W913 for one a later minor's file
    /// holds that this engine's projection leaves out.
    pub fn index_entry(&self, ann_id: &str) -> Result<&SamplingIndex> {
        let listing = self.index_listing()?;
        listing.items.get(ann_id).ok_or_else(|| match listing.left_out.get(ann_id) {
            Some(left_out) => left_out.clone(),
            None => Error::Key(format!("no index entry {}", repr_str(ann_id))),
        })
    }

    /// Index entries whose `source_digest` still matches their source.
    pub fn fresh_indices(&self) -> Result<&BTreeSet<String>> {
        cached(&self.root, &self.fresh, || {
            let stale: BTreeSet<String> = stale_index_entries(&self.root)?.into_iter().collect();
            Ok(self.index()?.keys().filter(|n| !stale.contains(*n)).cloned().collect())
        })
    }

    fn transform_listing(&self) -> Result<&Listing<Arc<Transforms>>> {
        cached(&self.root, &self.transforms, || {
            let grids = self.grid_listing()?;
            let mut groups = transform_groups(&self.root)?;
            let Some(version) = self.later_minor() else {
                return Ok(Listing::all(Arc::new(open_transforms(groups, grids.items.clone())?)));
            };
            let mut left_out: IndexMap<String, Error> = IndexMap::new();
            for (name, group) in &groups {
                // A field is sampled on its grid, a B-spline's lattice laid on its own.
                let mut on = None;
                for key in ["field_grid", "cp_grid"] {
                    let Some(gid) = attrs::get_str(group, key)? else { continue };
                    if let Some(cause) = grids.left_out.get(&gid) {
                        let relation = if key == "field_grid" { "is sampled on" } else { "lays its lattice on" };
                        on = Some(projection::depends("transform", name, relation, "grid", &gid, cause));
                        break;
                    }
                }
                if let Some(depends) = on {
                    left_out.insert(name.clone(), depends);
                } else if let Some(value) = projection::transform_value(group)? {
                    projection::read_up_to(&value, || TransformHeader::read(group))?;
                    left_out.insert(name.clone(), projection::holds("transform", name, &value, &version));
                }
            }
            // A composite is its components: one left out leaves it out, and
            // so on up a chain of composites.
            loop {
                let mut more: Vec<(String, Error)> = Vec::new();
                for (name, group) in &groups {
                    if left_out.contains_key(name) || attrs::get_str(group, "kind")?.as_deref() != Some("composite") {
                        continue;
                    }
                    let components = attrs::get_strs(group, "components")?.unwrap_or_default();
                    if let Some((step, cause)) = components.iter().find_map(|c| left_out.get(c).map(|e| (c, e))) {
                        more.push((
                            name.clone(),
                            projection::depends("transform", name, "composes", "transform", step, cause),
                        ));
                    }
                }
                if more.is_empty() {
                    break;
                }
                left_out.extend(more);
            }
            // What is left out is no sibling an `inverse_id` resolves to.
            groups.retain(|name, _| !left_out.contains_key(name));
            Ok(Listing { items: Arc::new(open_transforms(groups, grids.items.clone())?), left_out })
        })
    }

    /// The file's transforms, read once.  A later minor's file lists those
    /// its projection reads ([`left_out`]).
    ///
    /// [`left_out`]: Sample::left_out
    pub fn transforms(&self) -> Result<&Arc<IndexMap<String, Transform>>> {
        Ok(&self.transform_listing()?.items)
    }

    /// One transform: `KeyError` naming the available ones, or W913 for one
    /// a later minor's file holds that this engine's projection leaves out.
    pub fn transform(&self, transform_id: &str) -> Result<&Transform> {
        let listing = self.transform_listing()?;
        listing
            .items
            .get(transform_id)
            .ok_or_else(|| lookup_failure(&listing.left_out, "transform", transform_id, listing.items.keys()))
    }

    /// The transform relating two timepoints, grids or frames (§10), each
    /// key read as [`Sample::frames_for`] reads it: the same as
    /// [`Sample::transform_between_as`] with `by = None`.
    pub fn transform_between(&self, source: &str, target: &str) -> Result<Option<Transform>> {
        self.transform_between_as(source, target, None)
    }

    /// The transform relating two timepoints, grids or frames (§10), each key
    /// read as `by` says, or, with `None`, as a timepoint first, then a grid,
    /// then a frame uid.
    ///
    /// A visit has an answer only when its framed grids share one frame of
    /// reference: when either key names a visit holding several, no one
    /// transform relates the two, and this is an uncoded validation error
    /// naming each such visit's grids and frames (ask by grid, or resolve the
    /// frames).  Otherwise `None` when the two share a frame, when a side
    /// names no frame, or when no path relates them (1.0 §3.4: alignment is
    /// never assumed); E501 when two shortest paths tie.  An unknown key is a
    /// `KeyError`.
    pub fn transform_between_as(&self, source: &str, target: &str, by: Option<KeyKind>) -> Result<Option<Transform>> {
        let (source_kind, sources) = self.read_key(source, by)?;
        let (target_kind, targets) = self.read_key(target, by)?;
        let mut several: Vec<String> = Vec::new();
        for (key, kind, frames) in [(source, source_kind, &sources), (target, target_kind, &targets)] {
            if kind == KeyKind::Timepoint && frames.len() > 1 {
                let described = self.describe_visit(key)?;
                if !several.contains(&described) {
                    several.push(described);
                }
            }
        }
        if !several.is_empty() {
            return Err(Error::invalid(format!(
                "no one transform relates {} to {}: {}; a visit has an answer only when its grids share one frame \
                 of reference, so ask about two grids (by=\"grid\") or two frames (resolve_frames)",
                repr_str(source),
                repr_str(target),
                several.join("; ")
            )));
        }
        match (sources.first(), targets.first()) {
            (Some(a), Some(b)) if a == b => Ok(None),
            (Some(a), Some(b)) => self.resolve_frames(a, b),
            _ => Ok(None),
        }
    }

    /// `visit 'tp0' holds grids 'ct0' (frame 'F0'), 'pet0' (frame 'F1')`.
    fn describe_visit(&self, timepoint: &str) -> Result<String> {
        let grids: Vec<String> = self
            .grids()?
            .iter()
            .filter(|(_, g)| g.timepoint.as_deref() == Some(timepoint))
            .filter_map(|(id, g)| {
                g.frame_uid
                    .as_deref()
                    .filter(|f| !f.is_empty())
                    .map(|f| format!("{} (frame {})", repr_str(id), repr_str(f)))
            })
            .collect();
        Ok(format!("visit {} holds grids {}", repr_str(timepoint), grids.join(", ")))
    }

    /// The transform relating two frame uids, resolved once per handle.
    pub fn resolve_frames(&self, from_frame: &str, to_frame: &str) -> Result<Option<Transform>> {
        crate::h5::alive(&self.root)?;
        if from_frame == to_frame {
            return Ok(None);
        }
        let key = (from_frame.to_string(), to_frame.to_string());
        if let Some(found) = self.resolved.lock().unwrap_or_else(PoisonError::into_inner).get(&key) {
            return Ok(found.clone());
        }
        let found = resolve_between(self.transforms()?, from_frame, to_frame)?;
        self.resolved.lock().unwrap_or_else(PoisonError::into_inner).insert(key, found.clone());
        Ok(found)
    }

    /// The frames a key names: a timepoint's (every grid's at that visit),
    /// else a grid's own, else the key as a frame uid --- in that order,
    /// because the namespaces are separate (§2.3) and may collide.  A grid a
    /// later minor's projection leaves out names no frame: W913.
    pub fn frames_for(&self, key: &str) -> Result<Vec<String>> {
        Ok(self.read_key(key, None)?.1)
    }

    /// What a key is read as, and the frames it names: as `by` says, or the
    /// first of a timepoint, a grid and a frame uid that it is.
    fn read_key(&self, key: &str, by: Option<KeyKind>) -> Result<(KeyKind, Vec<String>)> {
        let timeline = self.timepoints()?;
        if matches!(by, None | Some(KeyKind::Timepoint)) && timeline.contains(key) {
            return Ok((KeyKind::Timepoint, frames_of_timepoint(self.grids()?, key)));
        }
        let grids = self.grid_listing()?;
        if matches!(by, None | Some(KeyKind::Grid)) {
            if let Some(grid) = grids.items.get(key) {
                return Ok((KeyKind::Grid, grid.frame_uid.iter().filter(|f| !f.is_empty()).cloned().collect()));
            }
            if let Some(left_out) = grids.left_out.get(key) {
                return Err(left_out.clone());
            }
        }
        if matches!(by, None | Some(KeyKind::Frame)) && self.known_frames()?.contains(key) {
            return Ok((KeyKind::Frame, vec![key.to_string()]));
        }
        Err(match by {
            None => Error::Key(format!(
                "{} is not a timepoint, a grid or a frame of reference in this sample (timepoints: {})",
                repr_str(key),
                repr_list(&timeline.ids())
            )),
            Some(KeyKind::Timepoint) => Error::Key(format!(
                "{} is not a timepoint of this sample (timepoints: {})",
                repr_str(key),
                repr_list(&timeline.ids())
            )),
            Some(KeyKind::Grid) => missing_key("grid", key, grids.items.keys().cloned()),
            Some(KeyKind::Frame) => missing_key("frame of reference", key, self.known_frames()?),
        })
    }

    fn known_frames(&self) -> Result<BTreeSet<String>> {
        let mut frames: BTreeSet<String> =
            self.grids()?.values().filter_map(|g| g.frame_uid.clone()).filter(|f| !f.is_empty()).collect();
        for t in self.transforms()?.values() {
            frames.insert(t.from_frame().to_string());
            frames.insert(t.to_frame().to_string());
        }
        if let Some(node) = ops::child_group(&self.root, "annotations") {
            for name in ops::members(&node)? {
                if let Some(g) = ops::child_group(&node, &name) {
                    if let Some(f) = attrs::get_str(&g, "frame_uid")? {
                        frames.insert(f);
                    }
                }
            }
        }
        Ok(frames)
    }

    pub fn is_longitudinal(&self) -> Result<bool> {
        Ok(self.timepoints()?.is_longitudinal())
    }

    /// Image ids on a timepoint, in name order.
    pub fn images_at(&self, timepoint: &str) -> Result<Vec<String>> {
        let mut out = Vec::new();
        for (name, image) in self.images()? {
            if image.timepoint()?.as_deref() == Some(timepoint) {
                out.push(name.clone());
            }
        }
        Ok(out)
    }

    /// Annotation ids pertaining to a timepoint, in name order.
    pub fn annotations_at(&self, timepoint: &str) -> Result<Vec<String>> {
        Ok(self
            .annotations()?
            .iter()
            .filter(|(_, a)| a.timepoints().iter().any(|t| t == timepoint))
            .map(|(n, _)| n.clone())
            .collect())
    }

    // -- integrity -------------------------------------------------------------------

    /// Object path -> the attribute names `content_id` covers (§13.2).
    pub fn attr_name_map(&self) -> Result<AttrNameMap> {
        attr_name_map_of(&self.root)
    }

    /// Verify digests, `content_id` and index currency.
    pub fn verify(&self, partial: Option<&[String]>) -> Result<VerifyResult> {
        if partial.is_none() {
            if let Some(algo) = crate::sample::writer::unknown_algo(&self.root)? {
                let shown = self.path.clone().unwrap_or_else(|| PathBuf::from(self.root.name()));
                return Err(crate::integrity::repair::unknown_algo_refused(&shown, &algo));
            }
        }
        verify_root(&self.root, Some(&self.attr_name_map()?), partial, true)
    }

    /// Recompute the `content_id` from the stored digests.
    pub fn compute_content_id(&self) -> Result<String> {
        let algo = attrs::get_str(&self.root, "digest_algo")?.unwrap_or_else(|| "sha256".into());
        compute_content_id(&self.root, &self.attr_name_map()?, &algo, None)
    }

    // -- reporting ---------------------------------------------------------------------

    pub fn summary(&self) -> Result<Value> {
        let mut out = Map::new();
        out.insert("path".into(), json!(self.path.as_ref().map(|p| p.to_string_lossy().into_owned())));
        out.insert("version".into(), json!(self.version()?));
        out.insert("kind".into(), json!(self.kind()?));
        out.insert("profiles".into(), json!(self.profiles()?.into_iter().collect::<Vec<_>>()));
        out.insert("content_id".into(), json!(self.content_id()?));
        if let Value::Object(doc) = self.document()?.summary() {
            out.extend(doc);
        }
        out.insert("grids".into(), Value::Array(self.grids()?.values().map(Grid::summary).collect()));
        out.insert("images".into(), Value::Array(self.images()?.values().map(Image::summary).collect::<Result<_>>()?));
        out.insert(
            "annotations".into(),
            Value::Array(self.annotations()?.values().map(|a| a.summary()).collect::<Result<_>>()?),
        );
        out.insert(
            "transforms".into(),
            Value::Array(self.transforms()?.values().map(Transform::summary).collect::<Result<_>>()?),
        );
        let mut index: Vec<String> = self.index()?.keys().cloned().collect();
        index.sort();
        out.insert("index".into(), json!(index));
        if let Some(clinical) = self.clinical()? {
            out.insert("clinical".into(), clinical.summary());
        }
        let left_out = self.left_out()?;
        if !left_out.is_empty() {
            let why: Map<String, Value> = left_out.into_iter().map(|(path, e)| (path, json!(e.message()))).collect();
            out.insert("left_out".into(), Value::Object(why));
        }
        Ok(Value::Object(out))
    }
}

fn window_slices(window: &[(usize, usize)]) -> Vec<Slice> {
    window.iter().map(|(a, b)| Slice::new(*a as i64, *b as i64)).collect()
}

/// Read the raw `/meta` payload as text.
pub fn read_document_text(root: &hdf5::Group) -> Result<String> {
    crate::h5::alive(root)?;
    match ops::child_dataset(root, META_DATASET) {
        None => Err(Error::Schema(format!("{}: required dataset `meta` is absent", root.name()))),
        Some(ds) => data::read_scalar_string(&ds),
    }
}

/// Read and parse `<root>/meta`: exactly in a version this engine
/// implements, and in a higher minor as its supported projection, which a
/// reader must accept (1.1 §2.2, [`SampleDocument::loads_version`]).
pub fn read_document(root: &hdf5::Group) -> Result<SampleDocument> {
    let version = attrs::get_str(root, "medh5_version").ok().flatten();
    SampleDocument::loads_version(&read_document_text(root)?, version.as_deref())
}

/// Every frame-of-reference UID in a file, and the attributes naming it.
pub fn frame_references(root: &hdf5::Group) -> Result<IndexMap<String, Vec<String>>> {
    let mut out: std::collections::BTreeMap<String, Vec<String>> = Default::default();
    for (group, names) in FRAME_ATTRS {
        let Some(node) = ops::child_group(root, group) else { continue };
        for name in ops::members(&node)? {
            let loc: Option<hdf5::Location> = match ops::node_kind(&node, &name) {
                Some(ops::NodeKind::Group) => Some((*node.group(&name)?).clone()),
                Some(ops::NodeKind::Dataset) => {
                    let d = node.dataset(&name)?;
                    Some((**d).clone())
                }
                _ => None,
            };
            let Some(loc) = loc else { continue };
            for attr in names {
                let uid = attrs::get_str(&loc, attr)?.unwrap_or_default();
                if !uid.is_empty() {
                    out.entry(uid).or_default().push(format!("{group}.{name}.{attr}"));
                }
            }
        }
    }
    Ok(out.into_iter().collect())
}

/// Object path -> the attribute names `content_id` covers (§13.2): one fixed
/// list per kind of object, which is not every attribute the specification
/// defines (§8.5's `correspondence` is not covered).
///
/// Derived from the file, and shared by the reader and the writer, so the two
/// agree by construction.
pub fn attr_name_map_of(root: &hdf5::Group) -> Result<AttrNameMap> {
    let mut out = AttrNameMap::new();
    out.insert(String::new(), ROOT_DIGEST_ATTRS.to_vec());
    let groups: [(&str, &[&'static str]); 4] = [
        ("grids", &SPEC_GRID_ATTRS),
        ("images", &SPEC_IMAGE_ATTRS),
        ("annotations", &SPEC_ANNOTATION_ATTRS),
        ("transforms", &SPEC_TRANSFORM_ATTRS),
    ];
    for (group, names) in groups {
        let Some(node) = ops::child_group(root, group) else { continue };
        let names: Vec<&'static str> = names.iter().copied().filter(|a| *a != "digest").collect();
        for name in ops::members(&node)? {
            out.insert(format!("{group}/{name}"), names.clone());
        }
    }
    Ok(out)
}

/// An annotation id from a reference that may be written as a path.
pub fn annotation_id(reference: &str) -> &str {
    reference.strip_prefix("annotations/").unwrap_or(reference)
}

/// The file's `medh5_version`, or a refusal of a major this reader lacks.
pub fn require_major(root: &hdf5::Location, path: &Path) -> Result<String> {
    let shown = repr_str(&path.to_string_lossy());
    let Some(version) = attrs::read(root, "medh5_version")? else {
        return Err(Error::Version(format!(
            "{shown} declares no `medh5_version`; a 0.x file must be converted with `medh5 migrate`"
        )));
    };
    let text = match version {
        AttrValue::Str(s) => s,
        other => attrs::stringify_value(&other),
    };
    // `support` decides: the major is the text before the first `.`, so a
    // version it calls unsupported is the one every reader refuses.
    if crate::version::support(&text) == crate::version::Support::Unsupported {
        return Err(Error::Version(format!("{shown} is MEDH5 {text}; this reader implements {FORMAT_VERSION}")));
    }
    Ok(text)
}

/// Open a `.medh5` sample file, read-only.
pub fn open_sample(path: &Path) -> Result<Sample> {
    let handle = file::open_read(path)?;
    require_major(&handle, path)?;
    let kind = attrs::get_str(&handle, "medh5_kind")?.unwrap_or_else(|| "sample".into());
    if kind != "sample" {
        return Err(Error::File(format!(
            "{} is a {}; open it with open_collection()",
            repr_str(&path.to_string_lossy()),
            repr_str(&kind)
        )));
    }
    let root = handle.as_group()?;
    Ok(Sample::from_root(root, Some(handle), Some(path.to_path_buf())))
}

/// Rewrite `path` so freed space is not carried forward (§14.4).
///
/// HDF5 does not reclaim storage: an amend that copies an object and *then*
/// rewrites one of its attributes leaves the superseded value physically in
/// the file, where `strings` still finds it.  For de-identification that is
/// the difference between a pseudonymised file and one still carrying the
/// original UID.  The live graph is copied into a fresh file
/// ([`crate::h5::graph`]); filters, chunking, attributes and every link come
/// across untouched, so every digest and the `content_id` survive.
pub fn repack(path: &Path) -> Result<()> {
    crate::h5::file::atomic_rewrite(path, None, |src, dst| {
        require_major(src, path)?;
        let (src_root, dst_root) = (src.as_group()?, dst.as_group()?);
        ops::refuse_references(&src_root, "a repack")?;
        crate::h5::graph::copy_root_raw(&src_root, &dst_root, &[])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array::NdArray;

    /// A repack rewrites storage, not content: the digests, the `content_id`,
    /// every image's values and the file's permissions come through.
    #[test]
    fn repack_preserves_content_and_permissions() {
        let dir = tempfile::tempdir().unwrap();
        let path = crate::bench::synthetic_pair(dir.path(), &[8, 12, 10], "portable", 7).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
        }
        let snapshot = |path: &Path| {
            let sample = open_sample(path).unwrap();
            let images: Vec<(String, NdArray)> = sample
                .images()
                .unwrap()
                .iter()
                .map(|(id, image)| (id.clone(), image.read(None, false, None).unwrap()))
                .collect();
            (sample.content_id().unwrap(), images)
        };
        let before = snapshot(&path);
        repack(&path).unwrap();
        assert_eq!(snapshot(&path), before);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&path).unwrap().permissions().mode() & 0o777, 0o600);
        }
    }

    /// A grid the sample lacks is a `KeyError` naming the ones it has, as
    /// every other lookup by id is; it named only the missing one.
    #[test]
    fn s3_a_missing_grid_names_the_grids_there_are() {
        let dir = tempfile::tempdir().unwrap();
        let path = crate::bench::synthetic_pair(dir.path(), &[8, 12, 10], "portable", 7).unwrap();
        let sample = open_sample(&path).unwrap();
        let mut known: Vec<String> = sample.grids().unwrap().keys().cloned().collect();
        known.sort();
        assert!(!known.is_empty());
        let Err(Error::Key(message)) = sample.grid("nope") else { panic!("a missing grid is a KeyError") };
        assert_eq!(message, format!("no grid 'nope'; available: {}", repr_list(&known)));
    }

    fn refused_as_closed<T>(what: &str, result: Result<T>) {
        match result {
            Err(Error::File(message)) => assert_eq!(message, crate::h5::CLOSED, "{what}"),
            Err(other) => panic!("{what}: {other} ({:?}), not the closed-file error", other.code()),
            Ok(_) => panic!("{what} answered from a closed file"),
        }
    }

    /// An annotation, a transform or a root a Rust caller keeps past the
    /// close of its file refuses with the closed-file error.  HDF5 finds no
    /// member in a closed file, which read as an annotation without `data`
    /// (E410) or masks, a transform without its parameters (E502), and a
    /// sample without grids (E008) or transforms.
    #[test]
    fn s14_4_a_view_kept_past_the_close_refuses_rather_than_reads_empty() {
        let dir = tempfile::tempdir().unwrap();
        let shape = [8, 12, 10];
        let labelled =
            crate::bench::synthetic_sample(dir.path(), &shape, 2, "portable", false, 7, "labelled.medh5").unwrap();
        let pair = crate::bench::synthetic_pair(dir.path(), &shape, "portable", 7).unwrap();
        let mut writer = crate::sample::amend(&pair, None).unwrap();
        let rigid = crate::sample::writer_annotations::TransformSpec {
            matrix: Some(ndarray::Array2::<f64>::eye(4).into_dyn()),
            ..Default::default()
        };
        writer.add_transform("rigid", "affine", "bench:frame-0", "bench:frame-1", rigid).unwrap();
        writer.commit(true).unwrap();

        let sample = open_sample(&labelled).unwrap();
        let organs = sample.annotation("organs").unwrap();
        organs.dataset("data").unwrap();
        sample.close_file().unwrap();
        refused_as_closed("Annotation::dataset", organs.dataset("data"));
        refused_as_closed("Annotation::optional_dataset", organs.optional_dataset("data"));
        refused_as_closed("Annotation::has_masks", organs.has_masks());
        refused_as_closed("read_grids", read_grids(&sample.root));

        let sample = open_sample(&pair).unwrap();
        let (rigid, warp) = (sample.transform("rigid").unwrap(), sample.transform("warp").unwrap());
        rigid.matrix().unwrap();
        warp.field().unwrap();
        sample.close_file().unwrap();
        refused_as_closed("Transform::matrix", rigid.matrix());
        refused_as_closed("Transform::field", warp.field());
        refused_as_closed("transform_groups", transform_groups(&sample.root));
    }

    /// `grids/g_tp0` given an axis kind only a later minor defines, and the
    /// file `version`.
    fn with_a_later_axis_kind(path: &Path, version: &str) {
        crate::h5::init();
        let file = crate::h5::file::marked(|| hdf5::File::open_rw(path)).unwrap();
        let root = file.as_group().unwrap();
        let kinds = ["depth", "spatial", "spatial"].map(String::from).to_vec();
        attrs::write(&root.group("grids/g_tp0").unwrap(), "axis_kinds", &AttrValue::Strs(kinds)).unwrap();
        attrs::write(&root, "medh5_version", &AttrValue::Str(version.into())).unwrap();
        drop(root);
        file.close().unwrap();
    }

    /// 1.1 §2.2: a grid of an axis kind only a later minor defines is left
    /// out of the projection with what lies on it --- its image, and the
    /// field sampled on it --- and a lookup of any of them is W913, the
    /// frames the grid's id names included; in a file of 1.1 it is the E110
    /// it was.
    #[test]
    fn s2_2_a_later_minors_grid_is_left_out_with_what_lies_on_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = crate::bench::synthetic_pair(dir.path(), &[4, 6, 5], "portable", 7).unwrap();
        with_a_later_axis_kind(&path, &crate::version::later_minor());
        let sample = open_sample(&path).unwrap();
        let left_out: Vec<String> = sample.left_out().unwrap().into_keys().collect();
        assert_eq!(left_out, ["grids/g_tp0", "images/CT_tp0", "transforms/warp"]);
        assert_eq!(sample.grids().unwrap().keys().collect::<Vec<_>>(), ["g_tp1"]);
        assert_eq!(sample.images().unwrap().keys().collect::<Vec<_>>(), ["CT_tp1"]);
        assert!(sample.transforms().unwrap().is_empty());
        let lookups = [sample.grid("g_tp0").err(), sample.frames_for("g_tp0").err(), sample.transform("warp").err()];
        for failed in lookups {
            let failed = failed.expect("a lookup of what is left out fails");
            assert_eq!(failed.code(), Some("W913"), "{failed:?}");
        }
        assert_eq!(sample.frames_for("g_tp1").unwrap(), ["bench:frame-1"]);
        drop(sample);

        with_a_later_axis_kind(&path, "1.1");
        let sample = open_sample(&path).unwrap();
        assert!(sample.left_out().unwrap().is_empty());
        assert_eq!(sample.grids().unwrap_err().code(), Some("E110"));
    }
}
