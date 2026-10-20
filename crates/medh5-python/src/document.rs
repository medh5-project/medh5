//! `medh5.document`: the `/meta` sample document (spec §2.4).

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde_json::{Map, Value};

use medh5::document as engine;

use crate::convert::{json_to_py, py_to_json};
use crate::curation::{
    timeline_arg, Activity, Cohort, Deidentification, Identity, QualityRecord, SplitClaim, Timeline, Timepoint,
};
use crate::errors::R;
use crate::exclusive::Exclusive;
use crate::labels::{label_set_arg, with_classes, LabelClass, LabelSet};

/// Where a document object's fields live.
enum Source {
    /// Its own copy: a document read from a file or built by hand, changed by
    /// one thread at a time (E4).
    Owned(Exclusive<Box<engine::SampleDocument>>),
    /// A writer's document, read and edited in place: `SampleWriter.document`,
    /// as 1.x had it, so `w.document.label_set = ...` reaches the file.  The
    /// writer's lock is its lock.
    Writer(Py<crate::writer::SampleWriter>),
}

/// The whole document, typed.  Fields read and write the engine's types.
#[pyclass(module = "medh5.document", name = "SampleDocument", skip_from_py_object, frozen)]
pub struct SampleDocument {
    source: Source,
}

fn object(value: Value) -> Map<String, Value> {
    match value {
        Value::Object(m) => m,
        _ => Map::new(),
    }
}

impl SampleDocument {
    pub fn owned(inner: engine::SampleDocument) -> Self {
        SampleDocument { source: Source::Owned(Exclusive::new("a SampleDocument", Box::new(inner))) }
    }

    /// The live document of `writer`.
    pub fn of_writer(writer: Py<crate::writer::SampleWriter>) -> Self {
        SampleDocument { source: Source::Writer(writer) }
    }

    /// Whether this is the live document of the writer at `writer`.
    pub fn is_view_of(&self, writer: *mut pyo3::ffi::PyObject) -> bool {
        matches!(&self.source, Source::Writer(w) if w.as_ptr() == writer)
    }

    fn read<T>(&self, py: Python<'_>, f: impl FnOnce(&engine::SampleDocument) -> T) -> R<T> {
        match &self.source {
            Source::Owned(doc) => Ok(f(&**doc.read()?)),
            Source::Writer(w) => w.bind(py).get().read_document(f),
        }
    }

    pub(crate) fn edit<T>(&self, py: Python<'_>, f: impl FnOnce(&mut engine::SampleDocument) -> T) -> R<T> {
        match &self.source {
            Source::Owned(doc) => Ok(f(&mut **doc.write()?)),
            Source::Writer(w) => w.bind(py).get().edit_document(f),
        }
    }

    /// A copy of the document as it stands.
    pub fn current(&self, py: Python<'_>) -> R<engine::SampleDocument> {
        self.read(py, |doc| doc.clone())
    }
}

/// A free-form member mapping, which a caller edits in place (`medh5._live`).
#[derive(Clone, Copy)]
enum Members {
    Quality,
    Acquisition,
    Extra,
}

impl Members {
    fn of(field: &str) -> PyResult<Members> {
        match field {
            "quality" => Ok(Members::Quality),
            "acquisition" => Ok(Members::Acquisition),
            "extra" => Ok(Members::Extra),
            other => Err(PyKeyError::new_err(other.to_string())),
        }
    }

    /// The JSON members of `acquisition` or `extra`.
    fn json(self, doc: &engine::SampleDocument) -> &Map<String, Value> {
        match self {
            Members::Acquisition => &doc.acquisition,
            _ => &doc.extra,
        }
    }

    /// The JSON members of `acquisition` or `extra`, to edit.
    fn json_mut(self, doc: &mut engine::SampleDocument) -> &mut Map<String, Value> {
        match self {
            Members::Acquisition => &mut doc.acquisition,
            _ => &mut doc.extra,
        }
    }

    /// A copy of the members, taken under the document's guard and handed
    /// to Python after it.
    fn held(self, doc: &engine::SampleDocument) -> Held {
        match self {
            Members::Quality => Held::Quality(doc.quality.clone()),
            members => Held::Json(members.json(doc).clone()),
        }
    }
}

/// The members of one mapping as the document held them.
enum Held {
    Quality(indexmap::IndexMap<String, medh5::curation::quality::QualityRecord>),
    Json(Map<String, Value>),
}

impl Held {
    /// `(key, value, token)` for each member, the token its canonical JSON.
    fn list(self, py: Python<'_>) -> R<Bound<'_, PyList>> {
        let out = PyList::empty(py);
        match self {
            Held::Quality(records) => {
                for (k, v) in records {
                    let token = medh5::json::canonical(&v.to_json());
                    out.append((k, QualityRecord::wrap(v), token))?;
                }
            }
            Held::Json(found) => {
                for (k, v) in found {
                    out.append((k, json_to_py(py, &v)?, medh5::json::canonical(&v)))?;
                }
            }
        }
        Ok(out)
    }
}

/// The live view of one free-form member mapping of `doc`.
fn live_members<'py>(doc: &Bound<'py, SampleDocument>, field: &str) -> PyResult<Bound<'py, PyAny>> {
    members_view(doc.as_any(), field, None)
}

/// The view of `field` of `doc`, holding `members` (as `_members` lists
/// them) when they were read already, or reading them.
fn members_view<'py>(
    doc: &Bound<'py, PyAny>,
    field: &str,
    members: Option<Bound<'py, PyList>>,
) -> PyResult<Bound<'py, PyAny>> {
    static MEMBERS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();
    let py = doc.py();
    let class = MEMBERS.get_or_try_init(py, || py.import("medh5._live")?.getattr("Members").map(Bound::unbind))?;
    class.bind(py).call_method1("_of", (doc, field, members))
}

/// The `acquisition` view of `doc`, holding `acquisition`: what an edit left,
/// read under the guard it took, so the view is not refused after it.
pub(crate) fn acquisition_view<'py>(doc: &Bound<'py, PyAny>, acquisition: Map<String, Value>) -> R<Bound<'py, PyAny>> {
    let members = Held::Json(acquisition).list(doc.py())?;
    Ok(members_view(doc, "acquisition", Some(members))?)
}

/// `f` under the live views' lock (`medh5._live._LOCK`), which every change
/// to a container nested in a document member holds with its rewrite: what
/// `f` reads of those containers is one state of them, not a mix of two.
/// `f` must not wait for another thread.
pub fn under_live_lock<T>(py: Python<'_>, f: impl FnOnce() -> R<T>) -> R<T> {
    under_live_lock_if(py, true, f)
}

/// `f`, under the live views' lock if `anchored`: if what it reads came from
/// a document, whose containers' edits take that lock.  What no document
/// handed out holds the caller's own containers, which the lock never
/// guards, and the caller's objects, whose conversion is the caller's code:
/// that is read without it.
pub fn under_live_lock_if<T>(py: Python<'_>, anchored: bool, f: impl FnOnce() -> R<T>) -> R<T> {
    if !anchored {
        return f();
    }
    // Looked up on each use: a forked child replaces it (`_after_fork_in_child`).
    static LIVE: PyOnceLock<Py<PyModule>> = PyOnceLock::new();
    let lock = LIVE.get_or_try_init(py, || py.import("medh5._live").map(Bound::unbind))?.bind(py).getattr("_LOCK")?;
    lock.call_method0("acquire")?;
    let out = f();
    lock.call_method0("release")?;
    out
}

/// How a member is named to a caller: `document.identity`,
/// `document.extra['tool']`, `document.provenance.activity('act_1')`.
fn member_name(field: &str, key: Option<&str>) -> String {
    match (field, key) {
        (_, None) => format!("document.{field}"),
        ("provenance.activities", Some(id)) => format!("document.provenance.activity({})", medh5::json::repr_str(id)),
        (_, Some(key)) => format!("document.{field}[{}]", medh5::json::repr_str(key)),
    }
}

fn changed(field: &str, key: Option<&str>) -> medh5::Error {
    medh5::Error::Runtime(format!(
        "`{}` has changed since it was read, and an edit to what was read would undo that; \
         read it again and edit that",
        member_name(field, key)
    ))
}

/// Replace `slot` with `value`, if it is still what was read as `token`.
fn swap<T>(slot: &mut T, dump: impl Fn(&T) -> Value, token: &str, value: T) -> bool {
    if medh5::json::canonical(&dump(slot)) != token {
        return false;
    }
    *slot = value;
    true
}

impl SampleDocument {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] = &[
        "identity",
        "timepoints",
        "cohort",
        "label_set",
        "provenance",
        "quality",
        "splits",
        "acquisition",
        "deidentification",
        "extra",
    ];
}

#[pymethods]
impl SampleDocument {
    #[classattr]
    fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        crate::geometry::dataclass_fields(py, Self::FIELDS)
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::values::match_args(py, Self::FIELDS)
    }

    /// `copy.replace(obj, **changes)`.
    #[pyo3(signature = (**changes))]
    fn __replace__(slf: &Bound<'_, Self>, changes: Option<&Bound<'_, pyo3::types::PyDict>>) -> PyResult<Py<PyAny>> {
        crate::values::dataclass_replace(slf.as_any(), changes)
    }

    #[new]
    #[pyo3(signature = (identity, timepoints, cohort=None, label_set=None, provenance=None, quality=None, splits=None, acquisition=None, deidentification=None, extra=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        identity: &Bound<'_, PyAny>,
        timepoints: &Bound<'_, PyAny>,
        cohort: Option<&Bound<'_, PyAny>>,
        label_set: Option<&Bound<'_, PyAny>>,
        provenance: Option<&Bound<'_, PyAny>>,
        quality: Option<&Bound<'_, PyAny>>,
        splits: Option<&Bound<'_, PyAny>>,
        acquisition: Option<&Bound<'_, PyAny>>,
        deidentification: Option<&Bound<'_, PyAny>>,
        extra: Option<&Bound<'_, PyAny>>,
    ) -> R<Self> {
        let mut doc = engine::SampleDocument::new(Identity::arg(identity)?, timeline_arg(timepoints)?);
        let me = &mut doc;
        if let Some(c) = cohort.filter(|c| !c.is_none()) {
            me.cohort = Cohort::arg(c)?;
        }
        if let Some(ls) = label_set.filter(|c| !c.is_none()) {
            me.label_set = Some(label_set_arg(ls)?);
        }
        if let Some(p) = provenance.filter(|c| !c.is_none()) {
            me.provenance = provenance_arg(p)?;
        }
        if let Some(q) = quality.filter(|c| !c.is_none()) {
            me.quality = quality_arg(q)?;
        }
        if let Some(s) = splits.filter(|c| !c.is_none()) {
            me.splits = s.try_iter()?.map(|i| SplitClaim::arg(&i?)).collect::<R<Vec<_>>>()?;
        }
        if let Some(a) = acquisition.filter(|c| !c.is_none()) {
            me.acquisition = object(py_to_json(a)?);
        }
        if let Some(d) = deidentification.filter(|c| !c.is_none()) {
            me.deidentification = Some(Deidentification::arg(d)?);
        }
        if let Some(e) = extra.filter(|c| !c.is_none()) {
            me.extra = object(py_to_json(e)?);
        }
        Ok(SampleDocument::owned(doc))
    }
    /// The identity; an edit to its `extra` reaches the document.
    #[getter]
    fn identity(slf: &Bound<'_, Self>) -> R<Identity> {
        let found = slf.get().read(slf.py(), |d| d.identity.clone())?;
        Ok(Identity::read_from(found, slf.as_any(), "identity", None))
    }
    #[setter]
    fn set_identity(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = Identity::arg(value)?;
        self.edit(py, |d| d.identity = value)
    }
    /// The timepoints; an edit to one's `series_uids` reaches the document.
    #[getter]
    fn timepoints(slf: &Bound<'_, Self>) -> R<Timeline> {
        let found = slf.get().read(slf.py(), |d| d.timepoints.clone())?;
        Ok(Timeline::read_from(found, slf.as_any()))
    }
    #[setter]
    fn set_timepoints(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = timeline_arg(value)?;
        self.edit(py, |d| d.timepoints = value)
    }
    /// The cohort; an edit to its `extra` reaches the document.
    #[getter]
    fn cohort(slf: &Bound<'_, Self>) -> R<Cohort> {
        let found = slf.get().read(slf.py(), |d| d.cohort.clone())?;
        Ok(Cohort::read_from(found, slf.as_any(), "cohort", None))
    }
    #[setter]
    fn set_cohort(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = Cohort::arg(value)?;
        self.edit(py, |d| d.cohort = value)
    }
    /// The label set; an edit to a class's `properties` reaches the
    /// document.
    #[getter]
    fn label_set(slf: &Bound<'_, Self>) -> R<Option<LabelSet>> {
        let found = slf.get().read(slf.py(), |d| d.label_set.clone())?;
        Ok(found.map(|set| LabelSet::read_from(set, slf.as_any())))
    }
    #[setter]
    fn set_label_set(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = if value.is_none() { None } else { Some(label_set_arg(value)?) };
        self.edit(py, |d| d.label_set = value)
    }
    /// The provenance graph; an agent or activity added to it, and an edit
    /// to an activity's `params`, reach the document.
    #[getter]
    fn provenance(slf: &Bound<'_, Self>) -> R<crate::curation::Provenance> {
        let found = slf.get().read(slf.py(), |d| d.provenance.clone())?;
        Ok(crate::curation::Provenance::read_from(found, slf.as_any()))
    }
    #[setter]
    fn set_provenance(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = provenance_arg(value)?;
        self.edit(py, |d| d.provenance = value)
    }
    /// Review records by key: a dict whose edits reach the document.
    #[getter]
    fn quality<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        live_members(slf, "quality")
    }
    #[setter]
    fn set_quality(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = quality_arg(value)?;
        self.edit(py, |d| d.quality = value)
    }
    #[getter]
    fn splits<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        let claims = self.read(py, |d| d.splits.clone())?;
        Ok(PyTuple::new(py, claims.into_iter().map(SplitClaim::wrap))?)
    }
    #[setter]
    fn set_splits(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = value.try_iter()?.map(|i| SplitClaim::arg(&i?)).collect::<R<Vec<_>>>()?;
        self.edit(py, |d| d.splits = value)
    }
    /// Acquisition parameters by image id: a dict whose edits, nested ones
    /// too, reach the document.
    #[getter]
    fn acquisition<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        live_members(slf, "acquisition")
    }
    #[setter]
    fn set_acquisition(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = object(py_to_json(value)?);
        self.edit(py, |d| d.acquisition = value)
    }
    /// The de-identification record; an edit to its `extra` reaches the
    /// document.
    #[getter]
    fn deidentification(slf: &Bound<'_, Self>) -> R<Option<Deidentification>> {
        let found = slf.get().read(slf.py(), |d| d.deidentification.clone())?;
        Ok(found.map(|record| Deidentification::read_from(record, slf.as_any(), "deidentification", None)))
    }
    #[setter]
    fn set_deidentification(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = if value.is_none() { None } else { Some(Deidentification::arg(value)?) };
        self.edit(py, |d| d.deidentification = value)
    }
    /// Namespaced extensions: a dict whose edits, nested ones too, reach the
    /// document.
    #[getter]
    fn extra<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        live_members(slf, "extra")
    }
    #[setter]
    fn set_extra(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> R<()> {
        let value = object(py_to_json(value)?);
        self.edit(py, |d| d.extra = value)
    }
    fn to_json<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.read(py, |d| d.to_json())?)?)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Self> {
        Ok(SampleDocument::owned(engine::SampleDocument::from_json(&py_to_json(doc)?)?))
    }
    #[pyo3(signature = (*, indent=None))]
    fn dumps(&self, py: Python<'_>, indent: Option<usize>) -> R<String> {
        self.read(py, |d| match indent {
            Some(n) => d.dumps_indented(n),
            None => d.dumps(),
        })
    }
    /// Parse a document from its JSON text, or from that text's UTF-8 bytes.
    ///
    /// Bytes that are not UTF-8 are refused (§2.4), not decoded with
    /// replacement characters standing in for what was there.
    #[classmethod]
    fn loads(_cls: &Bound<'_, pyo3::types::PyType>, payload: &Bound<'_, PyAny>) -> R<Self> {
        let text: String = match payload.extract::<String>() {
            Ok(s) => s,
            Err(_) => {
                let bytes: Vec<u8> = payload.extract()?;
                String::from_utf8(bytes)
                    .map_err(|e| medh5::Error::Schema(format!("`meta` is not valid UTF-8: {}", e.utf8_error())))?
            }
        };
        Ok(SampleDocument::owned(engine::SampleDocument::loads(&text)?))
    }
    fn check_schema(&self, py: Python<'_>) -> R<Vec<String>> {
        self.read(py, |d| d.check_schema())
    }
    #[getter]
    fn subject_id(&self, py: Python<'_>) -> R<String> {
        self.read(py, |d| d.subject_id().to_string())
    }
    #[getter]
    fn group_id(&self, py: Python<'_>) -> R<String> {
        self.read(py, |d| d.group_id().to_string())
    }
    fn quality_of(slf: &Bound<'_, Self>, key: Option<&str>) -> R<Option<QualityRecord>> {
        let found = slf.get().read(slf.py(), |d| d.quality_of(key).cloned())?;
        Ok(found.map(|record| QualityRecord::read_from(record, slf.as_any(), "quality", key)))
    }
    fn summary<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.read(py, |d| d.summary())?)?)
    }
    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> R<bool> {
        let Ok(other) = other.cast::<SampleDocument>() else { return Ok(false) };
        let theirs = other.get().current(py)?;
        self.read(py, |d| *d == theirs)
    }
    fn __repr__(&self, py: Python<'_>) -> R<String> {
        self.read(py, |d| {
            format!(
                "SampleDocument(sample_id={}, subject_id={}, timepoints={})",
                medh5::json::repr_str(&d.identity.sample_id),
                medh5::json::repr_str(&d.identity.subject_id),
                medh5::json::repr_list(&d.timepoints.ids())
            )
        })
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        crate::values::reduce_via_json(slf.as_any())
    }

    // -- what the live views of the document's members call -------------------

    /// The members of `field` as `(key, value, token)`: the token is the
    /// member's canonical JSON, which `_replace_member` compares.
    fn _members<'py>(&self, py: Python<'py>, field: &str) -> R<Bound<'py, PyList>> {
        let members = Members::of(field)?;
        self.read(py, |d| members.held(d))?.list(py)
    }

    /// Set (or replace) one member, converted as `SampleWriter.extra`
    /// converts a value; the members as the edit left them, read under the
    /// same guard (as `_members` reads them), so a view brought up to date
    /// after an edit kept is never refused.
    fn _set_member<'py>(
        &self,
        py: Python<'py>,
        field: &str,
        key: &str,
        value: &Bound<'_, PyAny>,
    ) -> R<Bound<'py, PyList>> {
        let members = Members::of(field)?;
        let held = match members {
            Members::Quality => {
                let record = QualityRecord::arg(value)?;
                self.edit(py, |d| {
                    d.quality.insert(key.to_string(), record);
                    members.held(d)
                })?
            }
            _ => {
                let value = py_to_json(value)?;
                self.edit(py, |d| {
                    members.json_mut(d).insert(key.to_string(), value);
                    members.held(d)
                })?
            }
        };
        held.list(py)
    }

    /// Replace one member with `value`, if it is still the one read as
    /// `token`; the new token.  A member is a key of `acquisition` or
    /// `extra`, or a record: `identity`, `cohort` or `deidentification`, a
    /// quality record by key, a timepoint or an activity by id, a label
    /// set's class by key.
    #[pyo3(signature = (field, key, token, value))]
    fn _replace_member(
        &self,
        py: Python<'_>,
        field: &str,
        key: Option<&str>,
        token: &str,
        value: &Bound<'_, PyAny>,
    ) -> R<String> {
        let dump = |v: Value| medh5::json::canonical(&v);
        let replaced = match (field, key) {
            ("identity", None) => {
                let (new, json) = Identity::current_of(value)?;
                self.edit(py, |d| swap(&mut d.identity, |v| v.to_json(), token, new))?.then(|| dump(json))
            }
            ("cohort", None) => {
                let (new, json) = Cohort::current_of(value)?;
                self.edit(py, |d| swap(&mut d.cohort, |v| v.to_json(), token, new))?.then(|| dump(json))
            }
            ("deidentification", None) => {
                let (new, json) = Deidentification::current_of(value)?;
                let read = |v: &Option<medh5::curation::identity::Deidentification>| {
                    v.as_ref().map(|r| r.to_json()).unwrap_or(Value::Null)
                };
                self.edit(py, |d| swap(&mut d.deidentification, read, token, Some(new)))?.then(|| dump(json))
            }
            ("quality", Some(k)) => {
                let (new, json) = QualityRecord::current_of(value)?;
                self.edit(py, |d| match d.quality.get_mut(k) {
                    Some(slot) => swap(slot, |v| v.to_json(), token, new),
                    None => false,
                })?
                .then(|| dump(json))
            }
            ("timepoints", Some(id)) => {
                let (new, json) = Timepoint::current_of(value)?;
                self.edit(py, |d| -> medh5::Result<bool> {
                    let mut points = d.timepoints.points().to_vec();
                    let Some(slot) = points.iter_mut().find(|t| t.id == id) else { return Ok(false) };
                    if !swap(slot, |v| v.to_json(), token, new) {
                        return Ok(false);
                    }
                    d.timepoints = medh5::curation::timeline::Timeline::new(points)?;
                    Ok(true)
                })??
                .then(|| dump(json))
            }
            ("provenance.activities", Some(id)) => {
                let (new, json) = Activity::current_of(value)?;
                self.edit(py, |d| -> medh5::Result<bool> {
                    match d.provenance.activity(id) {
                        Ok(found) if medh5::json::canonical(&found.to_json()) == token => {
                            d.provenance.add_activity(new, true)?;
                            Ok(true)
                        }
                        _ => Ok(false),
                    }
                })??
                .then(|| dump(json))
            }
            ("label_set", Some(class_key)) => {
                let (new, json) = LabelClass::current_of(value)?;
                self.edit(py, |d| -> medh5::Result<bool> {
                    let Some(set) = &d.label_set else { return Ok(false) };
                    match set.by_key(class_key) {
                        Some(found) if found.id == new.id && medh5::json::canonical(&found.to_json()) == token => {}
                        _ => return Ok(false),
                    }
                    let classes = set
                        .classes()
                        .iter()
                        .map(|c| if c.key == class_key { new.clone() } else { c.clone() })
                        .collect();
                    d.label_set = Some(with_classes(set, classes)?);
                    Ok(true)
                })??
                .then(|| dump(json))
            }
            (_, Some(k)) => {
                let members = Members::of(field)?;
                if matches!(members, Members::Quality) {
                    return Err(PyKeyError::new_err(field.to_string()).into());
                }
                let value = py_to_json(value)?;
                let written = medh5::json::canonical(&value);
                self.edit(py, |d| {
                    let found = members.json_mut(d);
                    match found.get(k) {
                        Some(current) if medh5::json::canonical(current) == token => {
                            found.insert(k.to_string(), value);
                            true
                        }
                        _ => false,
                    }
                })?
                .then_some(written)
            }
            (_, None) => return Err(PyKeyError::new_err(field.to_string()).into()),
        };
        replaced.ok_or_else(|| changed(field, key).into())
    }

    /// Delete one member (`KeyError` when there is none); the members left,
    /// read under the same guard.
    fn _delete_member<'py>(&self, py: Python<'py>, field: &str, key: &str) -> R<Bound<'py, PyList>> {
        let members = Members::of(field)?;
        let held = self.edit(py, |d| {
            let removed = match members {
                Members::Quality => d.quality.shift_remove(key).is_some(),
                _ => members.json_mut(d).shift_remove(key).is_some(),
            };
            removed.then(|| members.held(d))
        })?;
        match held {
            Some(held) => held.list(py),
            None => Err(PyKeyError::new_err(key.to_string()).into()),
        }
    }

    /// Delete every member; the members left, none.
    fn _clear_members<'py>(&self, py: Python<'py>, field: &str) -> R<Bound<'py, PyList>> {
        let members = Members::of(field)?;
        self.edit(py, |d| match members {
            Members::Quality => d.quality.clear(),
            _ => members.json_mut(d).clear(),
        })?;
        Ok(PyList::empty(py))
    }

    /// A value as a member holds it: converted to JSON and back, detached
    /// from the object given.
    #[staticmethod]
    fn _plain<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &py_to_json(value)?)
    }

    /// A value as member `field` holds it: a quality record made anew from
    /// the one or the mapping given, any other member as [`Self::_plain`]
    /// makes it.  What is made converts with no caller's code.
    #[staticmethod]
    fn _plain_member<'py>(py: Python<'py>, field: &str, value: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
        match Members::of(field)? {
            Members::Quality => Ok(Bound::new(py, QualityRecord::wrap(QualityRecord::arg(value)?))?.into_any()),
            _ => Ok(Self::_plain(py, value)?),
        }
    }
}

pub fn provenance_arg(obj: &Bound<'_, PyAny>) -> R<medh5::curation::provenance::Provenance> {
    if let Ok(p) = obj.cast::<crate::curation::Provenance>() {
        return p.get().current(obj.py());
    }
    Ok(medh5::curation::provenance::Provenance::from_json(Some(&py_to_json(obj)?))?)
}

fn quality_arg(obj: &Bound<'_, PyAny>) -> R<indexmap::IndexMap<String, medh5::curation::quality::QualityRecord>> {
    let mut out = indexmap::IndexMap::new();
    for item in obj.call_method0("items")?.try_iter()? {
        let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
        out.insert(k, QualityRecord::arg(&v)?);
    }
    Ok(out)
}

#[pyfunction]
fn schema<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, engine::schema())
}

#[pyfunction]
fn schema_text() -> &'static str {
    engine::schema_text()
}

#[pyfunction]
fn validate_against_schema(doc: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    Ok(engine::validate_against_schema(&py_to_json(doc)?))
}

/// Start a `SampleDocument` with its two required members, `identity` and `timepoints` (§2.4).
///
/// `subject_id` defaults to `sample_id`. `timepoints` is a `Timeline` or a list
/// of timepoint ids, indexed in order; it defaults to a single `tp0`. Further
/// keyword arguments are `identity` members, such as `sex` or `bodypart`
/// (§12.1).
#[pyfunction]
#[pyo3(signature = (sample_id, subject_id=None, *, timepoints=None, **identity_fields))]
fn new_document(
    sample_id: &str,
    subject_id: Option<&str>,
    timepoints: Option<&Bound<'_, PyAny>>,
    identity_fields: Option<&Bound<'_, PyDict>>,
) -> R<SampleDocument> {
    let mut doc = engine::new_document(sample_id, subject_id, None)?;
    if let Some(t) = timepoints.filter(|t| !t.is_none()) {
        doc.timepoints = match t.cast::<Timeline>() {
            Ok(tl) => tl.get().current(t.py())?,
            Err(_) => {
                let ids = crate::convert::strings(t)?;
                engine::new_document(sample_id, subject_id, Some(&ids))?.timepoints
            }
        };
    }
    if let Some(fields) = identity_fields {
        let mut identity = object(doc.identity.to_json());
        identity.extend(crate::convert::kwargs_to_map(Some(fields))?);
        doc.identity = medh5::curation::identity::Identity::from_json(&Value::Object(identity))?;
    }
    Ok(SampleDocument::owned(doc))
}

/// The raw `/meta` text under a sample root (`sample.root`).
#[pyfunction]
fn read_document_text(py: Python<'_>, root: &Bound<'_, PyAny>) -> R<String> {
    let group = crate::integrity::group_of(root)?;
    Ok(py.detach(move || medh5::sample::read_document_text(&group))?)
}

/// The sample document under a sample root, parsed: a later minor's as its
/// supported projection (1.1 §2.2).
#[pyfunction]
fn read_document(py: Python<'_>, root: &Bound<'_, PyAny>) -> R<SampleDocument> {
    let group = crate::integrity::group_of(root)?;
    Ok(SampleDocument::owned(py.detach(move || medh5::sample::read_document(&group))?))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SampleDocument>()?;
    m.add_function(wrap_pyfunction!(read_document_text, m)?)?;
    m.add_function(wrap_pyfunction!(read_document, m)?)?;
    m.add_function(wrap_pyfunction!(schema, m)?)?;
    m.add_function(wrap_pyfunction!(schema_text, m)?)?;
    m.add_function(wrap_pyfunction!(validate_against_schema, m)?)?;
    m.add_function(wrap_pyfunction!(new_document, m)?)?;
    m.add("META_DATASET", engine::META_DATASET)?;
    Ok(())
}
