//! The clinical profile (format 1.1) and the task-and-cache contract.
//!
//! Records cross this boundary as JSON-shaped dicts --- the logical-record
//! form of `medh5-clinical-1.schema.json` --- and `medh5/clinical.py` turns
//! them into frozen dataclasses.  The companion functions are stateless: a
//! task manifest is a dict, re-parsed per call, so it pickles into DataLoader
//! workers with nothing to rebuild.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use serde_json::{json, Value};

use medh5::clinical::model::{ClinicalRecords, Event, Link};
use medh5::clinical::select::SelectionPolicy;
use medh5::clinical::Clinical as EngineClinical;
use medh5::companion::cache::{CacheEntry, CacheHeader, CacheWriter, FeatureCache};
use medh5::companion::task::TaskManifest;
use medh5::companion::{CohortTask, Partitions, SourceAccess, SourceRef};
use medh5::sample::Sample as EngineSample;

use crate::convert::{json_to_py, nd_to_py, py_to_json, py_to_nd};
use crate::errors::R;
use crate::exclusive::Exclusive;

/// A Python value as JSON: a dict, or anything with `to_json()`.
pub fn record(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    if obj.hasattr("to_json")? && !obj.is_instance_of::<PyDict>() {
        return py_to_json(&obj.call_method0("to_json")?);
    }
    py_to_json(obj)
}

/// An exact instant given as one integer reads as `[t, t]`.
fn widen_instants(mut value: Value) -> Value {
    if let Value::Object(map) = &mut value {
        for key in ["effective_start_us", "effective_end_us", "available_us"] {
            if let Some(Value::Number(n)) = map.get(key) {
                if let Some(t) = n.as_i64() {
                    map.insert(key.into(), json!([t, t]));
                }
            }
        }
    }
    value
}

/// A record from a positional value and keyword fields (the fields win).
pub fn record_with(value: Option<&Bound<'_, PyAny>>, fields: Option<&Bound<'_, PyDict>>) -> PyResult<Value> {
    let mut out = match value {
        Some(v) if !v.is_none() => record(v)?,
        _ => json!({}),
    };
    if let (Value::Object(map), Some(f)) = (&mut out, fields) {
        if let Value::Object(extra) = py_to_json(f.as_any())? {
            map.extend(extra);
        }
    }
    Ok(widen_instants(out))
}

pub fn event_arg(value: Option<&Bound<'_, PyAny>>, fields: Option<&Bound<'_, PyDict>>) -> R<Event> {
    Ok(Event::from_json(&record_with(value, fields)?)?)
}

pub fn policy_arg(policy: Option<&Bound<'_, PyAny>>) -> R<SelectionPolicy> {
    match policy {
        None => Ok(SelectionPolicy::strict()),
        Some(p) if p.is_none() => Ok(SelectionPolicy::strict()),
        Some(p) if p.is_instance_of::<pyo3::types::PyString>() => {
            let mut out = SelectionPolicy::strict();
            out.selection = p.extract()?;
            out.check()?;
            Ok(out)
        }
        Some(p) => Ok(SelectionPolicy::from_json(&record(p)?)?),
    }
}

fn to_list<'py>(py: Python<'py>, values: impl IntoIterator<Item = Value>) -> PyResult<Bound<'py, PyList>> {
    let items: Vec<Bound<'py, PyAny>> = values.into_iter().map(|v| json_to_py(py, &v)).collect::<PyResult<_>>()?;
    PyList::new(py, items)
}

/// One sample's clinical profile, read.
#[pyclass(module = "medh5._core", name = "ClinicalHandle", frozen)]
pub struct ClinicalHandle {
    pub inner: Arc<EngineClinical>,
    /// The sample it was read from, kept open while this is alive.
    pub _sample: Arc<EngineSample>,
}

#[pymethods]
impl ClinicalHandle {
    #[getter]
    fn projection(&self) -> bool {
        self.inner.projection
    }
    fn descriptor<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.descriptor.to_json())
    }
    fn events<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_list(py, self.inner.events.iter().map(Event::to_json))
    }
    fn links<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_list(py, self.inner.links.iter().map(Link::to_json))
    }
    /// Document metadata (no text): id, media type, language, source type,
    /// and the text's length in UTF-8 bytes.
    fn documents<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
        to_list(
            py,
            self.inner.documents().iter().map(|d| {
                json!({
                    "document_id": d.document_id,
                    "media_type": d.media_type,
                    "language": d.language,
                    "source_type": d.source_type,
                    "n_bytes": d.n_bytes,
                })
            }),
        )
    }
    /// One document's text, read from the file now.
    fn text(&self, py: Python<'_>, document_id: &str) -> R<String> {
        let inner = self.inner.clone();
        let id = document_id.to_string();
        Ok(py.detach(move || inner.text(&id))?)
    }
    fn document<'py>(&self, py: Python<'py>, document_id: &str) -> R<Bound<'py, PyAny>> {
        let doc = crate::h5(py, || self.inner.document(document_id).map(|d| d.to_json()))?;
        Ok(json_to_py(py, &doc)?)
    }
    /// Every record, each document's text read from the file now.
    fn records<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        let inner = self.inner.clone();
        let doc = py.detach(move || inner.records().map(|r| r.to_json()))?;
        Ok(json_to_py(py, &doc)?)
    }
    #[pyo3(signature = (cutoff_us, policy=None))]
    fn select<'py>(&self, py: Python<'py>, cutoff_us: i64, policy: Option<&Bound<'py, PyAny>>) -> R<Bound<'py, PyAny>> {
        let policy = policy_arg(policy)?;
        let doc = crate::h5(py, || {
            self.inner.select(cutoff_us, &policy).map(|selection| selection.to_json(&self.inner.events))
        })?;
        Ok(json_to_py(py, &doc)?)
    }
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.inner.summary())
    }
}

/// Select from records not read from a file: `events` and `links` as dicts.
#[pyfunction]
#[pyo3(signature = (events, links, cutoff_us, policy=None))]
fn clinical_select<'py>(
    py: Python<'py>,
    events: Vec<Bound<'py, PyAny>>,
    links: Vec<Bound<'py, PyAny>>,
    cutoff_us: i64,
    policy: Option<&Bound<'py, PyAny>>,
) -> R<Bound<'py, PyAny>> {
    let events: Vec<Event> = events.iter().map(|e| event_arg(Some(e), None)).collect::<R<_>>()?;
    let links: Vec<Link> = links.iter().map(|l| Ok(Link::from_json(&record(l)?)?)).collect::<R<_>>()?;
    let refs: Vec<(usize, &Link)> = links.iter().map(|l| (0, l)).collect();
    let selection = medh5::clinical::select(&events, &refs, cutoff_us, &policy_arg(policy)?)?;
    Ok(json_to_py(py, &selection.to_json(&events))?)
}

/// Check a logical-record bundle against its schema and parse it back.
#[pyfunction]
fn clinical_records<'py>(py: Python<'py>, records: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    let parsed = ClinicalRecords::from_json(&record(records)?)?;
    Ok(json_to_py(py, &parsed.to_json())?)
}

#[pyfunction]
#[pyo3(signature = (path, records, out=None))]
fn clinical_augment<'py>(
    py: Python<'py>,
    path: PathBuf,
    records: &Bound<'py, PyAny>,
    out: Option<PathBuf>,
) -> R<Bound<'py, PyAny>> {
    let records = ClinicalRecords::from_json(&record(records)?)?;
    let report = py.detach(move || medh5::clinical::augment::augment(&path, records, out.as_deref()))?;
    Ok(json_to_py(py, &report.to_json())?)
}

#[pyfunction]
fn clinical_strip<'py>(py: Python<'py>, path: PathBuf, out: PathBuf) -> R<Bound<'py, PyAny>> {
    let report = py.detach(move || medh5::clinical::augment::strip(&path, &out))?;
    Ok(json_to_py(py, &report.to_json())?)
}

/// `(events, links, notes)`: imaging events from `days_from_baseline`.
#[pyfunction]
fn imaging_events_from_timepoints<'py>(py: Python<'py>, path: PathBuf) -> R<Bound<'py, PyAny>> {
    let (events, links, notes) = py.detach(move || medh5::clinical::augment::imaging_events_from_timepoints(&path))?;
    let value = json!([
        events.iter().map(Event::to_json).collect::<Vec<_>>(),
        links.iter().map(Link::to_json).collect::<Vec<_>>(),
        notes,
    ]);
    Ok(json_to_py(py, &value)?)
}

/// The clock `imaging_events_from_timepoints` measures on.
#[pyfunction]
fn baseline_day_clock<'py>(py: Python<'py>, clock_id: &str) -> PyResult<Bound<'py, PyAny>> {
    json_to_py(py, &medh5::clinical::augment::baseline_day_clock(clock_id).to_json())
}

/// `value` units of `unit_us` microseconds, rounded half to even: the
/// engine's rounding, which `medh5.clinical.hours` and `days` are held to.
#[pyfunction]
fn clinical_round_micros(value: f64, unit_us: i64) -> R<i64> {
    Ok(medh5::clinical::round_micros(value, unit_us)?)
}

/// The clock day holding `t`, as inclusive `(lo, hi)` bounds.
#[pyfunction]
fn clinical_day_containing(t: i64) -> (i64, i64) {
    let day = medh5::clinical::model::Bounds::day_containing(t);
    (day.lo, day.hi)
}

#[pyfunction]
fn clinical_schema_text() -> &'static str {
    medh5::clinical::schema::schema_text()
}

// -- the task-and-cache contract -----------------------------------------------------------------

/// A manifest is judged as the same JSON in a file would be: an integer past
/// 64 bits is the contract's T101, not a conversion error.
fn manifest(doc: &Bound<'_, PyAny>) -> R<TaskManifest> {
    let doc = if doc.hasattr("to_json")? && !doc.is_instance_of::<PyDict>() {
        doc.call_method0("to_json")?
    } else {
        doc.clone()
    };
    Ok(TaskManifest::from_json(&crate::convert::py_to_json_as_parsed(&doc)?)?)
}

/// Where sources may be opened: confined to `root` when one is given.
fn access(base: Option<PathBuf>, root: Option<PathBuf>) -> medh5::Result<SourceAccess> {
    match root {
        Some(root) => SourceAccess::confined(base.as_deref(), Some(&root)),
        None => Ok(SourceAccess::new(base.as_deref())),
    }
}

fn findings<'py>(py: Python<'py>, found: &[medh5::companion::Finding]) -> PyResult<Bound<'py, PyList>> {
    to_list(py, found.iter().map(medh5::companion::Finding::to_json))
}

/// Parse, check the schema, and return the normalised manifest.
#[pyfunction]
fn task_normalize<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &manifest(doc)?.to_json())?)
}

/// What the manifest document should state and does not, as written: the
/// normal form has filled the defaults in.
#[pyfunction]
fn task_notes<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyList>> {
    let doc = crate::convert::py_to_json_as_parsed(doc)?;
    Ok(to_list(py, medh5::companion::task::notes_of(&doc).iter().map(medh5::companion::Note::to_json))?)
}

/// Everything wrong with a manifest that opening no file can find.
#[pyfunction]
fn task_validate<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyList>> {
    Ok(findings(py, &manifest(doc)?.validate())?)
}

/// `{"task": ..., "manifest": ...}`.
#[pyfunction]
fn task_fingerprints<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    let m = manifest(doc)?;
    Ok(json_to_py(py, &json!({"task": m.task_fingerprint(), "manifest": m.manifest_fingerprint()}))?)
}

#[pyfunction]
fn task_row_fingerprint(doc: &Bound<'_, PyAny>, row_id: &str) -> R<String> {
    let m = manifest(doc)?;
    let row =
        m.rows.iter().find(|r| r.row_id == row_id).ok_or_else(|| medh5::Error::Key(format!("no row {row_id:?}")))?;
    Ok(m.row_fingerprint(row))
}

/// Every row's fingerprint by row id --- the first row of an id, as
/// `task_row_fingerprint` finds it --- from one parse of the manifest: a
/// parse per row made fingerprinting a task's rows quadratic.
#[pyfunction]
fn task_row_fingerprints<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyDict>> {
    let m = manifest(doc)?;
    let task = m.task_fingerprint();
    let mut subjects: std::collections::HashMap<&str, &medh5::companion::task::Subject> =
        std::collections::HashMap::new();
    for s in &m.subjects {
        subjects.entry(s.subject_id.as_str()).or_insert(s);
    }
    let out = PyDict::new(py);
    for row in &m.rows {
        if !out.contains(&row.row_id)? {
            let subject = subjects.get(row.subject_id.as_str()).copied();
            out.set_item(&row.row_id, m.row_fingerprint_with(&task, subject, row))?;
        }
    }
    Ok(out)
}

#[pyfunction]
fn task_subjects_digest(doc: &Bound<'_, PyAny>, partition: &str) -> R<String> {
    Ok(manifest(doc)?.subjects_digest(partition))
}

/// The manifest with every subject's duplicated events recorded.
#[pyfunction]
#[pyo3(signature = (doc, base=None, root=None))]
fn task_reconcile<'py>(
    py: Python<'py>,
    doc: &Bound<'py, PyAny>,
    base: Option<PathBuf>,
    root: Option<PathBuf>,
) -> R<Bound<'py, PyAny>> {
    let mut m = manifest(doc)?;
    let m = py.detach(move || -> medh5::Result<TaskManifest> {
        m.reconcile_all(&access(base, root)?)?;
        Ok(m)
    })?;
    Ok(json_to_py(py, &m.to_json())?)
}

/// The manifest with its sources repinned, and what changed:
/// `{"manifest": ..., "report": ...}`.
#[pyfunction]
#[pyo3(signature = (doc, base=None, root=None, sources=None))]
fn task_repin<'py>(
    py: Python<'py>,
    doc: &Bound<'py, PyAny>,
    base: Option<PathBuf>,
    root: Option<PathBuf>,
    sources: Option<Vec<String>>,
) -> R<Bound<'py, PyAny>> {
    let mut m = manifest(doc)?;
    let (m, report) = py.detach(move || -> medh5::Result<_> {
        let only: Option<std::collections::BTreeSet<String>> = sources.map(|s| s.into_iter().collect());
        let report = m.repin(&access(base, root)?, only.as_ref())?;
        Ok((m, report))
    })?;
    Ok(json_to_py(py, &json!({"manifest": m.to_json(), "report": report.to_json()}))?)
}

/// A task over a cohort (`TaskManifest::from_cohort`), partitioned by
/// `split` (a split's JSON) or by the claims the samples carry for `set_id`:
/// the manifest's JSON.
#[pyfunction]
#[pyo3(signature = (
    cohort, task_id, version, identity_namespace, *, split=None, set_id=None, fold=None, base=None,
    training_partition=None
))]
#[allow(clippy::too_many_arguments)]
fn task_from_cohort<'py>(
    py: Python<'py>,
    cohort: &Bound<'py, PyAny>,
    task_id: String,
    version: String,
    identity_namespace: String,
    split: Option<&Bound<'py, PyAny>>,
    set_id: Option<String>,
    fold: Option<u32>,
    base: Option<PathBuf>,
    training_partition: Option<String>,
) -> R<Bound<'py, PyAny>> {
    let training_partition = training_partition.unwrap_or_else(|| "train".into());
    let cohort = medh5::dataset::Manifest::from_json(&py_to_json(cohort)?)?;
    let split = match split {
        Some(s) => Some(medh5::dataset::Split::from_json(&py_to_json(s)?)?),
        None => None,
    };
    let task = py.detach(move || -> medh5::Result<TaskManifest> {
        let partitions = match (&split, &set_id) {
            (Some(split), None) => Partitions::Split { split, fold },
            (None, Some(set_id)) if fold.is_none() => Partitions::Claims { set_id },
            (None, Some(_)) => {
                return Err(medh5::Error::invalid("a fold is a split's; claims already name their partition"));
            }
            _ => return Err(medh5::Error::invalid("give a split or the set_id of the claims, one of the two")),
        };
        let named = CohortTask::new(&task_id, &version, &identity_namespace)
            .with_base(base.as_deref())
            .with_training_partition(&training_partition);
        TaskManifest::from_cohort(&cohort, partitions, &named)
    })?;
    Ok(json_to_py(py, &task.to_json())?)
}

/// Write the manifest as the engine writes it, refusing an output that is
/// one of its sources; `base` resolves relative sources (else both the
/// working directory and the output's directory).
#[pyfunction]
#[pyo3(signature = (doc, path, base=None))]
fn task_save(py: Python<'_>, doc: &Bound<'_, PyAny>, path: PathBuf, base: Option<PathBuf>) -> R<()> {
    let m = manifest(doc)?;
    py.detach(move || -> medh5::Result<()> {
        // Without a base, a source resolves against the working directory
        // now and against the output's directory once the manifest is loaded
        // from there: neither may be the output.
        let bases: Vec<Option<&Path>> = match &base {
            Some(base) => vec![Some(base.as_path())],
            None => vec![None, path.parent()],
        };
        for base in bases {
            m.check_output(&path, &SourceAccess::new(base))?;
        }
        m.save(&path)
    })?;
    Ok(())
}

/// A manifest file's document, unparsed: T101 for anything not JSON.
#[pyfunction]
fn task_load_document<'py>(py: Python<'py>, path: PathBuf) -> R<Bound<'py, PyAny>> {
    let doc = py.detach(move || TaskManifest::read_document(&path))?;
    Ok(json_to_py(py, &doc)?)
}

/// The preflight of a task, as columns (`crate::preflight`).
#[pyfunction]
#[pyo3(signature = (doc, base=None, deep=false, root=None))]
fn task_preflight<'py>(
    py: Python<'py>,
    doc: &Bound<'py, PyAny>,
    base: Option<PathBuf>,
    deep: bool,
    root: Option<PathBuf>,
) -> R<Bound<'py, PyAny>> {
    let m = manifest(doc)?;
    let access = access(base, root)?;
    Ok(crate::preflight::preflight_to_py(py, m, access, deep)?.into_any())
}

/// The URIs of the sources that confining to `base` would refuse, in
/// manifest order; no file is examined.  Empty when `base` cannot be
/// canonicalized: its sources fail to open whatever confinement does.
#[pyfunction]
#[pyo3(signature = (doc, base=None))]
fn task_sources_outside(doc: &Bound<'_, PyAny>, base: Option<PathBuf>) -> R<Vec<String>> {
    let m = manifest(doc)?;
    let Ok(confined) = SourceAccess::confined(base.as_deref(), None) else { return Ok(Vec::new()) };
    Ok(m.sources_outside(&confined).into_iter().map(|s| s.uri.clone()).collect())
}

/// A source reference pinned to what the sample is now.
#[pyfunction]
#[pyo3(signature = (path, sample_key=None, source_id=None, uri=None))]
fn source_pin<'py>(
    py: Python<'py>,
    path: PathBuf,
    sample_key: Option<String>,
    source_id: Option<String>,
    uri: Option<String>,
) -> R<Bound<'py, PyAny>> {
    let locator = uri.unwrap_or_else(|| path.to_string_lossy().into_owned());
    let found = py.detach(move || -> medh5::Result<SourceRef> {
        let mut src = SourceRef::new(path.to_string_lossy(), sample_key.clone(), "");
        let sample = src.open(None)?;
        src = SourceRef::pin(locator, sample_key, &sample)?;
        src.source_id = source_id.unwrap_or_default();
        Ok(src)
    })?;
    Ok(json_to_py(py, &found.to_json())?)
}

/// The file a source URI names, relative to `base` when it is relative: a
/// `file:` URI decoded as the engine decodes it.
#[pyfunction]
#[pyo3(signature = (uri, base=None, root=None))]
fn source_resolve(uri: &str, base: Option<PathBuf>, root: Option<PathBuf>) -> R<PathBuf> {
    let source = SourceRef::new(uri, None, "");
    Ok(match root {
        None => source.resolve(base.as_deref()),
        Some(_) => access(base, root)?.locate(&source)?,
    })
}

/// The id a source reference gets when none is given, widened past the ids
/// in `taken` (`SourceRef::default_id_among`).
#[pyfunction]
#[pyo3(signature = (uri, sample_key=None, taken=Vec::new()))]
fn source_default_id(uri: &str, sample_key: Option<&str>, taken: Vec<String>) -> String {
    let taken: std::collections::HashSet<String> = taken.into_iter().collect();
    SourceRef::default_id_among(uri, sample_key, &|id| taken.contains(id))
}

/// The findings of checking a reference against its sample now.
#[pyfunction]
#[pyo3(signature = (source, base=None, deep=false, root=None))]
fn source_check<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    base: Option<PathBuf>,
    deep: bool,
    root: Option<PathBuf>,
) -> R<Bound<'py, PyList>> {
    let src = SourceRef::from_json(&record(source)?)?;
    let found = py.detach(move || -> medh5::Result<Vec<medh5::companion::Finding>> {
        match src.open_in(&access(base, root)?) {
            Err(e) => Ok(vec![medh5::companion::Finding::new("T301", src.locator(), e.to_string())]),
            Ok(sample) => src.check(&sample, deep),
        }
    })?;
    Ok(findings(py, &found)?)
}

/// Each event version of a source and the documents it owns: the engine's
/// rule for what an event-level cache files under it.
#[pyfunction]
#[pyo3(signature = (source, base=None, root=None))]
fn cache_owned_documents<'py>(
    py: Python<'py>,
    source: &Bound<'py, PyAny>,
    base: Option<PathBuf>,
    root: Option<PathBuf>,
) -> R<Bound<'py, PyDict>> {
    let src = SourceRef::from_json(&record(source)?)?;
    let owned = py.detach(move || medh5::companion::cache::owned_documents(&src, &access(base, root)?))?;
    let out = PyDict::new(py);
    for (event, documents) in owned {
        out.set_item(event, documents)?;
    }
    Ok(out)
}

/// An entry as the engine reads it, written back: a malformed entry is
/// refused (T401) rather than read leniently.
#[pyfunction]
fn cache_entry_json<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &CacheEntry::from_json(&py_to_json(doc)?)?.to_json())?)
}

/// A `CacheReport` dataclass's engine JSON: its verdict, stale and corrupt
/// entries are the engine's.
#[pyfunction]
fn cache_report_json<'py>(py: Python<'py>, report: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    let mut findings = Vec::new();
    for f in report.getattr("findings")?.try_iter()? {
        let f = f?;
        findings.push(medh5::companion::Finding::new(
            &f.getattr("code")?.extract::<String>()?,
            f.getattr("location")?.extract::<String>()?,
            f.getattr("message")?.extract::<String>()?,
        ));
    }
    let engine = medh5::companion::CacheReport {
        path: report.getattr("path")?.extract()?,
        entries: report.getattr("entries")?.extract()?,
        findings,
        manifest_digest: report.getattr("manifest_digest")?.extract()?,
        level: report.getattr("level")?.extract()?,
    };
    Ok(json_to_py(py, &engine.to_json())?)
}

fn header(value: &Value) -> CacheHeader {
    let opt = |k: &str| value.get(k).and_then(Value::as_str).map(str::to_string);
    CacheHeader {
        level: opt("level").unwrap_or_else(|| "event".into()),
        encoder: value.get("encoder").cloned().unwrap_or(Value::Null),
        preprocessing: value.get("preprocessing").cloned().unwrap_or_else(|| json!({})),
        output: value.get("output").cloned().unwrap_or(Value::Null),
        task_fingerprint: opt("task_fingerprint"),
        selection: opt("selection"),
        fitted_on: value.get("fitted_on").filter(|v| !v.is_null()).cloned(),
    }
}

/// Writes a feature cache; `commit()` moves it into place.  Used by one
/// thread at a time (E4): a call while another runs is refused.
#[pyclass(module = "medh5._core", name = "CacheWriterHandle", frozen)]
pub struct CacheWriterHandle {
    inner: Exclusive<Option<CacheWriter>>,
}

#[pymethods]
impl CacheWriterHandle {
    /// Add one feature: `entry` as a dict (its `digest` is computed).
    fn add<'py>(&self, py: Python<'py>, entry: &Bound<'py, PyAny>, values: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
        // As the reader parses one: a cutoff that is not an integer was read
        // as none, and refused as a row entry pinning no cutoff.
        let entry = CacheEntry { digest: String::new(), ..CacheEntry::from_json(&record(entry)?)? };
        let values = py_to_nd(values)?;
        let mut held = self.inner.write()?;
        let writer = held.as_mut().ok_or_else(|| medh5::Error::invalid("this cache writer has finished"))?;
        // The digest and the write, of values already out of Python.
        let added = py.detach(move || writer.add(entry, &values).map(|e| e.to_json()))?;
        drop(held);
        Ok(json_to_py(py, &added)?)
    }
    /// Write the manifest and its checksum; returns the checksum.
    fn commit(&self, py: Python<'_>) -> R<String> {
        // Held across the write, as `add`'s is: a call meanwhile is refused,
        // not told the writer has finished before it has.
        let mut held = self.inner.write()?;
        let writer = held.take().ok_or_else(|| medh5::Error::invalid("this cache writer has finished"))?;
        let checksum = py.detach(move || writer.commit());
        drop(held);
        Ok(checksum?)
    }
    /// Discard the half-written cache.
    fn abort(&self, py: Python<'_>) -> R<()> {
        let taken = self.inner.write()?.take();
        crate::h5(py, move || drop(taken));
        Ok(())
    }
}

/// Start writing a feature cache (`medh5.cache/1`).
#[pyfunction]
fn cache_create(py: Python<'_>, path: PathBuf, header_doc: &Bound<'_, PyAny>) -> R<CacheWriterHandle> {
    let h = header(&record(header_doc)?);
    let writer = crate::h5(py, move || CacheWriter::create(&path, h))?;
    Ok(CacheWriterHandle { inner: Exclusive::new("a CacheWriterHandle", Some(writer)) })
}

/// An open feature cache; `close()` releases the file.
#[pyclass(module = "medh5._core", name = "FeatureCacheHandle", frozen)]
pub struct FeatureCacheHandle {
    inner: std::sync::Mutex<Option<Arc<FeatureCache>>>,
}

impl FeatureCacheHandle {
    fn cache(&self) -> R<Arc<FeatureCache>> {
        let guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        Ok(guard.clone().ok_or_else(|| medh5::Error::File("the feature cache is closed".into()))?)
    }
}

#[pymethods]
impl FeatureCacheHandle {
    #[getter]
    fn path(&self) -> R<String> {
        Ok(self.cache()?.path.to_string_lossy().into_owned())
    }
    #[getter]
    fn manifest_digest(&self) -> R<String> {
        Ok(self.cache()?.manifest_digest.clone())
    }
    #[getter]
    fn is_open(&self) -> bool {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }
    fn close(&self, py: Python<'_>) {
        let taken = self.inner.lock().unwrap_or_else(|e| e.into_inner()).take();
        crate::h5(py, move || drop(taken));
    }
    /// Forget the handle without closing it: what a forked child does with
    /// its parent's (§14.4) --- the descriptor is the parent's to close.
    fn abandon(&self) {
        if let Some(cache) = self.inner.lock().unwrap_or_else(|e| e.into_inner()).take() {
            std::mem::forget(cache);
        }
    }
    fn header<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        let cache = self.cache()?;
        let h = &cache.header;
        Ok(json_to_py(
            py,
            &json!({
                "level": h.level, "encoder": h.encoder, "preprocessing": h.preprocessing, "output": h.output,
                "task_fingerprint": h.task_fingerprint, "selection": h.selection, "fitted_on": h.fitted_on,
            }),
        )?)
    }
    fn entries<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyList>> {
        Ok(to_list(py, self.cache()?.entries.iter().map(CacheEntry::to_json))?)
    }
    /// How many entries the cache holds, without building them.
    fn entry_count(&self) -> R<usize> {
        Ok(self.cache()?.entries.len())
    }
    /// An entry's payload, its checksum verified.
    fn get<'py>(&self, py: Python<'py>, entry_id: &str) -> R<Bound<'py, PyAny>> {
        let cache = self.cache()?;
        let id = entry_id.to_string();
        Ok(nd_to_py(py, py.detach(move || cache.get(&id))?))
    }
    fn event_entry<'py>(&self, py: Python<'py>, content_id: &str, event_id: &str) -> R<Option<Bound<'py, PyAny>>> {
        Ok(self.cache()?.event_entry(content_id, event_id).map(|e| json_to_py(py, &e.to_json())).transpose()?)
    }
    fn row_entry<'py>(&self, py: Python<'py>, row_id: &str) -> R<Option<Bound<'py, PyAny>>> {
        Ok(self.cache()?.row_entry(row_id).map(|e| json_to_py(py, &e.to_json())).transpose()?)
    }
}

#[pyfunction]
fn cache_open(py: Python<'_>, path: PathBuf) -> R<FeatureCacheHandle> {
    let cache = py.detach(move || FeatureCache::open(&path))?;
    Ok(FeatureCacheHandle { inner: std::sync::Mutex::new(Some(Arc::new(cache))) })
}

/// Validate a cache; with a task, also against that task (and, with
/// `check_rows`, against what its preflight admits).
#[pyfunction]
#[pyo3(signature = (path, base=None, task=None, task_base=None, check_rows=true, root=None))]
fn cache_validate<'py>(
    py: Python<'py>,
    path: PathBuf,
    base: Option<PathBuf>,
    task: Option<&Bound<'py, PyAny>>,
    task_base: Option<PathBuf>,
    check_rows: bool,
    root: Option<PathBuf>,
) -> R<Bound<'py, PyAny>> {
    let task = match task {
        Some(t) if !t.is_none() => Some(manifest(t)?),
        _ => None,
    };
    let report = py.detach(move || -> medh5::Result<medh5::companion::CacheReport> {
        let task_access = access(task_base.clone(), root.clone())?;
        let admitted = match &task {
            Some(t) if check_rows => Some(medh5::companion::Admitted::preflight_in(t, &task_access, false)?),
            _ => None,
        };
        // The entries copy the task's references, relative to its base
        // (task-cache-1 §7.1): resolved against the cache's directory, a
        // cache kept anywhere else was all stale.
        let base = base.or_else(|| task.as_ref().and(task_base)).or_else(|| path.parent().map(PathBuf::from));
        let mut check = medh5::companion::CacheCheck::new(access(base, root)?);
        if let Some(t) = &task {
            check = check.with_task(t, &task_access);
        }
        if let Some(a) = &admitted {
            check = check.with_admitted(a);
        }
        medh5::companion::validate_cache_in(&path, &check)
    })?;
    Ok(json_to_py(py, &report.to_json())?)
}

#[pyfunction]
fn cache_fitted_on<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>, partition: &str) -> R<Bound<'py, PyAny>> {
    Ok(json_to_py(py, &CacheHeader::fitted_on(&manifest(doc)?, partition))?)
}

/// Why a `fitted_on` record is not this task's training partition (T405).
#[pyfunction]
fn cache_fitted_on_mismatches(doc: &Bound<'_, PyAny>, fitted: &Bound<'_, PyAny>) -> R<Vec<String>> {
    Ok(CacheHeader::fitted_on_mismatches(&manifest(doc)?, &record(fitted)?))
}

#[pyfunction]
fn cache_event_entry_id(content_id: &str, event_id: &str) -> String {
    medh5::companion::cache::event_entry_id(content_id, event_id)
}

#[pyfunction]
fn task_schema_text() -> &'static str {
    medh5::companion::task::schema_text()
}

#[pyfunction]
fn cache_schema_text() -> &'static str {
    medh5::companion::cache::schema_text()
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ClinicalHandle>()?;
    m.add_class::<CacheWriterHandle>()?;
    m.add_class::<FeatureCacheHandle>()?;
    for f in [
        wrap_pyfunction!(clinical_select, m)?,
        wrap_pyfunction!(clinical_records, m)?,
        wrap_pyfunction!(clinical_augment, m)?,
        wrap_pyfunction!(clinical_strip, m)?,
        wrap_pyfunction!(imaging_events_from_timepoints, m)?,
        wrap_pyfunction!(baseline_day_clock, m)?,
        wrap_pyfunction!(clinical_schema_text, m)?,
        wrap_pyfunction!(clinical_round_micros, m)?,
        wrap_pyfunction!(clinical_day_containing, m)?,
        wrap_pyfunction!(task_normalize, m)?,
        wrap_pyfunction!(task_validate, m)?,
        wrap_pyfunction!(task_notes, m)?,
        wrap_pyfunction!(task_fingerprints, m)?,
        wrap_pyfunction!(task_row_fingerprint, m)?,
        wrap_pyfunction!(task_row_fingerprints, m)?,
        wrap_pyfunction!(task_subjects_digest, m)?,
        wrap_pyfunction!(task_reconcile, m)?,
        wrap_pyfunction!(task_save, m)?,
        wrap_pyfunction!(task_repin, m)?,
        wrap_pyfunction!(task_from_cohort, m)?,
        wrap_pyfunction!(task_load_document, m)?,
        wrap_pyfunction!(task_preflight, m)?,
        wrap_pyfunction!(task_sources_outside, m)?,
        wrap_pyfunction!(source_pin, m)?,
        wrap_pyfunction!(source_check, m)?,
        wrap_pyfunction!(source_resolve, m)?,
        wrap_pyfunction!(source_default_id, m)?,
        wrap_pyfunction!(cache_create, m)?,
        wrap_pyfunction!(cache_open, m)?,
        wrap_pyfunction!(cache_validate, m)?,
        wrap_pyfunction!(cache_owned_documents, m)?,
        wrap_pyfunction!(cache_entry_json, m)?,
        wrap_pyfunction!(cache_report_json, m)?,
        wrap_pyfunction!(cache_fitted_on, m)?,
        wrap_pyfunction!(cache_fitted_on_mismatches, m)?,
        wrap_pyfunction!(cache_event_entry_id, m)?,
        wrap_pyfunction!(task_schema_text, m)?,
        wrap_pyfunction!(cache_schema_text, m)?,
    ] {
        m.add_function(f)?;
    }
    let py = m.py();
    let codes = PyDict::new(py);
    for (code, summary) in medh5::companion::CODES {
        codes.set_item(code, summary)?;
    }
    m.add("COMPANION_CODES", codes)?;
    m.add("CLINICAL_PROFILE", medh5::clinical::PROFILE)?;
    m.add("CLINICAL_SCHEMA", medh5::clinical::SCHEMA)?;
    m.add("TASK_SCHEMA", medh5::companion::task::SCHEMA)?;
    m.add("CACHE_SCHEMA", medh5::companion::cache::SCHEMA)?;
    m.add("CACHE_SUFFIX", medh5::companion::cache::SUFFIX)?;
    m.add("ASSESSMENT_SYSTEM", medh5::clinical::model::ASSESSMENT_SYSTEM)?;
    m.add("LESION_PRESENCE", medh5::clinical::model::LESION_PRESENCE)?;
    for (name, values) in [
        ("EVENT_KINDS", &medh5::clinical::EVENT_KINDS[..]),
        ("TEMPORAL_TYPES", &medh5::clinical::TEMPORAL_TYPES[..]),
        ("STATUSES", &medh5::clinical::STATUSES[..]),
        ("COMPARATORS", &medh5::clinical::COMPARATORS[..]),
        ("ENDPOINT_TYPES", &medh5::clinical::ENDPOINT_TYPES[..]),
        ("RELATIONS", &medh5::clinical::RELATIONS[..]),
        ("CLOCK_REFERENCES", &medh5::clinical::CLOCK_REFERENCES[..]),
        ("LESION_VALUES", &medh5::clinical::LESION_VALUES[..]),
        ("SELECTION_POLICIES", &medh5::clinical::POLICIES[..]),
    ] {
        m.add(name, pyo3::types::PyTuple::new(py, values.iter().copied())?)?;
    }
    Ok(())
}
