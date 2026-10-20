//! `medh5.curation` value types: identity, provenance, quality, timeline
//! (spec §3.7, §11, §12).

use std::sync::OnceLock;

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyFrozenSet, PySlice, PyString, PyTuple};
use serde_json::Value;

use medh5::curation::identity as id_engine;
use medh5::curation::provenance as prov_engine;
use medh5::curation::quality as q_engine;
use medh5::curation::timeline as tl_engine;

use crate::convert::{json_to_py, py_to_json};
use crate::errors::R;
use crate::exclusive::Exclusive;
use crate::record_class;
use crate::records::{field, required, Kind};

// -- identity ------------------------------------------------------------------------------

record_class!(
    /// The `identity` member of the sample document: who and what the sample is about (§12.1).
    ///
    /// `sample_id` and `subject_id` are required; a sample never spans subjects, so
    /// splitting by file is subject-safe. Per-occasion identifiers belong to the
    /// timepoints (§3.7).
    Identity,
    "Identity",
    "medh5.curation",
    id_engine::Identity,
    parse = id_engine::Identity::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("sample_id", Kind::Plain),
        required("subject_id", Kind::Plain),
        field("sex", Kind::Plain),
        field("laterality", Kind::Plain),
        field("bodypart", Kind::Plain),
        field("extra", Kind::Extra),
    ]
);

record_class!(
    /// The `cohort` member of the sample document: dataset, site, scanner and split group (§12.2).
    ///
    /// `group_id` defaults to the subject (`grouping_key`) and is the grouping key
    /// for leakage-free splits.
    Cohort,
    "Cohort",
    "medh5.curation",
    id_engine::Cohort,
    parse = |v| id_engine::Cohort::from_json(Some(v)),
    dump = |v| v.to_json(),
    fields = [
        field("dataset_id", Kind::Plain),
        field("site_id", Kind::Plain),
        field("scanner_id", Kind::Plain),
        field("group_id", Kind::Plain),
        field("acquisition_protocol", Kind::Plain),
        field("extra", Kind::Extra),
    ],
    methods = {
        /// The key subjects are grouped by when splitting (§12.2).
        fn grouping_key(&self, subject_id: &str) -> String {
            self.inner.grouping_key(subject_id).to_string()
        }
    }
);

record_class!(
    /// One entry of `splits`: a claim that the sample belongs to a partition of a split set (§12.3).
    ///
    /// A claim, not an authority: the dataset-level manifest decides, and
    /// `manifest_sha256` lets a reader tell a claim that predates the current
    /// split.
    SplitClaim,
    "SplitClaim",
    "medh5.curation",
    id_engine::SplitClaim,
    parse = id_engine::SplitClaim::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("set_id", Kind::Plain),
        required("partition", Kind::Plain),
        field("fold", Kind::Plain),
        field("assigned_by", Kind::Plain),
        field("assigned_at", Kind::Plain),
        field("manifest_sha256", Kind::Plain),
    ]
);

fn parse_deidentification(v: &Value) -> medh5::Result<id_engine::Deidentification> {
    id_engine::Deidentification::from_json(Some(v))?
        .ok_or_else(|| medh5::Error::invalid("a de-identification record needs at least its `method` (§11.4)"))
}

record_class!(
    /// The `deidentification` member: how identifiers were removed from the sample (§11.4).
    ///
    /// `method` is required. `from_json` of an absent or empty document is `None`,
    /// and a file without one is to be treated as potentially identifying.
    Deidentification,
    "Deidentification",
    "medh5.curation",
    id_engine::Deidentification,
    parse = parse_deidentification,
    dump = |v| v.to_json(),
    nullable = true,
    fields = [
        required("method", Kind::Plain),
        field("profile", Kind::Plain),
        field("date_shift_days", Kind::Plain),
        field("id_mapping", Kind::Plain),
        field("performed_by", Kind::Plain),
        field("date", Kind::Plain),
        field("burned_in_annotation_checked", Kind::Plain),
        field("extra", Kind::Extra).deprecated(
            "medh5 3.0 removes Deidentification.extra: the schema allows no other member, so anything it holds \
             makes the document invalid"
        ),
    ]
);

/// Parse the `splits` member of a sample document into a tuple of `SplitClaim`s (§12.3).
///
/// `None`, or anything but a list, gives an empty tuple.
#[pyfunction]
fn splits_from_json<'py>(py: Python<'py>, docs: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
    let doc = if docs.is_none() { None } else { Some(py_to_json(docs)?) };
    let claims = id_engine::splits_from_json(doc.as_ref())?;
    Ok(PyTuple::new(py, claims.into_iter().map(SplitClaim::wrap))?)
}

// -- provenance ------------------------------------------------------------------------------

record_class!(
    /// An agent of the provenance graph: a person, software or organization that acts (§11.1).
    Agent,
    "Agent",
    "medh5.curation",
    prov_engine::Agent,
    parse = prov_engine::Agent::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("id", Kind::Plain),
        required("type", Kind::Plain),
        required("name", Kind::Plain),
        field("role", Kind::Plain),
        field("version", Kind::Plain),
        field("qualification", Kind::Plain),
        field("organization", Kind::Plain),
    ]
);

record_class!(
    /// One activity of the provenance graph: what an agent did, with its inputs and outputs (§11.1).
    ///
    /// `type` is one of `import`, `annotate`, `review`, `predict`, `resample`,
    /// `register`, `derive`, `deidentify`, `transcode` or `other`; `started` and
    /// `ended` are RFC 3339 timestamps.
    Activity,
    "Activity",
    "medh5.curation",
    prov_engine::Activity,
    parse = prov_engine::Activity::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("id", Kind::Plain),
        required("type", Kind::Plain),
        field("agent", Kind::Plain),
        field("started", Kind::Plain),
        field("ended", Kind::Plain),
        field("tool", Kind::Plain),
        field("inputs", Kind::Tuple),
        field("outputs", Kind::Tuple),
        field("params", Kind::Dict),
    ]
);

/// Who did what (§11.1): agents and the activities they performed.
///
/// The activities are the graph's own objects, as 1.x kept them --- those it
/// was given, and from the first read those it hands out --- so an edit to
/// one's `params` is the graph's.  A document's provenance makes every change
/// to the document as well.
#[pyclass(module = "medh5.curation", name = "Provenance", skip_from_py_object, frozen)]
pub struct Provenance {
    /// Changed by one thread at a time (E4).
    state: Exclusive<ProvenanceState>,
}

struct ProvenanceState {
    /// The agents, and the activities as given or read.
    inner: prov_engine::Provenance,
    /// The activity objects, once given or read, in `inner`'s order.
    activities: Option<Vec<Py<Activity>>>,
    /// The document this is the provenance of.
    document: Option<Py<PyAny>>,
}

fn agent_arg(obj: &Bound<'_, PyAny>) -> R<prov_engine::Agent> {
    Agent::arg(obj)
}

fn activity_arg(obj: &Bound<'_, PyAny>) -> R<prov_engine::Activity> {
    Activity::arg(obj)
}

impl ProvenanceState {
    /// The object for an activity of the graph: for a document's
    /// provenance, one whose edits reach the document's activity.
    fn object(&self, py: Python<'_>, activity: prov_engine::Activity) -> PyResult<Py<Activity>> {
        match &self.document {
            Some(document) => {
                let id = activity.id.clone();
                Py::new(py, Activity::read_from(activity, document.bind(py), "provenance.activities", Some(&id)))
            }
            None => Py::new(py, Activity::wrap(activity)),
        }
    }

    /// The activity objects, made on the first read.
    fn objects(&mut self, py: Python<'_>) -> PyResult<&Vec<Py<Activity>>> {
        if self.activities.is_none() {
            let made = self.inner.activities().map(|a| self.object(py, a.clone())).collect::<PyResult<Vec<_>>>()?;
            self.activities = Some(made);
        }
        Ok(self.activities.as_ref().expect("made"))
    }
}

impl Provenance {
    fn of(inner: prov_engine::Provenance, activities: Option<Vec<Py<Activity>>>, document: Option<Py<PyAny>>) -> Self {
        Provenance { state: Exclusive::new("a Provenance", ProvenanceState { inner, activities, document }) }
    }

    pub fn wrap(inner: prov_engine::Provenance) -> Self {
        Provenance::of(inner, None, None)
    }

    /// The provenance of `document`, read as `inner`.
    pub fn read_from(inner: prov_engine::Provenance, document: &Bound<'_, PyAny>) -> Self {
        Provenance::of(inner, None, Some(document.clone().unbind()))
    }

    /// The activity objects whose activity `keep` selects.
    fn select<'py>(&self, py: Python<'py>, keep: impl Fn(&prov_engine::Activity) -> bool) -> R<Bound<'py, PyTuple>> {
        let pick = |state: &ProvenanceState, objects: &[Py<Activity>]| -> Vec<Py<Activity>> {
            state.inner.activities().zip(objects).filter(|(a, _)| keep(a)).map(|(_, o)| o.clone_ref(py)).collect()
        };
        let chosen = {
            let state = self.state.read()?;
            state.activities.as_ref().map(|objects| pick(&state, objects))
        };
        let chosen = match chosen {
            Some(chosen) => chosen,
            None => {
                // The first read makes the objects, alone.
                let mut state = self.state.write()?;
                let objects = state.objects(py)?.iter().map(|o| o.clone_ref(py)).collect::<Vec<_>>();
                pick(&state, &objects)
            }
        };
        Ok(PyTuple::new(py, chosen)?)
    }

    /// The graph as it stands: its activities as edited.
    pub fn current(&self, py: Python<'_>) -> R<prov_engine::Provenance> {
        loop {
            let (objects, anchored) = {
                let state = self.state.read()?;
                let Some(objects) = &state.activities else { return Ok(state.inner.clone()) };
                (objects.iter().map(|o| o.clone_ref(py)).collect::<Vec<_>>(), state.document.is_some())
            };
            // Read under the lock a document's activity takes for an edit to
            // its `params`, so the graph is one state of every activity; the
            // objects are read outside this provenance's lock (reading them
            // runs Python), so the graph is taken as it was read only if it
            // still holds them, and its agents with them.  An activity
            // replaced meanwhile, and edited by whoever kept it, reads again.
            let read = crate::document::under_live_lock_if(py, anchored, || {
                let activities = objects.iter().map(|a| activity_arg(a.bind(py).as_any())).collect::<R<Vec<_>>>()?;
                let state = self.state.read()?;
                let held = state
                    .activities
                    .as_ref()
                    .is_some_and(|now| now.len() == objects.len() && now.iter().zip(&objects).all(|(a, b)| a.is(b)));
                Ok(held.then(|| (state.inner.agents().cloned().collect::<Vec<_>>(), activities)))
            })?;
            if let Some((agents, activities)) = read {
                return Ok(prov_engine::Provenance::new(agents, activities)?);
            }
        }
    }
}

impl ProvenanceState {
    /// Make a change to the document's provenance, for a document's
    /// provenance; whether there was one to make it to.  The caller holds
    /// this provenance's lock across the change and its own update, so a
    /// second thread is refused rather than writing the document in one
    /// order and this graph in the other.  The document's lock is the
    /// engine's, and no Python runs under either.
    fn write_through(
        &self,
        py: Python<'_>,
        change: impl FnOnce(&mut prov_engine::Provenance) -> medh5::Result<()>,
    ) -> R<bool> {
        let Some(document) = &self.document else { return Ok(false) };
        let document = document.bind(py).cast::<crate::document::SampleDocument>().map_err(PyErr::from)?;
        document.get().edit(py, |d| change(&mut d.provenance))??;
        Ok(true)
    }
}

#[pymethods]
impl Provenance {
    #[new]
    #[pyo3(signature = (agents=None, activities=None))]
    fn new(py: Python<'_>, agents: Option<&Bound<'_, PyAny>>, activities: Option<&Bound<'_, PyAny>>) -> R<Self> {
        let mut a = Vec::new();
        if let Some(items) = agents {
            for item in items.try_iter()? {
                a.push(agent_arg(&item?)?);
            }
        }
        let mut b = Vec::new();
        let mut objects = Vec::new();
        if let Some(items) = activities {
            for item in items.try_iter()? {
                let item = item?;
                let value = activity_arg(&item)?;
                objects.push(match item.cast::<Activity>() {
                    Ok(given) => given.clone().unbind(),
                    Err(_) => Py::new(py, Activity::wrap(value.clone()))?,
                });
                b.push(value);
            }
        }
        Ok(Provenance::of(prov_engine::Provenance::new(a, b)?, Some(objects), None))
    }
    fn __repr__(&self) -> R<String> {
        Ok(self.state.read()?.inner.repr())
    }
    #[getter]
    fn agents<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        let agents = self.state.read()?.inner.agents().cloned().map(Agent::wrap).collect::<Vec<_>>();
        Ok(PyTuple::new(py, agents)?)
    }
    #[getter]
    fn activities<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        self.select(py, |_| true)
    }
    fn agent(&self, agent_id: &str) -> R<Agent> {
        Ok(Agent::wrap(self.state.read()?.inner.agent(agent_id)?.clone()))
    }
    fn activity(&self, py: Python<'_>, activity_id: &str) -> R<Py<Activity>> {
        self.state.read()?.inner.activity(activity_id)?;
        let found = self.select(py, |a| a.id == activity_id)?;
        Ok(found.get_item(0)?.cast_into::<Activity>().map_err(PyErr::from)?.unbind())
    }
    fn has_activity(&self, activity_id: &str) -> R<bool> {
        Ok(self.state.read()?.inner.has_activity(activity_id))
    }
    fn has_agent(&self, agent_id: &str) -> R<bool> {
        Ok(self.state.read()?.inner.has_agent(agent_id))
    }
    /// Add an agent; an id already in the graph is refused unless `replace`.
    #[pyo3(signature = (agent, *, replace=false))]
    fn add_agent<'py>(&self, agent: &Bound<'py, PyAny>, replace: bool) -> R<Bound<'py, PyAny>> {
        let value = agent_arg(agent)?;
        // The document first: what it refuses is refused here too.
        let mut state = self.state.write()?;
        let written = state.write_through(agent.py(), |p| p.add_agent(value.clone(), replace).map(drop))?;
        state.inner.add_agent(value.clone(), replace || written)?;
        drop(state);
        Ok(match agent.cast::<Agent>() {
            Ok(given) => given.clone().into_any(),
            Err(_) => Bound::new(agent.py(), Agent::wrap(value))?.into_any(),
        })
    }
    /// Add an activity; an id already in the graph is refused unless
    /// `replace`.  The activity given is kept, as 1.x kept it; a document's
    /// provenance keeps, and returns, the document's.
    #[pyo3(signature = (activity, *, replace=false))]
    fn add_activity(&self, activity: &Bound<'_, PyAny>, replace: bool) -> R<Py<Activity>> {
        let py = activity.py();
        let value = activity_arg(activity)?;
        let mut state = self.state.write()?;
        let written = state.write_through(py, |p| p.add_activity(value.clone(), replace).map(drop))?;
        state.objects(py)?;
        let at = state.inner.activities().position(|a| a.id == value.id);
        state.inner.add_activity(value.clone(), replace || written)?;
        let object = match activity.cast::<Activity>() {
            Ok(given) if !written => given.clone().unbind(),
            _ => state.object(py, value)?,
        };
        let objects = state.activities.get_or_insert_with(Vec::new);
        match at {
            Some(i) => objects[i] = object.clone_ref(py),
            None => objects.push(object.clone_ref(py)),
        }
        Ok(object)
    }
    fn activities_by_type<'py>(&self, py: Python<'py>, activity_type: &str) -> R<Bound<'py, PyTuple>> {
        self.select(py, |a| a.r#type == activity_type)
    }
    fn produced_by<'py>(&self, py: Python<'py>, object_path: &str) -> R<Bound<'py, PyTuple>> {
        self.select(py, |a| a.outputs.iter().any(|o| o == object_path))
    }
    fn dangling_agent_refs<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyTuple>> {
        let refs = self.state.read()?.inner.dangling_agent_refs();
        Ok(PyTuple::new(py, refs)?)
    }
    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(slf.getattr("activities")?.try_iter()?.into_any())
    }
    /// Whether the graph holds anything --- an agent or an activity --- as
    /// 1.x's did: `len` counts activities only, and a graph that has
    /// declared its agents and no activity yet is not empty.
    fn __bool__(&self) -> R<bool> {
        Ok(!self.state.read()?.inner.is_empty())
    }
    fn __len__(&self) -> R<usize> {
        Ok(self.state.read()?.inner.n_activities())
    }
    /// The graph as it stands, as JSON.
    fn to_json<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.current(py)?.to_json())?)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Self> {
        let value = if doc.is_none() { None } else { Some(py_to_json(doc)?) };
        Ok(Provenance::wrap(prov_engine::Provenance::from_json(value.as_ref())?))
    }
    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        let Ok(other) = other.cast::<Provenance>() else { return false };
        match (self.current(py), other.get().current(py)) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        crate::values::reduce_via_json(slf.as_any())
    }
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        let Some(state) = self.state.peek() else { return Ok(()) };
        for activity in state.activities.iter().flatten() {
            visit.call(activity)?;
        }
        visit.call(&state.document)
    }
    fn __clear__(&self) {
        if let Some(mut state) = self.state.try_write() {
            state.activities = None;
            state.document = None;
        }
    }
}

/// Return `value` if it is an RFC 3339 timestamp, else raise E604 (§11.1).
///
/// `where` names the field in the message of the `MEDH5ValidationError`.
#[pyfunction]
#[pyo3(signature = (value, *, r#where))]
fn check_timestamp(value: &str, r#where: &str) -> R<String> {
    prov_engine::check_timestamp(value, r#where)?;
    Ok(value.to_string())
}

// -- quality --------------------------------------------------------------------------------

record_class!(
    /// One agreement measurement of a quality record, such as a Dice against another rater (§11.2).
    Agreement,
    "Agreement",
    "medh5.curation",
    q_engine::Agreement,
    parse = q_engine::Agreement::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("metric", Kind::Plain),
        required("value", Kind::Float),
        field("against", Kind::Plain),
        field("per_class", Kind::Dict),
    ]
);

record_class!(
    /// A known defect recorded in a quality record: its code, severity, classes and note (§11.2).
    ///
    /// `severity` is `info` (the default), `warning` or `error`.
    Issue,
    "Issue",
    "medh5.curation",
    q_engine::Issue,
    parse = q_engine::Issue::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("code", Kind::Plain),
        field("severity", Kind::Default("info")),
        field("class_ids", Kind::Tuple),
        field("note", Kind::Plain),
    ]
);

record_class!(
    /// What is known about an annotation's trustworthiness: status, confidence, agreement, issues (§11.2).
    ///
    /// An annotation names its record by its `quality` attribute. Status changes
    /// are provenance activities (§11.1), not fields with a history of their own.
    QualityRecord,
    "QualityRecord",
    "medh5.curation",
    q_engine::QualityRecord,
    parse = q_engine::QualityRecord::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("status", Kind::Plain),
        field("confidence", Kind::Float),
        field("reviewed_by", Kind::Tuple),
        field("agreement", Kind::Records(Agreement::py_from_json)),
        field("issues", Kind::Records(Issue::py_from_json)),
        field("edit_effort_s", Kind::Float),
    ],
    methods = {
        /// Whether the record may be trained on: not rejected or deprecated.
        #[getter]
        fn is_usable(&self) -> bool {
            self.inner.is_usable()
        }
    }
);

/// Parse the `quality` member of a sample document into a dict of key to `QualityRecord` (§11.2).
///
/// `None`, or anything but an object, gives an empty dict.
#[pyfunction]
fn quality_from_json<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> R<Bound<'py, PyDict>> {
    let value = if doc.is_none() { None } else { Some(py_to_json(doc)?) };
    let records = q_engine::quality_from_json(value.as_ref())?;
    let out = PyDict::new(py);
    for (k, v) in records {
        out.set_item(k, QualityRecord::wrap(v))?;
    }
    Ok(out)
}

/// Serialize a mapping of key to `QualityRecord` (or its JSON form) as the `quality` member (§11.2).
#[pyfunction]
fn quality_to_json<'py>(py: Python<'py>, records: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
    let mut map = indexmap::IndexMap::new();
    for item in records.call_method0("items")?.try_iter()? {
        let (k, v): (String, Bound<'_, PyAny>) = item?.extract()?;
        map.insert(k, QualityRecord::arg(&v)?);
    }
    Ok(json_to_py(py, &q_engine::quality_to_json(&map))?)
}

/// Return a mean-Dice `Agreement` from a mapping of class id to that class's Dice (§11.2).
///
/// `against` names what the annotation was compared with. An empty mapping is
/// refused: a mean over no class is not 0.
#[pyfunction]
#[pyo3(signature = (per_class, against=None))]
fn dice_agreement(per_class: &Bound<'_, PyAny>, against: Option<String>) -> R<Agreement> {
    let mut pairs = Vec::new();
    for item in per_class.call_method0("items")?.try_iter()? {
        let (k, v): (i64, f64) = item?.extract()?;
        pairs.push((k, v));
    }
    Ok(Agreement::wrap(q_engine::dice_agreement(&pairs, against)?))
}

// -- timeline --------------------------------------------------------------------------------

record_class!(
    /// One timepoint of the sample: an observation occasion, in DICOM terms usually one study (§3.7).
    ///
    /// `id` and `index`, the 0-based acquisition order, are required.
    /// `days_from_baseline` survives date shifting and is what models should
    /// consume.
    Timepoint,
    "Timepoint",
    "medh5.curation",
    tl_engine::Timepoint,
    parse = tl_engine::Timepoint::from_json,
    dump = |v| v.to_json(),
    fields = [
        required("id", Kind::Plain),
        required("index", Kind::Plain),
        field("label", Kind::Plain),
        field("date", Kind::Plain),
        field("days_from_baseline", Kind::Plain),
        field("study_uid", Kind::Plain),
        field("series_uids", Kind::Dict),
        field("subject_age_years", Kind::Plain),
        field("description", Kind::Plain),
    ]
);

/// The sample's timepoints, in acquisition order; indexable by position or id.
///
/// The timepoints are the timeline's own objects, as 1.x kept them --- those
/// it was given, and from the first read those it hands out --- so an edit to
/// one's `series_uids` is the timeline's, and for a document's timeline, the
/// document's.
#[pyclass(module = "medh5.curation", name = "Timeline", skip_from_py_object, frozen)]
pub struct Timeline {
    /// The timepoints as given or read.
    pub inner: tl_engine::Timeline,
    /// The timepoint objects, once given or read, in `inner`'s order.
    points: OnceLock<Py<PyTuple>>,
    /// The document this is the timeline of.
    document: Option<Py<PyAny>>,
}

impl Timeline {
    pub fn wrap(inner: tl_engine::Timeline) -> Self {
        Timeline { inner, points: OnceLock::new(), document: None }
    }

    /// The timeline of `document`, read as `inner`.
    pub fn read_from(inner: tl_engine::Timeline, document: &Bound<'_, PyAny>) -> Self {
        Timeline { inner, points: OnceLock::new(), document: Some(document.clone().unbind()) }
    }

    fn objects<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyTuple>> {
        if self.points.get().is_none() {
            let made = self
                .inner
                .iter()
                .map(|t| match &self.document {
                    Some(document) => {
                        Py::new(py, Timepoint::read_from(t.clone(), document.bind(py), "timepoints", Some(&t.id)))
                    }
                    None => Py::new(py, Timepoint::wrap(t.clone())),
                })
                .collect::<PyResult<Vec<_>>>()?;
            // A read on another thread may have got there first; either will do.
            let _ = self.points.set(PyTuple::new(py, made)?.unbind());
        }
        Ok(self.points.get().expect("made").bind(py))
    }

    fn position(&self, id: &str) -> Option<usize> {
        self.inner.iter().position(|t| t.id == id)
    }

    /// The timeline as it stands: its timepoints as edited.
    pub fn current(&self, py: Python<'_>) -> R<tl_engine::Timeline> {
        let Some(points) = self.points.get() else { return Ok(self.inner.clone()) };
        // Every timepoint in one step under the live views' lock, so that
        // edits to two timepoints are read both or neither.
        let points = crate::document::under_live_lock_if(py, self.document.is_some(), || {
            points.bind(py).iter().map(|p| Timepoint::arg(&p)).collect::<R<Vec<_>>>()
        })?;
        Ok(tl_engine::Timeline::new(points)?)
    }
}

/// A timeline argument: the binding class as it stands, or a list of
/// timepoints.
pub fn timeline_arg(obj: &Bound<'_, PyAny>) -> R<tl_engine::Timeline> {
    if let Ok(t) = obj.cast::<Timeline>() {
        return t.get().current(obj.py());
    }
    let mut points = Vec::new();
    for item in obj.try_iter()? {
        points.push(Timepoint::arg(&item?)?);
    }
    Ok(tl_engine::Timeline::new(points)?)
}

#[pymethods]
impl Timeline {
    /// A timeline of `timepoints`, which it keeps, in index order, as 1.x
    /// kept them.
    #[new]
    fn new(py: Python<'_>, timepoints: &Bound<'_, PyAny>) -> R<Self> {
        let mut given = Vec::new();
        for item in timepoints.try_iter()? {
            let item = item?;
            let value = Timepoint::arg(&item)?;
            let object = match item.cast::<Timepoint>() {
                Ok(point) => point.clone().unbind(),
                Err(_) => Py::new(py, Timepoint::wrap(value.clone()))?,
            };
            given.push((value, object));
        }
        // Stable, as the engine's own sort is: the objects stay beside theirs.
        given.sort_by_key(|(value, _)| value.index);
        let (values, objects): (Vec<_>, Vec<_>) = given.into_iter().unzip();
        let made = Timeline::wrap(tl_engine::Timeline::new(values)?);
        let _ = made.points.set(PyTuple::new(py, objects)?.unbind());
        Ok(made)
    }
    fn check(&self) -> R<()> {
        Ok(self.inner.check()?)
    }
    /// The position of the first timepoint equal to `value` (`Sequence.index`).
    #[pyo3(signature = (value, start=0, stop=None))]
    fn index(&self, py: Python<'_>, value: &Bound<'_, PyAny>, start: isize, stop: Option<isize>) -> PyResult<usize> {
        let items = self.objects(py)?;
        let n = items.len() as isize;
        let clamp = |i: isize| (if i < 0 { (i + n).max(0) } else { i.min(n) }) as usize;
        let (lo, hi) = (clamp(start), clamp(stop.unwrap_or(n)));
        for i in lo..hi.max(lo) {
            if items.get_item(i)?.eq(value)? {
                return Ok(i);
            }
        }
        Err(pyo3::exceptions::PyValueError::new_err("timepoint is not in the timeline"))
    }
    /// How many timepoints equal `value` (`Sequence.count`).
    fn count(&self, py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<usize> {
        let mut found = 0;
        for item in self.objects(py)?.iter() {
            if item.eq(value)? {
                found += 1;
            }
        }
        Ok(found)
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(self.objects(py)?.try_iter()?.into_any())
    }
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let objects = self.objects(py)?;
        if let Ok(id) = key.cast::<PyString>() {
            let id = id.to_string();
            return match self.position(&id) {
                Some(i) => objects.get_item(i),
                None => Err(PyKeyError::new_err(format!(
                    "undeclared timepoint {}; declared: {}",
                    medh5::json::repr_str(&id),
                    medh5::json::repr_list(&self.inner.ids())
                ))),
            };
        }
        if key.is_instance_of::<PySlice>() {
            return objects.as_any().get_item(key);
        }
        objects.as_any().get_item(key)
    }
    fn __contains__(&self, py: Python<'_>, item: &Bound<'_, PyAny>) -> PyResult<bool> {
        if let Ok(id) = item.cast::<PyString>() {
            return Ok(self.inner.contains(&id.to_string()));
        }
        self.objects(py)?.contains(item)
    }
    fn __repr__(&self) -> String {
        self.inner.repr()
    }
    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        let Ok(other) = other.cast::<Timeline>() else { return false };
        match (self.current(py), other.get().current(py)) {
            (Ok(a), Ok(b)) => a == b,
            _ => false,
        }
    }
    #[getter]
    fn ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.ids())
    }
    #[getter]
    fn baseline<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.objects(py)?.get_item(0)
    }
    #[getter]
    fn is_longitudinal(&self) -> bool {
        self.inner.is_longitudinal()
    }
    fn interval_days(&self, a: &str, b: &str) -> R<Option<f64>> {
        Ok(self.inner.interval_days(a, b)?)
    }
    #[pyo3(signature = (timepoint_id, *, r#where=""))]
    fn require<'py>(&self, py: Python<'py>, timepoint_id: &str, r#where: &str) -> R<Bound<'py, PyAny>> {
        let found = self.inner.require(timepoint_id, r#where)?;
        let index = self.position(&found.id).unwrap_or(0);
        Ok(self.objects(py)?.get_item(index)?)
    }
    /// The timeline as it stands, as JSON.
    fn to_json<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.current(py)?.to_json())?)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, docs: &Bound<'_, PyAny>) -> R<Self> {
        Ok(Timeline::wrap(tl_engine::Timeline::from_json(&py_to_json(docs)?)?))
    }
    #[classmethod]
    #[pyo3(signature = (timepoint_id="tp0", **kwargs))]
    fn single(
        _cls: &Bound<'_, pyo3::types::PyType>,
        timepoint_id: &str,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> R<Self> {
        let mut doc = crate::convert::kwargs_to_map(kwargs)?;
        doc.insert("id".into(), Value::String(timepoint_id.into()));
        doc.insert("index".into(), Value::from(0));
        let point = tl_engine::Timepoint::from_json(&Value::Object(doc))?;
        Ok(Timeline::wrap(tl_engine::Timeline::new(vec![point])?))
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        crate::values::reduce_via_json(slf.as_any())
    }
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(self.points.get())?;
        visit.call(&self.document)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add_class::<Identity>()?;
    m.add_class::<Cohort>()?;
    m.add_class::<SplitClaim>()?;
    m.add_class::<Deidentification>()?;
    m.add_class::<Agent>()?;
    m.add_class::<Activity>()?;
    m.add_class::<Provenance>()?;
    m.add_class::<Agreement>()?;
    m.add_class::<Issue>()?;
    m.add_class::<QualityRecord>()?;
    m.add_class::<Timepoint>()?;
    m.add_class::<Timeline>()?;
    for f in [
        wrap_pyfunction!(splits_from_json, m)?,
        wrap_pyfunction!(check_timestamp, m)?,
        wrap_pyfunction!(quality_from_json, m)?,
        wrap_pyfunction!(quality_to_json, m)?,
        wrap_pyfunction!(dice_agreement, m)?,
    ] {
        m.add_function(f)?;
    }
    m.add("SEX_VALUES", PyTuple::new(py, id_engine::SEX_VALUES)?)?;
    m.add("LATERALITY_VALUES", PyTuple::new(py, id_engine::LATERALITY_VALUES)?)?;
    m.add("PARTITIONS", PyTuple::new(py, id_engine::PARTITIONS)?)?;
    m.add("ID_SOURCE", id_engine::ID_SOURCE)?;
    m.add("PSEUDONYM_SOURCE", id_engine::PSEUDONYM_SOURCE)?;
    m.add("AGENT_TYPES", PyTuple::new(py, prov_engine::AGENT_TYPES)?)?;
    m.add("AGENT_FIELDS", PyFrozenSet::new(py, prov_engine::AGENT_FIELDS)?)?;
    m.add("ACTIVITY_FIELDS", PyFrozenSet::new(py, prov_engine::ACTIVITY_FIELDS)?)?;
    m.add("ACTIVITY_TYPES", PyTuple::new(py, prov_engine::ACTIVITY_TYPES)?)?;
    m.add("QUALITY_FIELDS", PyFrozenSet::new(py, q_engine::QUALITY_FIELDS)?)?;
    m.add("QUALITY_STATUS", PyTuple::new(py, q_engine::QUALITY_STATUS)?)?;
    m.add("ISSUE_SEVERITY", PyTuple::new(py, q_engine::ISSUE_SEVERITY)?)?;
    m.add("TIMEPOINT_FIELDS", PyFrozenSet::new(py, tl_engine::TIMEPOINT_FIELDS)?)?;
    m.add("RFC3339_REGEX", prov_engine::RFC3339)?;
    m.add("DATE_REGEX", tl_engine::DATE_PATTERN)?;
    Ok(())
}
