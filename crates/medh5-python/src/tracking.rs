//! `medh5.curation`: joining `instance_id` across timepoints (§7.4).

use indexmap::IndexMap;
use ndarray::{ArrayD, IxDyn};
use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple, PyType};

use medh5::curation::tracking as engine;

use crate::convert::{array_to_py, json_to_py, typed_array};
use crate::errors::R;

/// One object seen once: a row of one annotation.
#[pyclass(module = "medh5.curation", name = "Observation", skip_from_py_object, frozen)]
#[derive(Clone)]
pub struct Observation(pub engine::Observation);

impl Observation {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] = &[
        "timepoint",
        "annotation",
        "index",
        "instance_id",
        "class_id",
        "box",
        "voxel_count",
        "volume",
        "units",
        "score",
        "grid",
    ];

    /// What `_rebuild` takes: the fields, and `kind`.
    const STATE: &'static [&'static str] = &[
        "timepoint",
        "annotation",
        "kind",
        "index",
        "instance_id",
        "class_id",
        "box",
        "voxel_count",
        "volume",
        "units",
        "score",
        "grid",
    ];
}

#[pymethods]
impl Observation {
    #[classattr]
    fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        crate::geometry::dataclass_fields(py, Self::FIELDS)
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::values::match_args(py, Self::FIELDS)
    }

    #[getter]
    fn timepoint(&self) -> &str {
        &self.0.timepoint
    }
    #[getter]
    fn annotation(&self) -> &str {
        &self.0.annotation
    }
    /// The annotation's kind: `instances` (a mask), `boxes`, `obb`, or
    /// `points` or `polylines` (presence only).  Not one of the 1.x
    /// dataclass's fields, so its shape and `repr` are unchanged.
    #[getter]
    fn kind(&self) -> &str {
        &self.0.kind
    }
    #[getter]
    fn index(&self) -> usize {
        self.0.index
    }
    #[getter]
    fn instance_id(&self) -> u64 {
        self.0.instance_id
    }
    #[getter]
    fn class_id(&self) -> i64 {
        self.0.class_id
    }
    #[getter(r#box)]
    fn bbox<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        array_to_py(py, self.0.bbox.clone().into_dyn())
    }
    #[getter]
    fn voxel_count(&self) -> Option<u64> {
        self.0.voxel_count
    }
    /// Physical volume, in `units` to the power of the box's axes; `None`
    /// where nothing measured the object: a point, which marks it without
    /// measuring it, a box or mask in index space without a grid, or a join
    /// run with `measure=False`.
    #[getter]
    fn volume(&self) -> Option<f64> {
        self.0.volume
    }
    #[getter]
    fn units(&self) -> Option<&str> {
        self.0.units.as_deref()
    }
    #[getter]
    fn score(&self) -> Option<f64> {
        self.0.score
    }
    #[getter]
    fn grid(&self) -> Option<&str> {
        self.0.grid.as_deref()
    }
    #[getter]
    fn centroid<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let c = self.0.centroid();
        Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[c.len()]), c).expect("1-D")))
    }
    /// `volume` in millimetres to the power of the box's axes, or `None`
    /// where it is unmeasured or its grid is uncalibrated.
    #[getter]
    fn volume_mm(&self) -> Option<f64> {
        self.0.volume_mm()
    }
    #[getter]
    fn extent<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let e = self.0.extent();
        Ok(array_to_py(py, ArrayD::from_shape_vec(IxDyn(&[e.len()]), e).expect("1-D")))
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.to_json())
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<Observation>().map(|o| o.get().0 == self.0).unwrap_or(false)
    }
    fn __repr__(&self) -> String {
        self.0.repr()
    }
    /// Pickling and `copy`: the fields and `kind`, through `_rebuild` (only
    /// a join observes an object).
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        crate::values::reduce_fields(slf.as_any(), Some("_rebuild"), Self::STATE)
    }
    #[classmethod]
    #[allow(clippy::too_many_arguments)]
    fn _rebuild(
        _cls: &Bound<'_, PyType>,
        timepoint: String,
        annotation: String,
        kind: String,
        index: usize,
        instance_id: u64,
        class_id: i64,
        bbox: &Bound<'_, PyAny>,
        voxel_count: Option<u64>,
        volume: Option<f64>,
        units: Option<String>,
        score: Option<f64>,
        grid: Option<String>,
    ) -> R<Self> {
        Ok(Observation(engine::Observation {
            timepoint,
            annotation,
            kind,
            index,
            instance_id,
            class_id,
            bbox: typed_array::<f32>(bbox, "float32")?.into_dimensionality()?,
            voxel_count,
            volume,
            units,
            score,
            grid,
        }))
    }
}

/// Every observation of one physical object, ordered by timepoint.
#[pyclass(module = "medh5.curation", name = "Track", skip_from_py_object, frozen)]
#[derive(Clone)]
pub struct Track(pub engine::Track);

impl Track {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] = &["instance_id", "class_ids", "observations", "class_key"];
}

#[pymethods]
impl Track {
    #[classattr]
    fn __dataclass_fields__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        crate::geometry::dataclass_fields(py, Self::FIELDS)
    }

    #[classattr]
    fn __match_args__(py: Python<'_>) -> PyResult<Py<pyo3::types::PyTuple>> {
        crate::values::match_args(py, Self::FIELDS)
    }

    #[getter]
    fn instance_id(&self) -> u64 {
        self.0.instance_id
    }
    #[getter]
    fn class_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.0.class_ids)
    }
    #[getter]
    fn observations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.observations.iter().cloned().map(Observation))
    }
    #[getter]
    fn class_key(&self) -> Option<&str> {
        self.0.class_key.as_deref()
    }
    #[getter]
    fn class_id(&self) -> i64 {
        self.0.class_id()
    }
    #[getter]
    fn has_class_conflict(&self) -> bool {
        self.0.has_class_conflict()
    }
    #[getter]
    fn timepoints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.timepoints())
    }
    /// The observation that answers for `timepoint` --- the one
    /// `measurements_at` finds --- or `None` where the object was not seen.
    /// Refused (`MEDH5ValidationError`) where it finds two, as two raters'
    /// masks of one visit: which one is meant is the caller's to choose.
    /// `volume`, `volumes` and `relative_change` answer by the same rule.
    fn at(&self, timepoint: &str) -> R<Option<Observation>> {
        Ok(self.0.at(timepoint)?.cloned().map(Observation))
    }
    /// Every observation at `timepoint`: one per annotation that saw the
    /// object there --- several where raters annotated the same visit, or a
    /// detection sits beside a segmentation.
    fn observations_at<'py>(&self, py: Python<'py>, timepoint: &str) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.observations_at(timepoint).into_iter().cloned().map(Observation))
    }
    /// The observations that measure the object at `timepoint`: its masks
    /// (`instances`), or --- where no mask saw it --- its boxes, axis-aligned
    /// or oriented.  A point marks the object and measures nothing, so it
    /// answers only where nothing else saw it.
    fn measurements_at<'py>(&self, py: Python<'py>, timepoint: &str) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.0.measurements_at(timepoint).into_iter().cloned().map(Observation))
    }
    /// The volume of the observation `at` finds, or `None`; refused where
    /// `at` is.
    fn volume(&self, timepoint: &str) -> R<Option<f64>> {
        Ok(self.0.volume(timepoint)?)
    }
    /// `{timepoint: volume}` for every timepoint the object was seen at,
    /// under the rule of `at`.
    #[getter]
    fn volumes<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.0.volumes()? {
            out.set_item(k, v)?;
        }
        Ok(out)
    }
    /// `(v2 - v1) / v1` between the volumes at `first` and `second`, compared
    /// in millimetres; `None` where either is unmeasured, or the two share no
    /// physical measure (an uncalibrated grid beside a calibrated one, an
    /// area beside a volume).
    fn relative_change(&self, first: &str, second: &str) -> R<Option<f64>> {
        Ok(self.0.relative_change(first, second)?)
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.to_json())
    }
    fn __len__(&self) -> usize {
        self.0.observations.len()
    }
    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(slf.getattr("observations")?.try_iter()?.into_any())
    }
    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<Track>().map(|o| o.get().0 == self.0).unwrap_or(false)
    }
    fn __repr__(&self) -> String {
        self.0.repr()
    }
    /// Pickling and `copy`: the fields, through `_rebuild` (only a join
    /// builds a track).
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        crate::values::reduce_fields(slf.as_any(), Some("_rebuild"), Self::FIELDS)
    }
    #[classmethod]
    fn _rebuild(
        _cls: &Bound<'_, PyType>,
        instance_id: u64,
        class_ids: Vec<i64>,
        observations: Vec<Bound<'_, Observation>>,
        class_key: Option<String>,
    ) -> Self {
        let observations = observations.iter().map(|o| o.get().0.clone()).collect();
        Track(engine::Track { instance_id, class_ids, observations, class_key })
    }
}

/// `{instance_id: Track}` with the coverage needed to tell *resolved* from
/// *unexamined*, and the reason for every absence.
#[pyclass(module = "medh5.curation", name = "Tracking", frozen)]
pub struct Tracking(pub engine::LocatedTracking);

impl Tracking {
    pub fn wrap(t: engine::LocatedTracking) -> Self {
        Tracking(t)
    }

    fn tracks(&self) -> &std::collections::BTreeMap<u64, engine::Track> {
        &self.0.tracking.tracks
    }

    fn require(&self, instance_id: u64) -> PyResult<&engine::Track> {
        self.tracks().get(&instance_id).ok_or_else(|| PyKeyError::new_err(instance_id))
    }
}

#[pymethods]
impl Tracking {
    /// `KeyError` for any key no track has, as a mapping raises it: a key
    /// that is no instance id (`-1`, `"7"`) raised `OverflowError` or
    /// `TypeError`, which `get` and `in` already answered as absent.
    fn __getitem__(&self, instance_id: &Bound<'_, PyAny>) -> PyResult<Track> {
        match instance_id.extract::<u64>().ok().and_then(|i| self.tracks().get(&i)) {
            Some(track) => Ok(Track(track.clone())),
            None => Err(PyKeyError::new_err(instance_id.clone().unbind())),
        }
    }
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        Ok(PyTuple::new(py, self.tracks().keys())?.try_iter()?.into_any())
    }
    fn __len__(&self) -> usize {
        self.tracks().len()
    }
    fn __contains__(&self, instance_id: &Bound<'_, PyAny>) -> bool {
        instance_id.extract::<u64>().map(|i| self.tracks().contains_key(&i)).unwrap_or(false)
    }
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.tracks().keys())
    }
    fn values<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.tracks().values().cloned().map(Track))
    }
    fn items<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let pairs = self
            .tracks()
            .iter()
            .map(|(k, v)| {
                PyTuple::new(py, [k.into_pyobject(py)?.into_any(), Bound::new(py, Track(v.clone()))?.into_any()])
            })
            .collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, pairs)
    }
    #[pyo3(signature = (instance_id, default=None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        instance_id: &Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match instance_id.extract::<u64>().ok().and_then(|i| self.tracks().get(&i)) {
            Some(t) => Ok(Bound::new(py, Track(t.clone()))?.into_any()),
            None => Ok(default.unwrap_or_else(|| py.None().into_bound(py))),
        }
    }
    #[getter]
    fn timepoints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.0.tracking.timepoints)
    }
    /// `{annotation: why}` for the annotations that carry `instance_ids`
    /// and that no track joined.
    #[getter]
    fn skipped<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in &self.0.tracking.skipped {
            out.set_item(k, v)?;
        }
        Ok(out)
    }
    /// `{timepoint: frozenset of class ids}`: the classes the annotators at
    /// each timepoint committed to finding (§11.3), of the annotations there
    /// that ignore no part of their grid (§7.7).  Class-level: it does not
    /// say where anyone looked, which `reason_at` does.
    #[getter]
    fn coverage<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in &self.0.tracking.coverage {
            out.set_item(k, pyo3::types::PyFrozenSet::new(py, v)?)?;
        }
        Ok(out)
    }
    /// `present`, `resolved` or `unexamined`.  `resolved` is claimed only
    /// where the object was looked for there (`reason_at` says how); anywhere
    /// else an unseen object is `unexamined`, and its absence measures
    /// nothing.  `KeyError` for an id no track has.
    fn state_at(&self, instance_id: u64, timepoint: &str) -> PyResult<&'static str> {
        self.require(instance_id)?;
        Ok(self.0.state_at(instance_id, timepoint).unwrap_or(engine::UNEXAMINED))
    }
    /// `{timepoint: state_at(instance_id, timepoint)}` over every timepoint,
    /// in order.
    fn states<'py>(&self, py: Python<'py>, instance_id: u64) -> PyResult<Bound<'py, PyDict>> {
        self.require(instance_id)?;
        let out = PyDict::new(py);
        for (k, v) in self.0.states(instance_id) {
            out.set_item(k, v)?;
        }
        Ok(out)
    }
    /// Why the object is absent at `timepoint` (`REASONS`): `located` or
    /// `class_examined` where it is `resolved`, `outside_fov`, `ignored`,
    /// `unrelated`, `no_identity` or `not_examined` where it is
    /// `unexamined`; `None` where it was seen.
    fn reason_at(&self, instance_id: u64, timepoint: &str) -> PyResult<Option<&'static str>> {
        self.require(instance_id)?;
        Ok(self.0.reason_at(instance_id, timepoint).map(engine::Absence::as_str))
    }
    /// `{timepoint: reason_at(instance_id, timepoint)}` over every
    /// timepoint, in order.
    fn reasons<'py>(&self, py: Python<'py>, instance_id: u64) -> PyResult<Bound<'py, PyDict>> {
        self.require(instance_id)?;
        let out = PyDict::new(py);
        for (k, v) in self.0.reasons(instance_id) {
            out.set_item(k, v.map(engine::Absence::as_str))?;
        }
        Ok(out)
    }
    /// Absent-then-present: `resolved` at the first timepoint (looked for
    /// and not seen) and `present` at a later one.  With `located`, only
    /// where the object was located at the first (not by class alone).
    #[pyo3(signature = (instance_id, *, located=false))]
    fn is_new(&self, instance_id: u64, located: bool) -> PyResult<bool> {
        self.require(instance_id)?;
        Ok(self.0.is_new(instance_id, located))
    }
    /// Present-then-gone: `present` at the first timepoint and `resolved` at
    /// the last, whose annotators looked for it.  With `located`, only where
    /// the object was located at the last.
    #[pyo3(signature = (instance_id, *, located=false))]
    fn is_resolved(&self, instance_id: u64, located: bool) -> PyResult<bool> {
        self.require(instance_id)?;
        Ok(self.0.is_resolved(instance_id, located))
    }
    fn is_persistent(&self, instance_id: u64) -> PyResult<bool> {
        self.require(instance_id)?;
        Ok(self.0.is_persistent(instance_id))
    }
    fn class_conflicts<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.0.tracking.class_conflicts() {
            out.set_item(k, PyTuple::new(py, v)?)?;
        }
        Ok(out)
    }
    /// `{timepoint: instance ids}` of the objects that are `unexamined`
    /// there: not seen, and not looked for there.
    fn unexamined<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (k, v) in self.0.unexamined() {
            out.set_item(k, PyTuple::new(py, v)?)?;
        }
        Ok(out)
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.to_json())
    }
    fn summary<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.summary())
    }
    fn __repr__(&self) -> String {
        self.0.repr()
    }
    /// Pickling and `copy`: the tracks, timepoints, coverage, skipped
    /// annotations and reasons, through `_rebuild` (only a join builds one).
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let py = slf.py();
        let this = slf.get();
        let reasons = PyDict::new(py);
        for ((instance_id, timepoint), reason) in &this.0.absences {
            reasons.set_item((instance_id, timepoint), reason.as_str())?;
        }
        let state = PyTuple::new(
            py,
            [
                this.values(py)?.into_any(),
                this.timepoints(py)?.into_any(),
                this.coverage(py)?.into_any(),
                this.skipped(py)?.into_any(),
                reasons.into_any(),
            ],
        )?;
        Ok((slf.get_type().getattr("_rebuild")?, state))
    }
    /// A pickle from before 2.1 carries no reasons: each absence is then
    /// read by the class-level rule it was made under.
    #[classmethod]
    #[pyo3(signature = (tracks, timepoints, coverage, skipped, reasons=None))]
    fn _rebuild(
        _cls: &Bound<'_, PyType>,
        tracks: Vec<Bound<'_, Track>>,
        timepoints: Vec<String>,
        coverage: &Bound<'_, PyDict>,
        skipped: &Bound<'_, PyDict>,
        reasons: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let tracks: std::collections::BTreeMap<u64, engine::Track> =
            tracks.iter().map(|t| (t.get().0.instance_id, t.get().0.clone())).collect();
        let coverage = coverage
            .iter()
            .map(|(k, v)| Ok((k.extract()?, v.extract()?)))
            .collect::<PyResult<IndexMap<String, std::collections::BTreeSet<i64>>>>()?;
        let skipped =
            skipped.iter().map(|(k, v)| Ok((k.extract()?, v.extract()?))).collect::<PyResult<IndexMap<_, _>>>()?;
        let absences = match reasons {
            Some(reasons) => reasons
                .iter()
                .map(|(k, v)| {
                    let key: (u64, String) = k.extract()?;
                    let reason: String = v.extract()?;
                    let absence = engine::Absence::parse(&reason)
                        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err(format!("unknown reason {reason:?}")))?;
                    Ok((key, absence))
                })
                .collect::<PyResult<_>>()?,
            None => tracks
                .iter()
                .flat_map(|(instance_id, track)| {
                    let coverage = &coverage;
                    timepoints.iter().filter(|tp| track.observations_at(tp).is_empty()).map(move |tp| {
                        let looked = coverage.get(tp).is_some_and(|c| track.class_ids.iter().any(|i| c.contains(i)));
                        let reason = if looked { engine::Absence::ClassExamined } else { engine::Absence::NotExamined };
                        ((*instance_id, tp.clone()), reason)
                    })
                })
                .collect(),
        };
        let tracking = engine::Tracking { tracks, timepoints, coverage, skipped };
        Ok(Tracking(engine::LocatedTracking::new(tracking, absences)))
    }
}

/// Whether an annotation carries object identity (`instance_ids`) to join on.
#[pyfunction]
fn carries_instance_ids(annotation: &Bound<'_, PyAny>) -> R<bool> {
    let handle = match annotation.cast::<crate::reader::AnnotationHandle>() {
        Ok(h) => h.clone(),
        Err(_) => annotation.getattr("_handle")?.cast_into::<crate::reader::AnnotationHandle>().map_err(PyErr::from)?,
    };
    let annotation = &handle.get().inner;
    Ok(crate::h5(handle.py(), || engine::carries_instance_ids(annotation))?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(carries_instance_ids, m)?)?;
    m.add_class::<Observation>()?;
    m.add_class::<Track>()?;
    m.add_class::<Tracking>()?;
    m.add("PRESENT", engine::PRESENT)?;
    m.add("RESOLVED", engine::RESOLVED)?;
    m.add("UNEXAMINED", engine::UNEXAMINED)?;
    m.add("STATES", PyTuple::new(m.py(), engine::STATES)?)?;
    m.add("REASONS", PyTuple::new(m.py(), engine::REASONS)?)?;
    Ok(())
}
