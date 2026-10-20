//! `medh5.labels`: the vocabulary (spec §5).

use std::sync::OnceLock;

use pyo3::prelude::*;
use pyo3::pyclass::{PyTraverseError, PyVisit};
use pyo3::types::{PyDict, PyString, PyTuple};
use serde_json::{Map, Value};

use medh5::labels::{self as engine, registry, ClassKey};

use crate::convert::{json_to_py, map_to_py, py_to_json};
use crate::errors::R;
use crate::values::{dataclass_repr, json_hash, opt, reduce_via_json};

// -- OntologyCode ----------------------------------------------------------------------

/// A binding of a class to a code of an external vocabulary, such as SNOMED-CT or RadLex (§5.2).
#[pyclass(module = "medh5.labels", name = "OntologyCode", skip_from_py_object, frozen, eq)]
#[derive(Clone, PartialEq)]
pub struct OntologyCode(pub engine::OntologyCode);

impl OntologyCode {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] = &["system", "code", "name"];
}

#[pymethods]
impl OntologyCode {
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
    #[pyo3(signature = (system, code, name=None))]
    fn new(system: String, code: String, name: Option<String>) -> Self {
        OntologyCode(engine::OntologyCode { system, code, name })
    }
    #[getter]
    fn system(&self) -> &str {
        &self.0.system
    }
    #[getter]
    fn code(&self) -> &str {
        &self.0.code
    }
    #[getter]
    fn name(&self) -> Option<&str> {
        self.0.name.as_deref()
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.to_json())
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Self> {
        Ok(OntologyCode(engine::OntologyCode::from_json(&py_to_json(doc)?)?))
    }
    fn __hash__(&self) -> isize {
        json_hash(&self.0.to_json())
    }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        dataclass_repr(
            "OntologyCode",
            &[
                ("system", PyString::new(py, &self.0.system).into_any()),
                ("code", PyString::new(py, &self.0.code).into_any()),
                ("name", opt(py, self.0.name.as_deref())?),
            ],
        )
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        reduce_via_json(slf.as_any())
    }
}

// -- Relation ----------------------------------------------------------------------------

/// A non-`is_a` edge between two classes of a label set, such as `part_of` (§5.4).
///
/// `subject` and `object` are class ids, `predicate` the relation.
#[pyclass(module = "medh5.labels", name = "Relation", skip_from_py_object, frozen, eq, hash)]
#[derive(Clone, PartialEq, Hash)]
pub struct Relation(pub engine::Relation);

impl Relation {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] = &["subject", "predicate", "object"];
}

#[pymethods]
impl Relation {
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
    fn new(subject: i64, predicate: String, object: i64) -> Self {
        Relation(engine::Relation { subject, predicate, object })
    }
    #[getter]
    fn subject(&self) -> i64 {
        self.0.subject
    }
    #[getter]
    fn predicate(&self) -> &str {
        &self.0.predicate
    }
    #[getter]
    fn object(&self) -> i64 {
        self.0.object
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.to_json())
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Self> {
        Ok(Relation(engine::Relation::from_json(&py_to_json(doc)?)?))
    }
    fn __repr__(&self) -> String {
        format!(
            "Relation(subject={}, predicate={}, object={})",
            self.0.subject,
            medh5::json::repr_str(&self.0.predicate),
            self.0.object
        )
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        reduce_via_json(slf.as_any())
    }
}

// -- Skeleton ----------------------------------------------------------------------------

/// A keypoint topology declared by a label set: keypoint class ids and edges between them (§5.5).
#[pyclass(module = "medh5.labels", name = "Skeleton", skip_from_py_object, frozen, eq, hash)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Skeleton(pub engine::Skeleton);

impl Skeleton {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] = &["id", "keypoints", "edges"];
}

#[pymethods]
impl Skeleton {
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
    #[pyo3(signature = (id, keypoints, edges=Vec::new()), text_signature = "(id, keypoints, edges=())")]
    fn new(id: String, keypoints: Vec<i64>, edges: Vec<(i64, i64)>) -> Self {
        Skeleton(engine::Skeleton { id, keypoints, edges })
    }
    #[getter]
    fn id(&self) -> &str {
        &self.0.id
    }
    #[getter]
    fn keypoints<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.0.keypoints)
    }
    #[getter]
    fn edges<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.0.edges)
    }
    fn to_json<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        json_to_py(py, &self.0.to_json())
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Self> {
        Ok(Skeleton(engine::Skeleton::from_json(&py_to_json(doc)?)?))
    }
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        dataclass_repr(
            "Skeleton",
            &[
                ("id", PyString::new(py, &slf.get().0.id).into_any()),
                ("keypoints", slf.getattr("keypoints")?),
                ("edges", slf.getattr("edges")?),
            ],
        )
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        reduce_via_json(slf.as_any())
    }
}

// -- LabelClass --------------------------------------------------------------------------

/// One semantic class (§5.2).
///
/// `properties` is the class's own dict, as 1.x's dataclass field was: made
/// on its first read, the same object on every read after, and read back
/// whenever the class is used --- `to_json`, `==`, `is_lesion`, a label set
/// or a writer given it.  A class a document's label set hands out writes an
/// edit to it through to the document.
#[pyclass(module = "medh5.labels", name = "LabelClass", skip_from_py_object, frozen)]
pub struct LabelClass {
    /// The class as made.  Its `properties` may have been edited since:
    /// `current` is the class as it stands.
    pub inner: engine::LabelClass,
    /// `properties`, once read, and where the class was read from.
    held: crate::records::Held,
}

fn ontology_codes(codes: Option<&Bound<'_, PyAny>>) -> R<Vec<engine::OntologyCode>> {
    let mut out = Vec::new();
    let Some(codes) = codes else { return Ok(out) };
    for item in codes.try_iter()? {
        let item = item?;
        match item.cast::<OntologyCode>() {
            Ok(c) => out.push(c.get().0.clone()),
            Err(_) => out.push(engine::OntologyCode::from_json(&py_to_json(&item)?)?),
        }
    }
    Ok(out)
}

impl LabelClass {
    /// The fields of the 1.x dataclass, in order.
    const FIELDS: &'static [&'static str] =
        &["id", "key", "name", "parents", "category", "color", "codes", "laterality", "properties"];

    pub fn wrap(inner: engine::LabelClass) -> Self {
        LabelClass { inner, held: crate::records::Held::new(1) }
    }

    /// The class `key` of the label set of `document`, read as `inner`: an
    /// edit to its `properties` is written back to the document's class,
    /// while that is still what was read.
    pub fn read_from(inner: engine::LabelClass, document: &Bound<'_, PyAny>, key: &str) -> Self {
        let mut made = Self::wrap(inner);
        made.held.read_from(document.clone().unbind(), "label_set", Some(key));
        made
    }

    pub fn tuple_color<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        match &self.inner.color {
            Some(c) => Ok(PyTuple::new(py, c)?.into_any()),
            None => Ok(py.None().into_bound(py)),
        }
    }

    /// Whether `properties` has not been read: `inner` is then the class.
    fn is_untouched(&self) -> bool {
        self.held.is_untouched()
    }

    /// The class as it stands --- its `properties` as edited --- parsed again,
    /// so that every rule holds of it; and its JSON form.
    pub fn current(&self, py: Python<'_>) -> R<(engine::LabelClass, Value)> {
        let Some(properties) = self.held.get(0) else { return Ok((self.inner.clone(), self.inner.to_json())) };
        let mut doc = match self.inner.to_json() {
            Value::Object(m) => m,
            _ => Map::new(),
        };
        // Under the live views' lock, as a record's fields are read: an edit
        // nested in `properties` and its rewrite are one step under it.
        let properties =
            crate::document::under_live_lock_if(py, self.held.is_anchored(), || Ok(py_to_json(properties.bind(py))?))?;
        match properties {
            Value::Object(m) if m.is_empty() => {
                doc.shift_remove("properties");
            }
            v => {
                doc.insert("properties".into(), v);
            }
        }
        let class = engine::LabelClass::from_json(&Value::Object(doc))?;
        let json = class.to_json();
        Ok((class, json))
    }

    /// The engine value of a Python argument, and its JSON form: this class
    /// as it stands, or its JSON form.
    pub fn current_of(obj: &Bound<'_, PyAny>) -> R<(engine::LabelClass, Value)> {
        if let Ok(c) = obj.cast::<LabelClass>() {
            return c.get().current(obj.py());
        }
        let class = engine::LabelClass::from_json(&py_to_json(obj)?)?;
        let json = class.to_json();
        Ok((class, json))
    }

    /// The engine value of a Python argument: this class as it stands, or
    /// its JSON form.
    pub fn arg(obj: &Bound<'_, PyAny>) -> R<engine::LabelClass> {
        Ok(Self::current_of(obj)?.0)
    }
}

#[pymethods]
impl LabelClass {
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
    #[pyo3(signature = (id, key, name, parents=None, category=None, color=None, codes=None, laterality=None, properties=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        id: i64,
        key: String,
        name: String,
        parents: Option<Vec<i64>>,
        category: Option<String>,
        color: Option<Vec<i64>>,
        codes: Option<&Bound<'_, PyAny>>,
        laterality: Option<String>,
        properties: Option<&Bound<'_, PyAny>>,
    ) -> R<Self> {
        let properties = match properties {
            Some(p) if !p.is_none() => match py_to_json(p)? {
                Value::Object(m) => m,
                _ => Map::new(),
            },
            _ => Map::new(),
        };
        Ok(LabelClass::wrap(engine::LabelClass::build(
            id,
            key,
            name,
            parents.unwrap_or_default(),
            category,
            color,
            ontology_codes(codes)?,
            laterality,
            properties,
        )?))
    }
    #[getter]
    fn id(&self) -> i64 {
        self.inner.id
    }
    #[getter]
    fn key(&self) -> &str {
        &self.inner.key
    }
    #[getter]
    fn name(&self) -> &str {
        &self.inner.name
    }
    #[getter]
    fn parents<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.inner.parents)
    }
    #[getter]
    fn category(&self) -> Option<&str> {
        self.inner.category.as_deref()
    }
    #[getter]
    fn color<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.tuple_color(py)
    }
    #[getter]
    fn codes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.codes.iter().map(|c| OntologyCode(c.clone())))
    }
    #[getter]
    fn laterality(&self) -> Option<&str> {
        self.inner.laterality.as_deref()
    }
    /// Free-form properties: the class's own dict.
    #[getter]
    fn properties<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        let py = slf.py();
        let me = slf.get();
        if let Some(found) = me.held.get(0) {
            return Ok(found.bind(py).clone());
        }
        let plain = map_to_py(py, &me.inner.properties)?.into_any();
        let anchor = me.held.anchor(slf.as_any(), &me.inner.to_json(), None)?;
        Ok(me.held.keep(0, crate::records::containers(plain, anchor.as_ref())?))
    }
    /// Whether the class marks a lesion: `properties["is_lesion"]`, else its
    /// category.
    #[getter]
    fn is_lesion(&self, py: Python<'_>) -> R<bool> {
        Ok(self.current(py)?.0.is_lesion())
    }
    /// The class as it stands, as JSON.
    fn to_json<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.current(py)?.1)?)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Self> {
        Ok(LabelClass::wrap(engine::LabelClass::from_json(&py_to_json(doc)?)?))
    }
    // Equal as they stand; a class whose edits break a rule equals none.
    fn __eq__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> bool {
        let Ok(other) = other.cast::<LabelClass>() else { return false };
        match (self.current(py), other.get().current(py)) {
            (Ok(a), Ok(b)) => a.0 == b.0,
            _ => false,
        }
    }
    // Over the fields an edit cannot change, so that it holds while the class
    // lives; classes equal as they stand agree on those, so hash alike.
    fn __hash__(&self) -> isize {
        let mut fixed = self.inner.to_json();
        if let Value::Object(m) = &mut fixed {
            m.shift_remove("properties");
        }
        json_hash(&fixed)
    }
    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let py = slf.py();
        let me = slf.get();
        let fields = Self::FIELDS
            .iter()
            .map(|n| match (*n, me.held.get(0)) {
                ("properties", Some(found)) => Ok((*n, found.bind(py).clone())),
                ("properties", None) => Ok((*n, map_to_py(py, &me.inner.properties)?.into_any())),
                _ => Ok((*n, slf.getattr(*n)?)),
            })
            .collect::<PyResult<Vec<_>>>()?;
        dataclass_repr("LabelClass", &fields)
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        reduce_via_json(slf.as_any())
    }
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        self.held.traverse(&visit)
    }
}

// -- LabelSet ------------------------------------------------------------------------------

/// A controlled vocabulary (§5.1).
///
/// The classes are the set's own objects, as 1.x kept them --- those it was
/// given, and from the first read those it hands out (`ls[2] is
/// ls["liver"]`) --- so an edit to one's `properties` is the set's, digest
/// included, and for a document's label set, the document's.
#[pyclass(module = "medh5.labels", name = "LabelSet", skip_from_py_object, frozen)]
pub struct LabelSet {
    /// The vocabulary as given or read.
    pub inner: engine::LabelSet,
    /// The class objects, once given or read, in `inner`'s order.
    classes: OnceLock<Py<PyTuple>>,
    /// The document this is the label set of.
    document: Option<Py<PyAny>>,
}

/// `set` with its classes replaced by `classes` (the same ids and keys),
/// validated again.
pub fn with_classes(set: &engine::LabelSet, classes: Vec<engine::LabelClass>) -> medh5::Result<engine::LabelSet> {
    engine::LabelSet::new(
        set.id.clone(),
        classes,
        set.version.clone(),
        set.relations.clone(),
        set.skeletons.clone(),
        set.form.clone(),
        set.uri.clone(),
        set.declared_sha256().map(str::to_string),
    )
}

impl LabelSet {
    pub fn wrap(inner: engine::LabelSet) -> Self {
        LabelSet { inner, classes: OnceLock::new(), document: None }
    }

    /// The label set of `document`, read as `inner`.
    pub fn read_from(inner: engine::LabelSet, document: &Bound<'_, PyAny>) -> Self {
        LabelSet { inner, classes: OnceLock::new(), document: Some(document.clone().unbind()) }
    }

    /// The classes as Python objects, made once: `ls[2] is ls["liver"]`.
    fn class_objects<'py>(&self, py: Python<'py>) -> PyResult<&Bound<'py, PyTuple>> {
        if self.classes.get().is_none() {
            let made = self
                .inner
                .classes()
                .iter()
                .map(|c| match &self.document {
                    Some(document) => Py::new(py, LabelClass::read_from(c.clone(), document.bind(py), &c.key)),
                    None => Py::new(py, LabelClass::wrap(c.clone())),
                })
                .collect::<PyResult<Vec<_>>>()?;
            // A read on another thread may have got there first; either will do.
            let _ = self.classes.set(PyTuple::new(py, made)?.unbind());
        }
        Ok(self.classes.get().expect("made").bind(py))
    }

    fn class_object<'py>(&self, py: Python<'py>, class: &engine::LabelClass) -> PyResult<Bound<'py, PyAny>> {
        let position = self.inner.classes().iter().position(|c| c.id == class.id).unwrap_or(0);
        self.class_objects(py)?.get_item(position)
    }

    /// The vocabulary as it stands: its classes' `properties` as edited.
    pub fn current(&self, py: Python<'_>) -> R<engine::LabelSet> {
        let Some(objects) = self.classes.get() else { return Ok(self.inner.clone()) };
        let objects = objects.bind(py);
        if objects.iter().all(|c| c.cast::<LabelClass>().map(|c| c.get().is_untouched()).unwrap_or(true)) {
            return Ok(self.inner.clone());
        }
        // Every class in one step under the live views' lock, so that edits to
        // two classes' `properties` are read both or neither.
        let classes = crate::document::under_live_lock_if(py, self.document.is_some(), || {
            objects.iter().map(|c| LabelClass::arg(&c)).collect::<R<Vec<_>>>()
        })?;
        Ok(with_classes(&self.inner, classes)?)
    }
}

fn relations(items: Option<&Bound<'_, PyAny>>) -> R<Vec<engine::Relation>> {
    let mut out = Vec::new();
    let Some(items) = items else { return Ok(out) };
    for item in items.try_iter()? {
        let item = item?;
        match item.cast::<Relation>() {
            Ok(r) => out.push(r.get().0.clone()),
            Err(_) => out.push(engine::Relation::from_json(&py_to_json(&item)?)?),
        }
    }
    Ok(out)
}

fn skeletons(items: Option<&Bound<'_, PyAny>>) -> R<Vec<engine::Skeleton>> {
    let mut out = Vec::new();
    let Some(items) = items else { return Ok(out) };
    for item in items.try_iter()? {
        let item = item?;
        match item.cast::<Skeleton>() {
            Ok(s) => out.push(s.get().0.clone()),
            Err(_) => out.push(engine::Skeleton::from_json(&py_to_json(&item)?)?),
        }
    }
    Ok(out)
}

/// A label set argument: the binding class as it stands, or its JSON form.
pub fn label_set_arg(obj: &Bound<'_, PyAny>) -> R<engine::LabelSet> {
    if let Ok(ls) = obj.cast::<LabelSet>() {
        return ls.get().current(obj.py());
    }
    engine::LabelSet::from_json(Some(&py_to_json(obj)?))?
        .ok_or_else(|| medh5::Error::invalid("an empty label set document").into())
}

#[pymethods]
impl LabelSet {
    // The classes it is given are kept, in id order, as 1.x kept them.
    #[new]
    #[pyo3(signature = (id, classes=None, *, version="1.0.0", relations=None, skeletons=None, form="inline", uri=None, sha256=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        id: String,
        classes: Option<&Bound<'_, PyAny>>,
        version: &str,
        relations: Option<&Bound<'_, PyAny>>,
        skeletons: Option<&Bound<'_, PyAny>>,
        form: &str,
        uri: Option<String>,
        sha256: Option<String>,
    ) -> R<Self> {
        let mut given = Vec::new();
        if let Some(items) = classes {
            for item in items.try_iter()? {
                let item = item?;
                let value = LabelClass::arg(&item)?;
                let object = match item.cast::<LabelClass>() {
                    Ok(class) => class.clone().unbind(),
                    Err(_) => Py::new(py, LabelClass::wrap(value.clone()))?,
                };
                given.push((value, object));
            }
        }
        // Stable, as the engine's own sort is: the objects stay beside theirs.
        given.sort_by_key(|(value, _)| value.id);
        let (values, objects): (Vec<_>, Vec<_>) = given.into_iter().unzip();
        let made = LabelSet::wrap(engine::LabelSet::new(
            id,
            values,
            version.to_string(),
            self::relations(relations)?,
            self::skeletons(skeletons)?,
            form.to_string(),
            uri,
            sha256,
        )?);
        let _ = made.classes.set(PyTuple::new(py, objects)?.unbind());
        Ok(made)
    }
    #[getter]
    fn id(&self) -> &str {
        &self.inner.id
    }
    #[getter]
    fn version(&self) -> &str {
        &self.inner.version
    }
    #[getter]
    fn form(&self) -> &str {
        &self.inner.form
    }
    #[getter]
    fn uri(&self) -> Option<&str> {
        self.inner.uri.as_deref()
    }
    #[getter]
    fn relations<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.relations.iter().map(|r| Relation(r.clone())))
    }
    #[getter]
    fn skeletons<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.skeletons.iter().map(|s| Skeleton(s.clone())))
    }
    fn check(&self) -> R<()> {
        Ok(self.inner.check()?)
    }
    fn __len__(&self) -> usize {
        self.inner.len()
    }
    fn __iter__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        Ok(slf.getattr("classes")?.try_iter()?.into_any())
    }
    fn __contains__(&self, item: &Bound<'_, PyAny>) -> bool {
        if let Ok(s) = item.cast::<PyString>() {
            return self.inner.by_key(&s.to_string()).is_some();
        }
        match item.extract::<i64>() {
            Ok(i) => self.inner.contains_id(i),
            Err(_) => false,
        }
    }
    fn __repr__(&self) -> String {
        self.inner.repr()
    }
    #[getter]
    fn classes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        Ok(self.class_objects(py)?.clone())
    }
    #[getter]
    fn ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.ids())
    }
    #[getter]
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.keys())
    }
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> R<Bound<'py, PyAny>> {
        let key = lookup_key(key)?;
        Ok(self.class_object(py, self.inner.lookup(&key)?)?)
    }
    fn get<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Ok(key) = lookup_key(key) else { return Ok(None) };
        match self.inner.get(&key) {
            Some(c) => Ok(Some(self.class_object(py, c)?)),
            None => Ok(None),
        }
    }
    fn resolve<'py>(&self, py: Python<'py>, keys: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
        let keys = keys_arg(keys)?;
        let found = self.inner.resolve(&keys)?;
        let objects = found.iter().map(|c| self.class_object(py, c)).collect::<PyResult<Vec<_>>>()?;
        Ok(PyTuple::new(py, objects)?)
    }
    fn ids_for<'py>(&self, py: Python<'py>, keys: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
        let keys = keys_arg(keys)?;
        Ok(PyTuple::new(py, self.inner.ids_for(&keys)?)?)
    }
    fn missing<'py>(&self, py: Python<'py>, ids: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.inner.missing(crate::convert::class_ids(ids)?))
    }
    fn ancestors<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.inner.ancestors(&lookup_key(key)?)?)?)
    }
    fn descendants<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.inner.descendants(&lookup_key(key)?)?)?)
    }
    fn close<'py>(&self, py: Python<'py>, ids: &Bound<'py, PyAny>, closure: &str) -> R<Bound<'py, PyTuple>> {
        Ok(PyTuple::new(py, self.inner.close(&crate::convert::class_ids(ids)?, closure)?)?)
    }
    #[pyo3(signature = (key, predicate=None))]
    fn relations_of<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
        predicate: Option<&str>,
    ) -> R<Bound<'py, PyTuple>> {
        let found = self.inner.relations_of(&lookup_key(key)?, predicate)?;
        Ok(PyTuple::new(py, found.into_iter().map(|r| Relation(r.clone())))?)
    }
    fn skeleton(&self, skeleton_id: &str) -> R<Skeleton> {
        Ok(Skeleton(self.inner.skeleton(skeleton_id)?.clone()))
    }
    fn colors<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new(py);
        for (id, rgba) in self.inner.colors() {
            out.set_item(id, PyTuple::new(py, rgba)?)?;
        }
        Ok(out)
    }
    fn content_doc<'py>(&self, py: Python<'py>) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.current(py)?.content_doc())?)
    }
    #[pyo3(signature = (algo="sha256"))]
    fn digest(&self, py: Python<'_>, algo: &str) -> R<String> {
        Ok(self.current(py)?.digest(algo)?)
    }
    #[getter]
    fn sha256(&self, py: Python<'_>) -> R<String> {
        Ok(self.current(py)?.sha256())
    }
    /// The vocabulary as it stands, as JSON.
    #[pyo3(signature = (*, form=None))]
    fn to_json<'py>(&self, py: Python<'py>, form: Option<&str>) -> R<Bound<'py, PyAny>> {
        Ok(json_to_py(py, &self.current(py)?.to_json(form))?)
    }
    #[classmethod]
    fn from_json(_cls: &Bound<'_, pyo3::types::PyType>, doc: &Bound<'_, PyAny>) -> R<Option<Self>> {
        if doc.is_none() {
            return Ok(None);
        }
        Ok(engine::LabelSet::from_json(Some(&py_to_json(doc)?))?.map(LabelSet::wrap))
    }
    fn as_ref(&self, py: Python<'_>, uri: &str) -> R<LabelSet> {
        Ok(LabelSet::wrap(self.current(py)?.as_ref(uri)?))
    }
    #[pyo3(signature = (keys, *, id=None))]
    fn subset(&self, py: Python<'_>, keys: &Bound<'_, PyAny>, id: Option<&str>) -> R<LabelSet> {
        Ok(LabelSet::wrap(self.current(py)?.subset(&keys_arg(keys)?, id)?))
    }
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        reduce_via_json(slf.as_any())
    }
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        visit.call(self.classes.get())?;
        visit.call(&self.document)
    }
}

/// `ls[key]`: an `int` looks up an id, anything else a key.
fn lookup_key(key: &Bound<'_, PyAny>) -> PyResult<ClassKey> {
    if let Ok(s) = key.cast::<PyString>() {
        return Ok(ClassKey::Key(s.to_string()));
    }
    match key.extract::<i64>() {
        Ok(i) => Ok(ClassKey::Id(i)),
        Err(_) => Ok(ClassKey::Key(key.str()?.to_string())),
    }
}

fn keys_arg(keys: &Bound<'_, PyAny>) -> PyResult<Vec<ClassKey>> {
    let mut out = Vec::new();
    for item in keys.try_iter()? {
        out.push(lookup_key(&item?)?);
    }
    Ok(out)
}

// -- module functions ------------------------------------------------------------------------

/// Return the canonical JSON bytes of a document: sorted keys, no whitespace, UTF-8 (§5.1).
///
/// These are the bytes a label set's digest is computed over; the encoding is
/// defined to the byte.
#[pyfunction]
fn canonical_json<'py>(py: Python<'py>, doc: &Bound<'py, PyAny>) -> PyResult<Bound<'py, pyo3::types::PyBytes>> {
    Ok(pyo3::types::PyBytes::new(py, &engine::canonical_json(&py_to_json(doc)?)))
}

/// Mint an inline `LabelSet` from bare class keys, with ids counting up from `start`.
///
/// Each class is named from its key in title case, underscores as spaces
/// (`left_kidney` becomes `Left Kidney`): what the converters do on ingest. Ids
/// are checked as §5.3 requires.
#[pyfunction]
#[pyo3(signature = (keys, *, id, version="1.0.0", start=1))]
fn from_keys(keys: Vec<String>, id: &str, version: &str, start: i64) -> R<LabelSet> {
    Ok(LabelSet::wrap(engine::from_keys(&keys, id, version, start)?))
}

#[pyfunction]
fn registry_available<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, registry::available())
}

#[pyfunction]
fn registry_load(name: &str) -> R<LabelSet> {
    Ok(LabelSet::wrap(registry::load(name)?))
}

#[pyfunction]
fn registry_load_file(path: std::path::PathBuf) -> R<LabelSet> {
    Ok(LabelSet::wrap(registry::load_file(&path)?))
}

#[pyfunction]
fn registry_register(name: &str, label_set: &Bound<'_, PyAny>) -> R<LabelSet> {
    Ok(LabelSet::wrap(registry::register(name, label_set_arg(label_set)?)))
}

#[pyfunction]
fn registry_unregister(name: &str) {
    registry::unregister(name)
}

#[pyfunction]
fn registry_describe<'py>(py: Python<'py>) -> R<Bound<'py, PyDict>> {
    Ok(map_to_py(py, &registry::describe()?)?)
}

/// Sample files grouped by vocabulary, as `medh5 labels check --json` writes it.
#[pyfunction]
fn labels_check<'py>(py: Python<'py>, paths: Vec<std::path::PathBuf>) -> R<Bound<'py, PyAny>> {
    let found = py.detach(move || engine::check_vocabularies(&paths));
    Ok(json_to_py(py, &found.to_json())?)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<OntologyCode>()?;
    m.add_class::<Relation>()?;
    m.add_class::<Skeleton>()?;
    m.add_class::<LabelClass>()?;
    m.add_class::<LabelSet>()?;
    m.add_function(wrap_pyfunction!(canonical_json, m)?)?;
    m.add_function(wrap_pyfunction!(from_keys, m)?)?;
    m.add_function(wrap_pyfunction!(labels_check, m)?)?;
    m.add_function(wrap_pyfunction!(registry_available, m)?)?;
    m.add_function(wrap_pyfunction!(registry_load, m)?)?;
    m.add_function(wrap_pyfunction!(registry_load_file, m)?)?;
    m.add_function(wrap_pyfunction!(registry_register, m)?)?;
    m.add_function(wrap_pyfunction!(registry_unregister, m)?)?;
    m.add_function(wrap_pyfunction!(registry_describe, m)?)?;
    m.add("BACKGROUND_ID", engine::BACKGROUND_ID)?;
    m.add("IGNORE_ID", engine::IGNORE_ID)?;
    m.add("MAX_CLASS_ID", engine::MAX_CLASS_ID)?;
    m.add("CLOSURES", PyTuple::new(m.py(), engine::CLOSURES)?)?;
    m.add("FORMS", PyTuple::new(m.py(), engine::FORMS)?)?;
    m.add("INLINE_REQUIRED_BELOW", engine::INLINE_REQUIRED_BELOW)?;
    Ok(())
}
